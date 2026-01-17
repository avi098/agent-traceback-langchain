import boto3
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging
from collections import Counter

# FastAPI imports (for API mode)
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BedrockAgentTracer:
    """Captures and saves agent traces in CloudWatch-compatible format"""
    
    def __init__(self, agent_id: str, agent_alias_id: str, agent_version: str):
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.agent_version = agent_version
        self.traces = []
        self.session_id = None
        self.trace_counter = Counter()
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_filename = f"agent_traces_{timestamp}.json"
        self.log_filename = f"agent_logs_{timestamp}.log"
        
        # Setup log file handler
        self.log_handler = logging.FileHandler(self.log_filename)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(self.log_handler)
        
        # Log initialization
        logger.info("Found credentials in environment variables.")
        logger.info("=" * 80)
        logger.info("BEDROCK AGENT CLIENT INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Region: us-east-1")
        logger.info(f"Agent ID: {agent_id}")
        logger.info(f"Agent Alias ID: {agent_alias_id}")
        logger.info(f"Agent Version: {agent_version}")
        logger.info(f"Trace File: {self.trace_filename}")
        logger.info(f"Log File: {self.log_filename}")
        logger.info("=" * 80)
    
    def create_trace_event(self, trace_data: Dict, event_time: str = None) -> Dict:
        """Create a CloudWatch-compatible trace event"""
        if event_time is None:
            event_time = datetime.utcnow().isoformat() + 'Z'
        
        return {
            "agentAliasId": self.agent_alias_id,
            "agentId": self.agent_id,
            "agentVersion": self.agent_version,
            "callerChain": [
                {
                    "agentAliasArn": f"arn:aws:bedrock:us-east-1:{os.getenv('AWS_ACCOUNT_ID', '000000000000')}:agent-alias/{self.agent_id}/{self.agent_alias_id}"
                }
            ],
            "eventTime": event_time,
            "sessionId": self.session_id,
            "trace": trace_data
        }
    
    def add_trace(self, trace_event: Dict):
        """Add trace event to collection"""
        self.traces.append(trace_event)
        
        # Count trace types
        trace_type = self._get_trace_type(trace_event)
        if trace_type:
            self.trace_counter[trace_type] += 1
    
    def _get_trace_type(self, trace_event: Dict) -> Optional[str]:
        """Extract trace type from trace event"""
        trace = trace_event.get('trace', {}).get('orchestrationTrace', {})
        if 'modelInvocationInput' in trace:
            return 'modelInvocationInput'
        elif 'modelInvocationOutput' in trace:
            return 'modelInvocationOutput'
        elif 'rationale' in trace:
            return 'rationale'
        elif 'invocationInput' in trace:
            return 'invocationInput'
        elif 'observation' in trace:
            return 'observation'
        return None
    
    def save_traces(self):
        """Save all traces to JSON file"""
        if self.traces:
            with open(self.trace_filename, 'w') as f:
                json.dump(self.traces, f, indent=2)
            logger.info(f"[TRACE] Saved {len(self.traces)} traces to {self.trace_filename}")
            logger.info(f"[TRACE] Total traces in file: {len(self.traces)}")
        else:
            logger.warning("[TRACE] No traces to save")
    
    def print_trace_summary(self):
        """Print trace summary"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TRACE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Total Traces: {len(self.traces)}")
        logger.info(f"Trace File: {self.trace_filename}")
        logger.info("")
        logger.info("Trace Type Breakdown:")
        for trace_type, count in sorted(self.trace_counter.items()):
            logger.info(f"  - {trace_type}: {count}")
        logger.info("=" * 80)


class BedrockAgentClient:
    """Client for interacting with AWS Bedrock Agent with KB support"""
    
    def __init__(
        self,
        agent_id: str,
        agent_alias_id: str,
        agent_version: str = "DRAFT",
        region: str = "us-east-1"
    ):
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.agent_version = agent_version
        self.region = region
        
        # Initialize Bedrock Agent Runtime client
        self.client = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=region
        )
        
        # Initialize tracer
        self.tracer = BedrockAgentTracer(agent_id, agent_alias_id, agent_version)
    
    def invoke_agent(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        end_session: bool = False
    ) -> Dict:
        """Invoke the Bedrock agent with a prompt"""
        # Generate session ID if not provided
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        self.tracer.session_id = session_id
        
        # Log flow step 1
        logger.info("")
        logger.info("=" * 80)
        logger.info("[FLOW STEP 1] USER QUERY RECEIVED")
        logger.info("=" * 80)
        logger.info(f"Query: {prompt}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        try:
            # Log flow step 2
            logger.info("=" * 80)
            logger.info("[FLOW STEP 2] INVOKING BEDROCK AGENT")
            logger.info("=" * 80)
            logger.info("Calling AWS Bedrock Agent Runtime API")
            logger.info("Enable Trace: True")
            logger.info("=" * 80)
            
            # Invoke agent with trace enabled
            response = self.client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
                enableTrace=True,
                endSession=end_session
            )
            
            # Log flow step 3
            logger.info("=" * 80)
            logger.info("[FLOW STEP 3] PROCESSING AGENT RESPONSE STREAM")
            logger.info("=" * 80)
            
            # Process the event stream
            result = self._process_event_stream(response['completion'])
            
            # Log flow step 4
            logger.info("")
            logger.info("=" * 80)
            logger.info("[FLOW STEP 4] AGENT RESPONSE COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Response Length: {len(result['response'])} characters")
            logger.info(f"Traces Captured: {len(self.tracer.traces)}")
            logger.info("=" * 80)
            
            # Log final response
            logger.info("=" * 80)
            logger.info("FINAL RESPONSE")
            logger.info("=" * 80)
            logger.info(result['response'])
            logger.info("=" * 80)
            
            # Save traces to file
            self.tracer.save_traces()
            
            # Print trace summary
            self.tracer.print_trace_summary()
            
            return result
        
        except Exception as e:
            logger.error(f"Error invoking agent: {str(e)}")
            raise
    
    def _process_event_stream(self, event_stream) -> Dict:
        """Process the agent response event stream"""
        response_text = ""
        metadata = {
            "chunks_received": 0,
            "traces_captured": 0,
            "kb_lookups": 0,
            "tool_calls": 0
        }
        
        try:
            for event in event_stream:
                # Handle response chunks
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        chunk_text = chunk['bytes'].decode('utf-8')
                        response_text += chunk_text
                        metadata["chunks_received"] += 1
                        
                        # Log chunk received
                        preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                        logger.info(f"[CHUNK] Received: {preview}")
                
                # Handle trace events
                elif 'trace' in event:
                    trace_data = event['trace']
                    self._process_trace(trace_data)
                    metadata["traces_captured"] += 1
                    
                    # Count specific trace types
                    if 'orchestrationTrace' in trace_data.get('trace', {}):
                        orch_trace = trace_data['trace']['orchestrationTrace']
                        
                        # Knowledge Base lookup
                        if 'observation' in orch_trace:
                            obs = orch_trace['observation']
                            if obs.get('type') == 'KNOWLEDGE_BASE':
                                metadata["kb_lookups"] += 1
                        
                        # Tool/Action Group invocation
                        if 'invocationInput' in orch_trace:
                            inv_input = orch_trace['invocationInput']
                            if inv_input.get('invocationType') == 'ACTION_GROUP':
                                metadata["tool_calls"] += 1
        
        except Exception as e:
            logger.error(f"Error processing event stream: {str(e)}")
            raise
        
        return {
            "response": response_text,
            "metadata": metadata
        }
    
    def _process_trace(self, trace_data: Dict):
        """Process and log a trace event"""
        trace = trace_data.get('trace', {})
        
        # Handle orchestration trace
        if 'orchestrationTrace' in trace:
            orch_trace = trace['orchestrationTrace']
            event_time = trace_data.get('timestamp', datetime.utcnow().isoformat() + 'Z')
            
            # Model invocation input
            if 'modelInvocationInput' in orch_trace:
                logger.info("[TRACE] Added modelInvocationInput trace")
                trace_event = self.tracer.create_trace_event(
                    {"orchestrationTrace": {"modelInvocationInput": orch_trace['modelInvocationInput']}},
                    event_time
                )
                self.tracer.add_trace(trace_event)
            
            # Model invocation output
            if 'modelInvocationOutput' in orch_trace:
                logger.info("")
                logger.info("*" * 80)
                logger.info("[TRACE] MODEL INVOCATION OUTPUT")
                logger.info("*" * 80)
                
                output = orch_trace['modelInvocationOutput']
                metadata = output.get('metadata', {})
                usage = metadata.get('usage', {})
                
                logger.info(f"Start Time: N/A")
                logger.info(f"End Time: N/A")
                logger.info(f"Total Time: N/A ms")
                logger.info(f"Input Tokens: {usage.get('inputTokens', 0)}")
                logger.info(f"Output Tokens: {usage.get('outputTokens', 0)}")
                
                # Get raw response for stop reason
                raw_response = output.get('rawResponse', {})
                stop_reason = raw_response.get('stopReason', 'N/A')
                logger.info(f"Stop Reason: {stop_reason}")
                
                # Model ID
                trace_id = output.get('traceId', '')
                if 'claude' in trace_id.lower():
                    logger.info(f"Model: claude-3-5-sonnet-20240620")
                else:
                    logger.info(f"Model: N/A")
                
                logger.info("*" * 80)
                logger.info("[TRACE] Added modelInvocationOutput trace")
                
                trace_event = self.tracer.create_trace_event(
                    {"orchestrationTrace": {"modelInvocationOutput": output}},
                    event_time
                )
                self.tracer.add_trace(trace_event)
            
            # Rationale
            if 'rationale' in orch_trace:
                logger.info("")
                logger.info("~" * 80)
                logger.info("[TRACE] AGENT RATIONALE")
                logger.info("~" * 80)
                
                rationale_text = orch_trace['rationale'].get('text', '')
                logger.info(f"Reasoning: {rationale_text}")
                logger.info("~" * 80)
                logger.info("[TRACE] Added rationale trace")
                
                trace_event = self.tracer.create_trace_event(
                    {"orchestrationTrace": {"rationale": orch_trace['rationale']}},
                    event_time
                )
                self.tracer.add_trace(trace_event)
            
            # Invocation input (Tool or KB)
            if 'invocationInput' in orch_trace:
                inv_input = orch_trace['invocationInput']
                inv_type = inv_input.get('invocationType', 'UNKNOWN')
                
                if inv_type == 'ACTION_GROUP':
                    logger.info("")
                    logger.info("+" * 80)
                    logger.info("[TRACE] INVOCATION INPUT - TOOL CALL")
                    logger.info("+" * 80)
                    
                    action_input = inv_input.get('actionGroupInvocationInput', {})
                    logger.info(f"Action Group: {action_input.get('actionGroupName', 'N/A')}")
                    logger.info(f"API Path: {action_input.get('apiPath', 'N/A')}")
                    logger.info(f"HTTP Method: {action_input.get('verb', 'N/A')}")
                    logger.info(f"Execution Type: {action_input.get('executionType', 'N/A')}")
                    
                    # Request body
                    request_body = action_input.get('requestBody', {})
                    if request_body:
                        logger.info(f"Request Body: {json.dumps(request_body, indent=2)}")
                    
                    logger.info("+" * 80)
                    logger.info("[TRACE] Added invocationInput trace")
                    
                elif inv_type == 'KNOWLEDGE_BASE':
                    logger.info("")
                    logger.info("+" * 80)
                    logger.info("[TRACE] INVOCATION INPUT - KNOWLEDGE BASE")
                    logger.info("+" * 80)
                    
                    kb_input = inv_input.get('knowledgeBaseLookupInput', {})
                    logger.info(f"KB ID: {kb_input.get('knowledgeBaseId', 'N/A')}")
                    logger.info(f"Query Text: {kb_input.get('text', 'N/A')}")
                    
                    logger.info("+" * 80)
                    logger.info("[TRACE] Added invocationInput trace")
                
                trace_event = self.tracer.create_trace_event(
                    {"orchestrationTrace": {"invocationInput": inv_input}},
                    event_time
                )
                self.tracer.add_trace(trace_event)
            
            # Observation (Tool result or KB result)
            if 'observation' in orch_trace:
                obs = orch_trace['observation']
                obs_type = obs.get('type', 'UNKNOWN')
                
                logger.info("")
                logger.info("-" * 80)
                logger.info("[TRACE] OBSERVATION - TOOL RESPONSE")
                logger.info("-" * 80)
                logger.info(f"Observation Type: {obs_type}")
                
                if obs_type == 'ACTION_GROUP':
                    action_output = obs.get('actionGroupInvocationOutput', {})
                    response_text = action_output.get('text', '')
                    preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                    logger.info(f"Response Text: {preview}")
                
                elif obs_type == 'KNOWLEDGE_BASE':
                    kb_output = obs.get('knowledgeBaseLookupOutput', {})
                    refs = kb_output.get('retrievedReferences', [])
                    logger.info(f"Retrieved References: {len(refs)}")
                    
                    # Log first reference
                    if refs:
                        first_ref = refs[0]
                        content = first_ref.get('content', {}).get('text', '')
                        location = first_ref.get('location', {})
                        s3_loc = location.get('s3Location', {}).get('uri', 'Unknown')
                        logger.info(f"Source: {s3_loc}")
                        preview = content[:200] + "..." if len(content) > 200 else content
                        logger.info(f"Content: {preview}")
                
                elif obs_type == 'FINISH':
                    final_response = obs.get('finalResponse', {})
                    response_text = final_response.get('text', '')
                    preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                    logger.info(f"Final Response Text: {preview}")
                
                logger.info("-" * 80)
                logger.info("[TRACE] Added observation trace")
                
                trace_event = self.tracer.create_trace_event(
                    {"orchestrationTrace": {"observation": obs}},
                    event_time
                )
                self.tracer.add_trace(trace_event)


# ============================================================================
# CLI Mode
# ============================================================================

def run_cli():
    """Run in CLI mode for interactive testing"""
    print("\nAWS Bedrock Agent CLI with Knowledge Base Support")
    print("-" * 60)
    
    # Load configuration from environment
    agent_id = os.getenv('BEDROCK_AGENT_ID')
    agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
    agent_version = os.getenv('BEDROCK_AGENT_VERSION', 'DRAFT')
    
    if not agent_id or not agent_alias_id:
        print("Error: Missing required environment variables")
        print("Please set BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID in .env file")
        sys.exit(1)
    
    # Initialize client
    client = BedrockAgentClient(
        agent_id=agent_id,
        agent_alias_id=agent_alias_id,
        agent_version=agent_version
    )
    
    # Interactive loop
    session_id = None
    
    while True:
        try:
            print("\n" + "-" * 60)
            query = input("Your Query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            # Invoke agent
            result = client.invoke_agent(query, session_id=session_id)
            
            # Print response
            print("\nAGENT RESPONSE:")
            print("-" * 60)
            print(result['response'])
            print("-" * 60)
            
            # Update session ID for conversation continuity
            if session_id is None:
                session_id = client.tracer.session_id
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.exception("Detailed error:")


# ============================================================================
# API Mode (FastAPI)
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(title="Bedrock Agent API with KB")
    
    # Global client instance
    agent_client = None
    
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
        end_session: bool = False
    
    class QueryResponse(BaseModel):
        response: str
        session_id: str
        metadata: Dict
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize agent client on startup"""
        global agent_client
        
        agent_id = os.getenv('BEDROCK_AGENT_ID')
        agent_alias_id = os.getenv('BEDROCK_AGENT_ALIAS_ID')
        agent_version = os.getenv('BEDROCK_AGENT_VERSION', 'DRAFT')
        
        if not agent_id or not agent_alias_id:
            raise RuntimeError("Missing BEDROCK_AGENT_ID or BEDROCK_AGENT_ALIAS_ID")
        
        agent_client = BedrockAgentClient(
            agent_id=agent_id,
            agent_alias_id=agent_alias_id,
            agent_version=agent_version
        )
        
        logger.info("API server started with KB support")
    
    @app.post("/query", response_model=QueryResponse)
    async def query_agent(request: QueryRequest):
        """Query the Bedrock agent"""
        try:
            result = agent_client.invoke_agent(
                prompt=request.query,
                session_id=request.session_id,
                end_session=request.end_session
            )
            
            return QueryResponse(
                response=result['response'],
                session_id=agent_client.tracer.session_id,
                metadata=result['metadata']
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "agent_id": os.getenv('BEDROCK_AGENT_ID'),
            "kb_support": True
        }
    
    @app.get("/traces")
    async def get_traces():
        """Get current session traces"""
        if agent_client and agent_client.tracer.traces:
            return {
                "traces": agent_client.tracer.traces,
                "count": len(agent_client.tracer.traces)
            }
        return {"traces": [], "count": 0}
    
    def run_api():
        """Run in API mode"""
        port = int(os.getenv('API_PORT', 8080))
        uvicorn.run(app, host="0.0.0.0", port=port)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        if not FASTAPI_AVAILABLE:
            print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
            sys.exit(1)
        print("Starting API server with Knowledge Base support...")
        run_api()
    else:
        run_cli()


if __name__ == "__main__":
    main()