import os
import json
import boto3
import uuid
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_filename = f"agent_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
trace_filename = f"agent_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)


class BedrockAgentTracer:
    """Traces Bedrock Agent execution and saves to local file in CloudWatch format"""
    
    def __init__(self, agent_id: str, agent_alias_id: str, agent_version: str, region: str):
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.agent_version = agent_version
        self.region = region
        self.agent_alias_arn = f"arn:aws:bedrock:{region}:{self._get_account_id()}:agent-alias/{agent_id}/{agent_alias_id}"
        self.traces = []
        self.session_id = None
    
    def _get_account_id(self) -> str:
        """Get AWS account ID from STS"""
        try:
            sts = boto3.client('sts', region_name=self.region)
            return sts.get_caller_identity()['Account']
        except:
            return "762233739050"  # Fallback to known account ID
    
    def start_session(self):
        """Start a new tracing session"""
        self.session_id = f"{self._get_account_id()}{str(uuid.uuid4().int)[:6]}"
        self.traces = []
        logger.info(f"[TRACE] Started new session: {self.session_id}")
    
    def add_trace(self, trace_type: str, trace_data: Dict[str, Any], trace_id: Optional[str] = None):
        """Add a trace event in CloudWatch format"""
        if trace_id is None:
            trace_id = f"{uuid.uuid4()}-0"
        
        trace_event = {
            "agentAliasId": self.agent_alias_id,
            "agentId": self.agent_id,
            "agentVersion": self.agent_version,
            "callerChain": [
                {
                    "agentAliasArn": self.agent_alias_arn
                }
            ],
            "eventTime": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "sessionId": self.session_id,
            "trace": {
                "orchestrationTrace": {
                    trace_type: {
                        **trace_data,
                        "traceId": trace_id
                    }
                }
            }
        }
        
        self.traces.append(trace_event)
        logger.info(f"[TRACE] Added {trace_type} trace")
    
    def save_traces(self):
        """Save all traces to JSON file"""
        try:
            # Load existing traces if file exists
            existing_traces = []
            if os.path.exists(trace_filename):
                with open(trace_filename, 'r', encoding='utf-8') as f:
                    existing_traces = json.load(f)
            
            # Append new traces
            all_traces = existing_traces + self.traces
            
            # Save to file
            with open(trace_filename, 'w', encoding='utf-8') as f:
                json.dump(all_traces, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[TRACE] Saved {len(self.traces)} traces to {trace_filename}")
            logger.info(f"[TRACE] Total traces in file: {len(all_traces)}")
        except Exception as e:
            logger.error(f"[TRACE] Failed to save traces: {str(e)}")
    
    def log_trace_summary(self):
        """Log a summary of all traces"""
        logger.info("\n" + "="*80)
        logger.info("TRACE SUMMARY")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Total Traces: {len(self.traces)}")
        logger.info(f"Trace File: {trace_filename}")
        
        # Count trace types
        trace_types = {}
        for trace in self.traces:
            trace_type = list(trace['trace']['orchestrationTrace'].keys())[0]
            trace_types[trace_type] = trace_types.get(trace_type, 0) + 1
        
        logger.info("\nTrace Type Breakdown:")
        for trace_type, count in trace_types.items():
            logger.info(f"  - {trace_type}: {count}")
        
        logger.info("="*80 + "\n")


class BedrockAgentClient:
    """Client for interacting with AWS Bedrock Agent Runtime"""
    
    def __init__(self):
        # AWS Configuration
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.agent_id = os.getenv('AGENT_ID', 'PKK8N53FKR')
        self.agent_alias_id = os.getenv('AGENT_ALIAS_ID', 'YLOMGJD7RP')
        self.agent_version = os.getenv('AGENT_VERSION', '4')
        
        # Initialize Bedrock Agent Runtime client
        self.bedrock_agent_runtime = boto3.client(
            'bedrock-agent-runtime',
            region_name=self.region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Initialize tracer
        self.tracer = BedrockAgentTracer(
            self.agent_id,
            self.agent_alias_id,
            self.agent_version,
            self.region
        )
        
        logger.info("="*80)
        logger.info("BEDROCK AGENT CLIENT INITIALIZATION")
        logger.info("="*80)
        logger.info(f"Region: {self.region}")
        logger.info(f"Agent ID: {self.agent_id}")
        logger.info(f"Agent Alias ID: {self.agent_alias_id}")
        logger.info(f"Agent Version: {self.agent_version}")
        logger.info(f"Trace File: {trace_filename}")
        logger.info(f"Log File: {log_filename}")
        logger.info("="*80 + "\n")
    
    def invoke_agent(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke Bedrock Agent and capture detailed traces
        
        Args:
            query: User query string
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dictionary containing response and trace information
        """
        logger.info("\n" + "="*80)
        logger.info("[FLOW STEP 1] USER QUERY RECEIVED")
        logger.info("="*80)
        logger.info(f"Query: {query}")
        logger.info(f"Session ID: {session_id or 'NEW SESSION'}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")
        
        # Start new tracing session
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.tracer.start_session()
        
        try:
            # Create trace ID
            trace_id = f"{uuid.uuid4()}-0"
            
            # Log model invocation input
            logger.info("="*80)
            logger.info("[FLOW STEP 2] INVOKING BEDROCK AGENT")
            logger.info("="*80)
            logger.info(f"Calling AWS Bedrock Agent Runtime API")
            logger.info(f"Enable Trace: True")
            logger.info("="*80 + "\n")
            
            # Add modelInvocationInput trace
            self.tracer.add_trace(
                "modelInvocationInput",
                {
                    "foundationModel": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "inferenceConfiguration": {
                        "maximumLength": 2048,
                        "stopSequences": ["</invoke>", "</answer>", "</error>"],
                        "temperature": 0,
                        "topK": 250,
                        "topP": 1
                    },
                    "text": json.dumps({
                        "messages": [
                            {
                                "content": f"[{{text={query}, type=text}}]",
                                "role": "user"
                            }
                        ]
                    }),
                    "type": "ORCHESTRATION"
                },
                trace_id
            )
            
            # Invoke the agent with enable_trace=True
            response = self.bedrock_agent_runtime.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=query,
                enableTrace=True  # Enable tracing
            )
            
            # Process response and traces
            completion = ""
            traces_captured = []
            
            logger.info("="*80)
            logger.info("[FLOW STEP 3] PROCESSING AGENT RESPONSE STREAM")
            logger.info("="*80 + "\n")
            
            event_stream = response.get('completion', [])
            
            for event in event_stream:
                # Handle chunk events (actual response text)
                if 'chunk' in event:
                    chunk_data = event['chunk']
                    if 'bytes' in chunk_data:
                        chunk_text = chunk_data['bytes'].decode('utf-8')
                        completion += chunk_text
                        logger.info(f"[CHUNK] Received: {chunk_text[:100]}...")
                
                # Handle trace events
                if 'trace' in event:
                    trace_event = event['trace']
                    traces_captured.append(trace_event)
                    
                    # Log trace details
                    if 'trace' in trace_event:
                        trace_data = trace_event['trace']
                        
                        # Extract orchestration trace
                        if 'orchestrationTrace' in trace_data:
                            orch_trace = trace_data['orchestrationTrace']
                            
                            # Model invocation output
                            if 'modelInvocationOutput' in orch_trace:
                                logger.info("\n" + "*"*80)
                                logger.info("[TRACE] MODEL INVOCATION OUTPUT")
                                logger.info("*"*80)
                                
                                output = orch_trace['modelInvocationOutput']
                                
                                # Log metadata
                                if 'metadata' in output:
                                    metadata = output['metadata']
                                    logger.info(f"Start Time: {metadata.get('startTime', 'N/A')}")
                                    logger.info(f"End Time: {metadata.get('endTime', 'N/A')}")
                                    logger.info(f"Total Time: {metadata.get('totalTimeMs', 'N/A')} ms")
                                    
                                    if 'usage' in metadata:
                                        usage = metadata['usage']
                                        logger.info(f"Input Tokens: {usage.get('inputTokens', 0)}")
                                        logger.info(f"Output Tokens: {usage.get('outputTokens', 0)}")
                                
                                # Log raw response
                                if 'rawResponse' in output:
                                    raw = output['rawResponse']
                                    if 'content' in raw:
                                        raw_content = json.loads(raw['content']) if isinstance(raw['content'], str) else raw['content']
                                        logger.info(f"Stop Reason: {raw_content.get('stop_reason', 'N/A')}")
                                        logger.info(f"Model: {raw_content.get('model', 'N/A')}")
                                
                                logger.info("*"*80 + "\n")
                                
                                # Add to tracer
                                self.tracer.add_trace("modelInvocationOutput", output, trace_id)
                            
                            # Rationale
                            if 'rationale' in orch_trace:
                                logger.info("\n" + "~"*80)
                                logger.info("[TRACE] AGENT RATIONALE")
                                logger.info("~"*80)
                                
                                rationale = orch_trace['rationale']
                                logger.info(f"Reasoning: {rationale.get('text', 'N/A')}")
                                logger.info("~"*80 + "\n")
                                
                                # Add to tracer
                                self.tracer.add_trace("rationale", rationale, trace_id)
                            
                            # Invocation Input (Tool Call)
                            if 'invocationInput' in orch_trace:
                                logger.info("\n" + "+"*80)
                                logger.info("[TRACE] INVOCATION INPUT - TOOL CALL")
                                logger.info("+"*80)
                                
                                inv_input = orch_trace['invocationInput']
                                
                                if 'actionGroupInvocationInput' in inv_input:
                                    action_input = inv_input['actionGroupInvocationInput']
                                    logger.info(f"Action Group: {action_input.get('actionGroupName', 'N/A')}")
                                    logger.info(f"API Path: {action_input.get('apiPath', 'N/A')}")
                                    logger.info(f"HTTP Method: {action_input.get('verb', 'N/A')}")
                                    logger.info(f"Execution Type: {action_input.get('executionType', 'N/A')}")
                                    
                                    if 'requestBody' in action_input:
                                        req_body = action_input['requestBody']
                                        logger.info(f"Request Body: {json.dumps(req_body, indent=2)}")
                                
                                logger.info("+"*80 + "\n")
                                
                                # Add to tracer
                                self.tracer.add_trace("invocationInput", inv_input, trace_id)
                            
                            # Observation (Tool Response)
                            if 'observation' in orch_trace:
                                logger.info("\n" + "-"*80)
                                logger.info("[TRACE] OBSERVATION - TOOL RESPONSE")
                                logger.info("-"*80)
                                
                                observation = orch_trace['observation']
                                obs_type = observation.get('type', 'UNKNOWN')
                                logger.info(f"Observation Type: {obs_type}")
                                
                                if 'actionGroupInvocationOutput' in observation:
                                    action_output = observation['actionGroupInvocationOutput']
                                    logger.info(f"Response Text: {action_output.get('text', 'N/A')[:200]}...")
                                    
                                    if 'metadata' in action_output:
                                        metadata = action_output['metadata']
                                        logger.info(f"Start Time: {metadata.get('startTime', 'N/A')}")
                                        logger.info(f"End Time: {metadata.get('endTime', 'N/A')}")
                                        logger.info(f"Total Time: {metadata.get('totalTimeMs', 'N/A')} ms")
                                
                                if 'finalResponse' in observation:
                                    final_resp = observation['finalResponse']
                                    logger.info(f"Final Response Text: {final_resp.get('text', 'N/A')[:200]}...")
                                
                                logger.info("-"*80 + "\n")
                                
                                # Add to tracer
                                self.tracer.add_trace("observation", observation, trace_id)
            
            # Log final response
            logger.info("\n" + "="*80)
            logger.info("[FLOW STEP 4] AGENT RESPONSE COMPLETE")
            logger.info("="*80)
            logger.info(f"Response Length: {len(completion)} characters")
            logger.info(f"Traces Captured: {len(traces_captured)}")
            logger.info("="*80 + "\n")
            
            logger.info("="*80)
            logger.info("FINAL RESPONSE")
            logger.info("="*80)
            logger.info(completion)
            logger.info("="*80 + "\n")
            
            # Save traces to file
            self.tracer.save_traces()
            self.tracer.log_trace_summary()
            
            return {
                "response": completion,
                "session_id": session_id,
                "traces_captured": len(traces_captured),
                "trace_file": trace_filename,
                "log_file": log_filename
            }
        
        except Exception as e:
            logger.error(f"\n{'!'*80}")
            logger.error(f"[ERROR] AGENT INVOCATION FAILED")
            logger.error(f"{'!'*80}")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {str(e)}")
            logger.error(f"{'!'*80}\n")
            
            # Save traces even on error
            self.tracer.save_traces()
            
            raise


# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")


class QueryResponse(BaseModel):
    status: str
    query: str
    response: str
    session_id: str
    traces_captured: int
    trace_file: str
    log_file: str
    timestamp: str


# Global agent client
agent_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_client
    logger.info("="*80)
    logger.info("FASTAPI APPLICATION STARTUP")
    logger.info("="*80)
    agent_client = BedrockAgentClient()
    logger.info("[OK] Application ready")
    logger.info("="*80 + "\n")
    yield
    logger.info("\n" + "="*80)
    logger.info("FASTAPI APPLICATION SHUTDOWN")
    logger.info("="*80 + "\n")


# FastAPI app
app = FastAPI(
    title="Bedrock Agent API with Local Tracing",
    version="1.0.0",
    description="AWS Bedrock Agent API with CloudWatch-style local trace logging",
    lifespan=lifespan
)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the Bedrock agent and get response with traces
    """
    try:
        logger.info(f"[API] Received query: {request.query}")
        
        result = agent_client.invoke_agent(request.query, request.session_id)
        
        return QueryResponse(
            status="success",
            query=request.query,
            response=result["response"],
            session_id=result["session_id"],
            traces_captured=result["traces_captured"],
            trace_file=result["trace_file"],
            log_file=result["log_file"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"[API] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "agent_initialized": agent_client is not None,
        "agent_id": agent_client.agent_id if agent_client else None,
        "trace_file": trace_filename,
        "log_file": log_filename
    }


@app.get("/traces")
async def get_traces():
    """
    Get all captured traces
    """
    try:
        if os.path.exists(trace_filename):
            with open(trace_filename, 'r', encoding='utf-8') as f:
                traces = json.load(f)
            return {
                "status": "success",
                "total_traces": len(traces),
                "trace_file": trace_filename,
                "traces": traces
            }
        else:
            return {
                "status": "no_traces",
                "message": "No traces file found yet"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def cli_interface():
    """
    Command-line interface for testing the agent
    """
    print("\n" + "="*80)
    print("BEDROCK AGENT - CLI INTERFACE")
    print("="*80)
    print(f"Trace File: {trace_filename}")
    print(f"Log File: {log_filename}")
    print("="*80 + "\n")
    
    global agent_client
    agent_client = BedrockAgentClient()
    
    print("\nAgent Ready. Example queries:")
    print("  - What's the weather in London?")
    print("  - Compare weather in Paris and Tokyo")
    print("  - Get 5-day forecast for New York\n")
    
    session_id = str(uuid.uuid4())
    
    while True:
        try:
            user_input = input("Your Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\nTraces saved to: {trace_filename}")
                print(f"Logs saved to: {log_filename}")
                print("Goodbye!\n")
                break
            
            if not user_input:
                continue
            
            print("\nProcessing...\n")
            result = agent_client.invoke_agent(user_input, session_id)
            
            print("\n" + "="*80)
            print("RESPONSE")
            print("="*80)
            print(f"\n{result['response']}\n")
            print("="*80)
            print(f"Traces: {result['traces_captured']}")
            print(f"Session: {result['session_id']}")
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print(f"\n\nTraces saved to: {trace_filename}")
            print(f"Logs saved to: {log_filename}")
            print("Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run FastAPI server
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        # Run CLI interface
        cli_interface()