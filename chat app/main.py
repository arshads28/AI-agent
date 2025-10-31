import json
import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from typing import Optional

# Import the compiled LangGraph app
from agent import app as langgraph_app

# Load environment variables from .env file
load_dotenv()

# --- Setup Logging ---
# This provides visibility into what's happening
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LangGraph FastAPI Agent",
    description="A production-level chat agent using FastAPI and LangGraph with SSE streaming.",
    version="1.0.0",
)

# --- Pydantic Models ---
# These models define the expected request/response JSON structures
class ChatRequest(BaseModel):
    """Request model for the chat endpoint"""
    user_message: str
    model_name: Optional[str] = None # Allow client to specify model

# --- SSE Streaming Generator ---
async def stream_generator(thread_id: str, message: str, model_name: str | None = None):
    """
    This is the core generator function that streams events from LangGraph.
    It yields JSON strings formatted for Server-Sent Events (SSE).
    """
    logger.info(f"Starting stream for thread: {thread_id}, model: {model_name}")
    try:
        # The config dict specifies the 'thread_id' for persistence
        # and optionally the 'model_name'
        config_payload = {"thread_id": thread_id}
        if model_name:
            config_payload["model_name"] = model_name
            
        config = {"configurable": config_payload}
        
        # This is the input message for the graph
        input_messages = {"messages": [("user", message)]}

        # `astream_events` streams *all* events from the graph (LLM, tools, etc.)
        # `stream_mode="values"` streams the output of each node *as it's produced*
        async for event in langgraph_app.astream_events(
            input_messages,
            config,
            version="v2",
            stream_mode="values"
        ):
            event_type = event["event"]
            event_data = event["data"]
            
            logger.debug(f"Event: {event_type}, Data: {event_data}")

            # Stream LLM tokens as they are generated
            if event_type == "on_llm_stream":
                chunk = event_data.get("chunk")
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    # Yield a JSON object with type 'token'
                    yield json.dumps({"type": "token", "data": chunk.content})
            
            # Stream tool calls
            elif event_type == "on_tool_start":
                yield json.dumps({
                    "type": "tool_start", 
                    "name": event["name"], 
                    "input": event_data.get("input")
                })
            
            # Stream tool outputs
            elif event_type == "on_tool_end":
                output = event_data.get("output")
                # Truncate large tool outputs for cleaner logs/streaming
                if isinstance(output, str) and len(output) > 250:
                     output = output[:250] + "... (truncated)"
                
                yield json.dumps({
                    "type": "tool_end", 
                    "name": event["name"], 
                    "output": output
                })
        
        # Signal the end of the conversation turn
        logger.info(f"Finished stream for thread: {thread_id}")
        yield json.dumps({"type": "end"})

    except Exception as e:
        logger.error(f"Error in stream for thread {thread_id}: {e}", exc_info=True)
        # Signal an error to the client
        yield json.dumps({"type": "error", "data": str(e)})

# --- FastAPI Endpoints ---
@app.post("/chat/stream/{thread_id}")
async def chat_stream(thread_id: str, request: ChatRequest):
    """
    Main chat endpoint.
    Takes a `thread_id` and a `ChatRequest` body.
    The body can optionally specify a `model_name`.
    Streams responses back using Server-Sent Events (SSE).
    """
    generator = stream_generator(thread_id, request.user_message, request.model_name)
    return EventSourceResponse(generator, media_type="text/event-stream")

@app.get("/")
async def root():
    """
    Root endpoint for health checks.
    """
    return {"message": "LangGraph Agent Server is running. POST to /chat/stream/{thread_id}"}

# --- Main execution ---
if __name__ == "__main__":
    # This allows running the app directly with `python main.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)

