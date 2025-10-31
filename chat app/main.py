import logging
import json
import uvicorn
import aiosqlite
from fastapi import FastAPI, HTTPException, Request # <-- Add Request
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse # <-- Add RedirectResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager

# Import the graph definition and the async checkpointer
from agent import graph_definition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from fastapi.staticfiles import StaticFiles # <-- Add StaticFiles

# Setup logging
logger = logging.getLogger("agent")

# logging.basicConfig(level=logging.INFO)

# logger = logging.getLogger(__name__)

# This will hold our compiled-with-persistence app
langgraph_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI lifespan events.
    This is the new way to handle startup/shutdown in modern FastAPI.
    """
    global langgraph_app
    logger.info("Application startup...")
    
    # Create the async connection
    conn = await aiosqlite.connect("checkpoints.sqlite")
    
    # Pass the connection to the AsyncSqliteSaver
    memory = AsyncSqliteSaver(conn=conn)
    
    # Compile the graph with the checkpointer
    langgraph_app = graph_definition.compile(checkpointer=memory)
    
    logger.info("LangGraph app compiled with persistence.")
    
    yield  # This is where the application runs
    
    # --- Shutdown ---
    await conn.close()
    logger.info("Database connection closed. Application shutdown.")

# Pass the lifespan context manager to the FastAPI app
app = FastAPI(lifespan=lifespan)

# --- Additions ---
# Mount the current directory ('.') to serve static files from '/static'
# This is how the server will find and serve your index.html
app.mount("/static", StaticFiles(directory="."), name="static")
# --- End Additions ---


class ChatRequest(BaseModel):
    """Pydantic model for a chat request body."""
    input: str
    model_name: str = "string"  # Default value

@app.get("/")
async def get_root(request: Request): # <-- Add Request
    """Redirects the root URL '/' to our static 'index.html' file."""
    return RedirectResponse(url="/static/index.html") # <-- Updated this function

async def stream_generator(thread_id: str, request: ChatRequest):
    """
    Generator function to stream LangGraph events.
    """
    global langgraph_app
    if langgraph_app is None:
        logger.error("Graph app is not initialized!")
        yield "data: {\"error\": \"Graph app not initialized\"}\n\n"
        return

    # Configuration for the graph:
    # 'thread_id' is the key for persistence
    # 'model_name' is passed to our agent_node
    config = {
        "configurable": {
            "thread_id": thread_id,
            "model_name": request.model_name
        }
    }
    
    logger.info(f"Starting stream for thread: {thread_id}, model: {request.model_name}")
    logger.info(f"User message is @@@@@@@ ------->>>> {request.input}")
    
    # --- FIX: Add flag to track streaming ---
    # This prevents sending the full message at the end if we've already streamed chunks.
    streamed_content = False
    # --- END FIX ---
    
    try:
        # astream_events streams all events (nodes, tools, state changes)
        async for event in langgraph_app.astream_events(
            {"messages": [HumanMessage(content=request.input)]},
            config,
            version="v2",
        ):
            # We only stream 'on_chat_model_stream' events
            # which contain the LLM's streaming output chunks
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    
                    streamed_content = True
                   
                    
                    # Format as Server-Sent Event (SSE)
                    logger.info(f" ai response is @@@@@@@ : {chunk.content}")
                    data = json.dumps({"content": chunk.content})
                    yield f"data: {data}\n\n"
            
            
            # Check for the end of the 'agent' node
            # This handles cases where the model *doesn't* stream
            # (e.g., it returns a tool call, or a short, non-streamed message)
            if event["event"] == "on_chain_end" and event["name"] == "agent":
                if not streamed_content:
                    output = event["data"].get("output", {})
                    messages = output.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        
                        if hasattr(last_message, 'content') and last_message.content:
                            final_content = ""
                            content_to_process = last_message.content

                            if isinstance(content_to_process, str):
                                # Case 1: Content is a simple string (your "Success" case)
                                final_content = content_to_process
                            elif isinstance(content_to_process, list):
                                # Case 2: Content is a list of parts (your "Failure" case)
                                for part in content_to_process:
                                    if isinstance(part, str):
                                        final_content += part
                                    elif isinstance(part, dict) and part.get('type') == 'text':
                                        final_content += part.get('text', '')
                            
                            # Only yield if we actually processed some content
                            if final_content:
                                logger.info(f"Thread {thread_id}: Yielding non-streamed, processed agent output.")
                                logger.info(f"ai response is non streamed @@@@@@@ :{final_content} ")
                                data = json.dumps({"content": final_content})
                                yield f"data: {data}\n\n"
                                streamed_content = True 
            
            
            # Log node starts for debugging
            if event["event"] == "on_chain_start" and "agent" in event["name"]:
                logger.info(f"Thread {thread_id}: 'agent' node started")
            
            # Log tool calls for debugging
            if event["event"] == "on_tool_start":
                logger.info(f"Thread {thread_id}: Tool call started: {event['name']}")

    except Exception as e:
        logger.error(f"Error in stream for thread {thread_id}: {e}", exc_info=True)
        # Stream an error event to the client
        data = json.dumps({"error": str(e)})
        yield f"data: {data}\n\n"
    finally:
        logger.info(f"Stream ended for thread: {thread_id}")

@app.post("/chat/stream/{thread_id}")
async def chat_stream(thread_id: str, request: ChatRequest):
    """
    POST endpoint to stream chat responses using SSE.
    """
    logger.info(f"Received request for thread: {thread_id}")
    return StreamingResponse(
        stream_generator(thread_id, request),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    # This is the entry point for running the server directly
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True, 
        log_config="logging.yaml"
    )

