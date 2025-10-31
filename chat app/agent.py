import os
import logging
# We no longer need the 'sqlite3' import
from typing import Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
# We no longer need the checkpointer import here
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import tools

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class MessagesState(TypedDict):
    """
    The canonical state for our graph. It's a dictionary with one
    key, 'messages', which is a list.
    'add_messages' ensures that new messages are appended to the list,
    not overwritten.
    """
    messages: Annotated[list, add_messages]


# ---------------------------------
# Initialize Models and Tools
# ---------------------------------

available_models = {}
default_model = None

# Initialize OpenAI model
if os.getenv("OPENAI_API_KEY"):
    openai_model = ChatOpenAI(model="gpt-4o", streaming=True)
    openai_with_tools = openai_model.bind_tools(tools)
    available_models["openai"] = openai_with_tools
    if not default_model:
        default_model = "openai"
    logger.info("OpenAI model loaded.")
else:
    logger.warning("OPENAI_API_KEY not set. OpenAI model will not be available.") 

# Initialize Gemini model
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    gemini_model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash", 
        streaming=True, 
        google_api_key=gemini_api_key
    )
    
    gemini_with_tools = gemini_model.bind_tools(tools)
    available_models["gemini"] = gemini_with_tools
    if not default_model:
        default_model = "gemini"
    logger.info("Gemini model loaded.")
else:
    logger.warning("GEMINI_API_KEY not set. Gemini model will not be available.")

if not available_models:
    raise RuntimeError("No models were successfully loaded! Please set OPENAI_API_KEY or GEMINI_API_KEY.")

logger.info(f"Default model set to: {default_model}")

# ---------------------------------
# Define the graph nodes
# ---------------------------------

def agent_node(state: MessagesState, config: RunnableConfig):
    """
    The primary node that calls the LLM.
    It checks the config for a specified model, otherwise uses the default.
    It takes the current state (list of messages) and invokes the model.
    The model can respond with a message or a tool call.
    """
    # Get model_name from config, fallback to default
    model_name = config.get("configurable", {}).get("model_name", default_model)
    
    # Get the model from our available models, or use the default if invalid
    model = available_models.get(model_name)
    if not model:
        logger.warning(f"Invalid model_name: {model_name}. Falling back to default: {default_model}")
        model = available_models[default_model]
    
    logger.info(f"Using model: {model_name}")
    
    # Invoke the model with the current state
    response = model.invoke(state["messages"])
    
    # The response is added to the state via the 'add_messages' annotator
    return {"messages": [response]}

# The ToolNode is a prebuilt node that executes tools
# It takes the list of tools, finds the one(s) the agent called,
# executes them, and returns the output
tool_node = ToolNode(tools)


# ---------------------------------
# Define the graph edges
# ---------------------------------

def should_continue(state: MessagesState) -> Literal["call_tools", "__end__"]:
    """
    Conditional edge logic.
    This function is called after the 'agent' node.
    It inspects the last message in the state:
    - If it has tool calls, it routes to 'call_tools'.
    - Otherwise, it ends the graph execution.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tools"
    return END

# ---------------------------------
# Build the graph
# ---------------------------------

# 1. Initialize the StateGraph with our MessagesState
workflow = StateGraph(MessagesState)

# 2. Add the nodes
workflow.add_node("agent", agent_node)
workflow.add_node("call_tools", tool_node)

# 3. Define the entry point
# This tells the graph where to start
workflow.add_edge(START, "agent")

# 4. Add the conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tools": "call_tools",  # If tool calls, go to tool_node
        END: END                    # Otherwise, end
    }
)

# 5. Add the edge from tool node back to agent node
# This allows the agent to process the tool output
workflow.add_edge("call_tools", "agent")


# ---------------------------------
# Export the graph DEFINITION
# ---------------------------------

# *** THIS IS THE CHANGE ***
# We no longer compile here or set up memory.
# We will export the raw graph definition.
# It will be compiled in main.py during the server startup.
graph_definition = workflow

