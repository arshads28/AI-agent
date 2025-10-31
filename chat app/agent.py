from typing import Annotated

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import tools

class MessagesState(TypedDict):
    """
    The state for our graph is a list of messages.
    'add_messages' ensures that new messages are appended to the list,
    not overwritten.
    """
    messages: Annotated[list, add_messages]


# Initialize the LLM
# We use streaming to make it easier to stream tokens back to the client
model = ChatOpenAI(model="gpt-4o", streaming=True)

# Bind the tools to the model
# This tells the model what tools it can call
model_with_tools = model.bind_tools(tools)

# ---------------------------------
# Define the graph nodes
# ---------------------------------

def agent_node(state: MessagesState):
    """
    The primary node that calls the LLM.
    It takes the current state (list of messages) and invokes the model.
    The model can respond with a message or a tool call.
    """
    response = model_with_tools.invoke(state["messages"])
    # The response is added to the state via the 'add_messages' annotator
    return {"messages": [response]}

# The ToolNode is a prebuilt node that executes tools
# It takes the tool calls from the last AIMessage and runs them
tool_node = ToolNode(tools)


# ---------------------------------
# Define the graph edges
# ---------------------------------

def should_continue(state: MessagesState):
    """
    Conditional edge logic.
    It checks the last message in the state.
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
# Compile the graph with persistence
# ---------------------------------

# SqliteSaver.from_conn_string(":memory:") can be used for in-memory persistence
# For production, we use a file to persist state across server restarts
memory = SqliteSaver.from_conn_string("checkpoints.sqlite")

# `compile()` creates a runnable instance of the graph
# We pass the checkpointer to enable persistence
app = workflow.compile(checkpointer=memory)

# This compiled 'app' is what we'll use in our FastAPI server
