import os
import logging
from langchain_community.tools.tavily_search import TavilySearchResults

# Setup logging
logger = logging.getLogger(__name__)

# --- Tool 1: Tavily Search ---
tavily_tool = None
if os.environ.get("TAVILY_API_KEY"):
    tavily_tool = TavilySearchResults(max_results=3)
    logger.info("Tavily Search tool loaded.")
else:
    logger.warning("TAVILY_API_KEY not set. Tavily Search tool will not be available.")


tools = []
if tavily_tool:
    tools.append(tavily_tool)

if not tools:
    logger.error("No tools were successfully loaded! The agent will not have web search capabilities.")

