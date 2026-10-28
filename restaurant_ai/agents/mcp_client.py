"""
MCP client used by the agents (Module 3 <-> Module 4 bridge).

`open_mcp_tools()` starts the MCP server as a subprocess (stdio transport),
asks it which tools it has, and converts them into LangChain tools that an
LLM can call. One server session is shared by all agents of one request.

Usage:
    async with open_mcp_tools() as tools:          # tools = {"search_restaurants": <tool>, ...}
        text = await call_tool(tools, "search_restaurants", {"query": "sushi"})
"""

import json
import os
import sys
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from restaurant_ai import config

SERVER_NAME = "restaurant-data"
SERVER_CONFIG = {
    SERVER_NAME: {
        "transport": "stdio",
        "command": sys.executable,  # the same Python (.venv) that runs this code
        "args": ["-m", "restaurant_ai.mcp_server.server"],
        "cwd": str(config.ROOT_DIR),
        "env": dict(os.environ),
    }
}


@asynccontextmanager
async def open_mcp_tools():
    """Open ONE session to the MCP server and yield its tools as a {name: tool} dict."""
    client = MultiServerMCPClient(SERVER_CONFIG)
    async with client.session(SERVER_NAME) as session:
        tools = await load_mcp_tools(session)
        yield {tool.name: tool for tool in tools}


def tool_output_to_text(output) -> str:
    """MCP tools may return a string or a list of content blocks - always give back plain text."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in output)
    return str(output)


async def call_tool(tools: dict, name: str, args: dict) -> str:
    """Call one MCP tool directly from code (no LLM involved) and return its text output."""
    output = await tools[name].ainvoke(args)
    return tool_output_to_text(output)


async def call_tool_json(tools: dict, name: str, args: dict) -> dict:
    """Same as call_tool, but parses the JSON text our server returns."""
    return json.loads(await call_tool(tools, name, args))
