"""
"Ask the Concierge" - a single tool-using assistant (the MCP *host* from Module 4).

Unlike the multi-agent workflow, this is one LLM in a ReAct loop that can call
ANY tool on the MCP server to answer free-form questions such as
"Tell me about Iron & Embers" or "What do people say about Sakura Garden?".
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from restaurant_ai.agents.mcp_client import open_mcp_tools
from restaurant_ai.agents.tool_loop import run_tool_calling_loop
from restaurant_ai.llm import get_text_llm
from restaurant_ai.observability import setup_tracing

SYSTEM_PROMPT = """You are the Connoisseur Concierge, a friendly guide to California restaurants and home recipes.
Always use the tools to look up facts - never invent restaurants, ratings or reviews.
If a tool finds nothing, say so. Keep answers short and use bullet points for lists."""


async def ask_concierge(message: str, history: list[dict] | None = None) -> tuple[str, list]:
    """Answer one chat message. `history` uses Gradio's format: [{"role": ..., "content": ...}]."""
    setup_tracing()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in (history or [])[-6:]:  # last few turns are enough context
        content = turn.get("content")
        if not isinstance(content, str):
            continue
        messages.append(HumanMessage(content=content) if turn.get("role") == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    async with open_mcp_tools() as tools:
        return await run_tool_calling_loop(get_text_llm(temperature=0.3), list(tools.values()), messages)
