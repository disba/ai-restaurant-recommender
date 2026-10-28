"""
A small "ReAct" tool-calling loop (Reason + Act).

    1. Send the conversation + the list of available tools to the LLM
    2. If the LLM asks for tool calls -> run them on the MCP server, add the results
    3. Repeat until the LLM answers with plain text (or we hit max_steps)

Used by the Retrieval Agent and by the "Ask the Concierge" chat assistant.
"""

from langchain_core.messages import BaseMessage, ToolMessage

from restaurant_ai.agents.mcp_client import tool_output_to_text


async def run_tool_calling_loop(llm, tools: list, messages: list[BaseMessage], max_steps: int = 5):
    """Returns (final_answer_text, tool_log). tool_log = [{"tool", "args", "output"}, ...]"""
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}
    tool_log = []

    for _ in range(max_steps):
        ai_message = await llm_with_tools.ainvoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:  # no more tools needed -> final answer
            return str(ai_message.content), tool_log

        for call in ai_message.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                output = f"Error: unknown tool '{call['name']}'."
            else:
                try:
                    output = tool_output_to_text(await tool.ainvoke(call["args"]))
                except Exception as error:  # bad arguments from the LLM etc.
                    output = f"Error calling {call['name']}: {error}"
            tool_log.append({"tool": call["name"], "args": call["args"], "output": output})
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    return "I could not finish within the allowed number of steps.", tool_log
