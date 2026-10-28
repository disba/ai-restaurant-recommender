"""
Module 3 - The LangGraph multi-agent workflow.

    START
      -> profile_agent          (sequential)
      -> retrieval_agent        (sequential, ReAct over MCP tools)
      -> cuisine_agent  \\
      -> nutrition_agent /      (PARALLEL fan-out)
      -> recommendation_agent   (fan-in: waits for both parallel agents)
    END

Run from the terminal:
    python -m restaurant_ai.agents.graph "cozy Japanese dinner in Little Tokyo, nothing too pricey"
    python -m restaurant_ai.agents.graph "something like my past favourites" --user-id USER_FUSION_FINDER_99
    python -m restaurant_ai.agents.graph --mermaid     # print the graph diagram
"""

import argparse
import asyncio

from langgraph.graph import END, START, StateGraph

from restaurant_ai.agents.formatting import format_agent_trace, format_recommendations
from restaurant_ai.agents.mcp_client import open_mcp_tools
from restaurant_ai.agents.nodes import (cuisine_agent, nutrition_agent, profile_agent,
                                        recommendation_agent, retrieval_agent)
from restaurant_ai.agents.state import RecommendationState
from restaurant_ai.observability import setup_tracing


def build_graph():
    graph = StateGraph(RecommendationState)

    graph.add_node("profile_agent", profile_agent)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("cuisine_agent", cuisine_agent)
    graph.add_node("nutrition_agent", nutrition_agent)
    graph.add_node("recommendation_agent", recommendation_agent)

    # Chain (sequential)
    graph.add_edge(START, "profile_agent")
    graph.add_edge("profile_agent", "retrieval_agent")
    # Fan-out: two edges leaving the same node -> both agents run in parallel
    graph.add_edge("retrieval_agent", "cuisine_agent")
    graph.add_edge("retrieval_agent", "nutrition_agent")
    # Fan-in: a list of start nodes -> wait until BOTH are finished
    graph.add_edge(["cuisine_agent", "nutrition_agent"], "recommendation_agent")
    graph.add_edge("recommendation_agent", END)

    return graph.compile()


workflow = build_graph()


async def run_recommendation(user_request: str, user_id: str | None = None) -> dict:
    """Run the full multi-agent workflow and return the final state."""
    setup_tracing()
    async with open_mcp_tools() as mcp_tools:
        return await workflow.ainvoke(
            {"user_request": user_request, "user_id": user_id or "", "tool_calls": [], "agent_steps": []},
            config={"configurable": {"mcp_tools": mcp_tools}, "run_name": "restaurant-recommendation"},
        )


def main():
    parser = argparse.ArgumentParser(description="Run the multi-agent recommender")
    parser.add_argument("request", nargs="?", default="A cozy, affordable Japanese dinner with great noodles")
    parser.add_argument("--user-id", default=None, help="e.g. USER_FUSION_FINDER_99 to personalise with past reviews")
    parser.add_argument("--mermaid", action="store_true", help="Print the workflow diagram (Mermaid) and exit")
    args = parser.parse_args()

    if args.mermaid:
        print(workflow.get_graph().draw_mermaid())
        return

    state = asyncio.run(run_recommendation(args.request, args.user_id))
    print(format_recommendations(state))
    print("\n" + format_agent_trace(state))


if __name__ == "__main__":
    main()
