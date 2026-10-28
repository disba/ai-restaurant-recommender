"""
Shared state of the LangGraph workflow.

Every agent (node) reads what it needs from this dictionary and returns ONLY
the keys it wants to update. LangGraph merges the updates for us.
"""

import operator
from typing import Annotated, TypedDict


class RecommendationState(TypedDict, total=False):
    # Input
    user_request: str
    user_id: str

    # Written by the agents
    user_profile: dict            # Profile Agent
    retrieved_restaurants: list   # Retrieval Agent
    retrieved_recipes: list       # Retrieval Agent
    cuisine_analysis: dict        # Cuisine & Trend Analyst (parallel)
    nutrition_analysis: dict      # Nutrition Agent (parallel)
    final_recommendations: dict   # Recommendation Agent

    # Logs. "Annotated[list, operator.add]" is a *reducer*: when two agents running
    # in parallel both return a log entry, LangGraph ADDS the lists together instead
    # of one overwriting the other.
    tool_calls: Annotated[list, operator.add]
    agent_steps: Annotated[list, operator.add]
