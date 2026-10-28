"""
The 5 agents, written as LangGraph nodes.

Each node is an async function:  (state, config) -> dict of state updates.
`config["configurable"]["mcp_tools"]` holds the MCP tools opened in graph.py,
so every agent gets its data through the MCP server (no direct file access).
"""

import asyncio
import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from restaurant_ai.agents.agent_configs import system_prompt
from restaurant_ai.agents.mcp_client import call_tool, call_tool_json
from restaurant_ai.agents.state import RecommendationState
from restaurant_ai.agents.tool_loop import run_tool_calling_loop
from restaurant_ai.llm import ask_llm_async, get_text_llm, parse_json

MAX_CANDIDATES = 6
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _mcp_tools(config) -> dict:
    return config["configurable"]["mcp_tools"]


async def _ask_agent_json(agent_key: str, prompt: str, rules: str = "") -> dict:
    """Ask one agent for a JSON answer. Never crashes the workflow on bad JSON."""
    rules += "\nRespond with valid JSON only."
    text = await ask_llm_async(system_prompt(agent_key, rules), prompt, json_mode=True)
    try:
        return parse_json(text)
    except (ValueError, json.JSONDecodeError):
        logger.exception("Agent %s returned invalid JSON", agent_key)
        return {"error": "Agent did not return valid JSON", "raw_output": text[:500]}


def _step(agent: str, start: float, summary: str) -> list:
    """One entry for the agent_steps log (shown in the UI and used by the evals)."""
    return [{"agent": agent, "seconds": round(time.time() - start, 1), "summary": summary}]


def _short_restaurants(restaurants: list) -> list:
    """Only the fields the LLM needs - keeps prompts short for small local models."""
    keys = ("name", "location", "food_style", "vibe", "price_range", "rating")
    return [{k: r.get(k) for k in keys} for r in restaurants]


def _short_recipes(recipes: list) -> list:
    return [{k: r.get(k) for k in ("name", "cuisine", "total_time")} for r in recipes]


def _collect_search_results(tool_log: list) -> tuple[list, list]:
    """Pull restaurant/recipe results out of the raw tool outputs (de-duplicated)."""
    restaurants, recipes, seen = [], [], set()
    for entry in tool_log:
        try:
            data = json.loads(entry["output"])
        except (json.JSONDecodeError, TypeError):
            continue
        for r in data.get("restaurants", []):
            if ("r", r["item_id"]) not in seen:
                seen.add(("r", r["item_id"]))
                restaurants.append(r)
        for r in data.get("recipes", []):
            if ("p", r["recipe_id"]) not in seen:
                seen.add(("p", r["recipe_id"]))
                recipes.append(r)
    return restaurants[:MAX_CANDIDATES], recipes[:MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# Agent 1 - Profile Agent (sequential)
# ---------------------------------------------------------------------------
async def profile_agent(state: RecommendationState, config) -> dict:
    start, tool_calls = time.time(), []
    history = "No dining history available."

    # Deterministic MCP call: fetch the user's past reviews if we know who they are
    if state.get("user_id"):
        args = {"user_id": state["user_id"]}
        history = await call_tool(_mcp_tools(config), "get_user_dining_history", args)
        tool_calls.append({"agent": "profile", "tool": "get_user_dining_history", "args": args, "mode": "direct"})

    prompt = f"""User request: "{state['user_request']}"

Past dining history: {history}

Build the user's profile as JSON with these keys:
- favorite_cuisines: list of strings
- dietary_restrictions: list of strings (e.g. "vegan", "gluten-free"), [] if none
- budget: integer 1-4 ($ to $$$$) ONLY if the request mentions price/cheap/expensive, else null
- location: city/neighborhood ONLY if the request mentions one, else null
- occasion: e.g. "date night", "quick lunch", or null
- vibe_preferences: list of strings (e.g. "cozy", "romantic")
- craving: short phrase describing the food they want right now
- summary: one sentence describing this user"""
    profile = await _ask_agent_json("profile", prompt)
    return {
        "user_profile": profile,
        "tool_calls": tool_calls,
        "agent_steps": _step("Profile Agent", start, profile.get("summary", "profile created")),
    }


# ---------------------------------------------------------------------------
# Agent 2 - Retrieval Agent (sequential, LLM chooses MCP tools = ReAct)
# ---------------------------------------------------------------------------
RETRIEVAL_TOOLS = ["search_restaurants", "search_recipes", "get_restaurant_details", "get_restaurant_reviews"]


async def retrieval_agent(state: RecommendationState, config) -> dict:
    start, tools = time.time(), _mcp_tools(config)
    profile = state.get("user_profile", {})

    rules = (
        "You have database tools. You MUST call `search_restaurants` and `search_recipes` at least once each. "
        "Only pass `location` or `max_price` filters if the user profile contains them. "
        "Write short, descriptive search queries (cuisine + dish + vibe). "
        "When you have results, reply with one short sentence."
    )
    messages = [
        SystemMessage(content=system_prompt("retrieval", rules)),
        HumanMessage(content=f"User request: {state['user_request']}\nUser profile: {json.dumps(profile)}"),
    ]
    _, tool_log = await run_tool_calling_loop(get_text_llm(temperature=0), [tools[n] for n in RETRIEVAL_TOOLS],
                                              messages, max_steps=4)
    tool_calls = [{"agent": "retrieval", "tool": e["tool"], "args": e["args"], "mode": "llm"} for e in tool_log]
    restaurants, recipes = _collect_search_results(tool_log)
    if profile.get("location"):
        restaurants = [r for r in restaurants
                       if profile["location"].lower() in str(r.get("location", "")).lower()]
    if profile.get("budget"):
        restaurants = [r for r in restaurants
                       if isinstance(r.get("price_range"), int) and 1 <= r["price_range"] <= profile["budget"]]

    # Safety net: if the small LLM forgot a tool (or its filters were too strict),
    # call the tools directly so the workflow always has real candidates.
    if not restaurants:
        args = {"query": state["user_request"], "k": MAX_CANDIDATES}
        if profile.get("location"):
            args["location"] = profile["location"]
        if profile.get("budget"):
            args["max_price"] = profile["budget"]
        restaurants = (await call_tool_json(tools, "search_restaurants", args))["restaurants"]
        tool_calls.append({"agent": "retrieval", "tool": "search_restaurants", "args": args, "mode": "fallback"})
    if not recipes:
        args = {"query": profile.get("craving") or state["user_request"], "k": MAX_CANDIDATES}
        recipes = (await call_tool_json(tools, "search_recipes", args))["recipes"]
        tool_calls.append({"agent": "retrieval", "tool": "search_recipes", "args": args, "mode": "fallback"})

    return {
        "retrieved_restaurants": restaurants,
        "retrieved_recipes": recipes,
        "tool_calls": tool_calls,
        "agent_steps": _step("Retrieval Agent", start,
                             f"{len(restaurants)} restaurants, {len(recipes)} recipes via {len(tool_calls)} MCP calls"),
    }


# ---------------------------------------------------------------------------
# Agent 3a - Cuisine & Trend Analyst (runs in PARALLEL with 3b)
# ---------------------------------------------------------------------------
async def cuisine_agent(state: RecommendationState, config) -> dict:
    start = time.time()
    prompt = f"""User profile: {json.dumps(state.get('user_profile', {}))}
Candidate restaurants: {json.dumps(_short_restaurants(state['retrieved_restaurants']))}
Candidate recipes: {json.dumps(_short_recipes(state['retrieved_recipes']))}

Rate how well each candidate's cuisine and style fits this user. Return JSON:
{{"restaurant_fit": [{{"name": str, "fit_score": 1-10, "reason": str}}],
  "recipe_fit": [{{"name": str, "fit_score": 1-10, "reason": str}}],
  "trends": [str, str]}}"""
    analysis = await _ask_agent_json("cuisine", prompt, "Keep each reason under 20 words.")
    return {
        "cuisine_analysis": analysis,
        "agent_steps": _step("Cuisine & Trend Analyst", start, f"scored {len(analysis.get('restaurant_fit', []))} restaurants"),
    }


# ---------------------------------------------------------------------------
# Agent 3b - Nutrition Agent (runs in PARALLEL with 3a)
# ---------------------------------------------------------------------------
async def nutrition_agent(state: RecommendationState, config) -> dict:
    start, tools = time.time(), _mcp_tools(config)

    # Fetch the ingredients of the candidate recipes from the MCP server (all at once)
    names = [r["name"] for r in state["retrieved_recipes"][:4]]
    details = await asyncio.gather(*[call_tool_json(tools, "get_recipe_details", {"recipe_name": n}) for n in names])
    ingredients = {d["recipe"]["name"]: d["recipe"].get("ingredients", []) for d in details if d.get("status") == "found"}
    tool_calls = [{"agent": "nutrition", "tool": "get_recipe_details", "args": {"recipe_name": n}, "mode": "direct"}
                  for n in names]

    prompt = f"""Dietary restrictions: {state.get('user_profile', {}).get('dietary_restrictions', [])}
Recipes with ingredients: {json.dumps(ingredients)}
Restaurants: {json.dumps(_short_restaurants(state['retrieved_restaurants']))}

Check each item against the dietary restrictions. Return JSON:
{{"recipes": [{{"name": str, "diet_ok": true/false, "note": str}}],
  "restaurants": [{{"name": str, "diet_ok": true/false, "note": str}}],
  "advice": str}}"""
    analysis = await _ask_agent_json("nutrition", prompt, "Keep each note under 15 words. If there are no restrictions, diet_ok is true.")
    return {
        "nutrition_analysis": analysis,
        "tool_calls": tool_calls,
        "agent_steps": _step("Nutrition Agent", start, analysis.get("advice", "nutrition checked")[:120]),
    }


# ---------------------------------------------------------------------------
# Agent 4 - Recommendation Agent (sequential, waits for both parallel agents)
# ---------------------------------------------------------------------------
def _attach_details(picks: list, candidates: list) -> list:
    """Add candidate data (location, price, image...) to each pick and flag whether it really exists."""
    by_name = {c["name"].lower(): c for c in candidates}
    enriched = []
    for pick in picks if isinstance(picks, list) else []:
        if not isinstance(pick, dict):
            continue
        match = by_name.get(str(pick.get("name", "")).lower())
        enriched.append({**pick, **(match or {}), "grounded": match is not None})
    return enriched


async def recommendation_agent(state: RecommendationState, config) -> dict:
    start = time.time()
    prompt = f"""User request: "{state['user_request']}"
User profile: {json.dumps(state.get('user_profile', {}))}
Candidate restaurants: {json.dumps(_short_restaurants(state['retrieved_restaurants']))}
Candidate recipes: {json.dumps(_short_recipes(state['retrieved_recipes']))}
Cuisine analysis: {json.dumps(state.get('cuisine_analysis', {}))}
Nutrition analysis: {json.dumps(state.get('nutrition_analysis', {}))}

Pick the top 3 restaurants and top 3 recipes for this user. Return JSON:
{{"summary": str,
  "restaurants": [{{"name": str, "reason": str}}],
  "recipes": [{{"name": str, "reason": str}}]}}"""
    rules = ("Choose ONLY from the candidate lists and copy names exactly. Prefer high fit_score and diet_ok items. "
             "Each reason is 1-2 friendly sentences.")
    result = await _ask_agent_json("recommender", prompt, rules)

    result["restaurants"] = _attach_details(result.get("restaurants", []), state["retrieved_restaurants"])
    result["recipes"] = _attach_details(result.get("recipes", []), state["retrieved_recipes"])
    return {
        "final_recommendations": result,
        "agent_steps": _step("Recommendation Agent", start, result.get("summary", "")[:120]),
    }
