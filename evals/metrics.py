"""
Simple evaluation metrics.

Every metric returns a number between 0 and 1 (higher = better), except
latency and the LLM judge score (1-5), so results are easy to compare between runs.
"""

import json

from restaurant_ai.llm import ask_llm_async, parse_json

# Which fields we look at when deciding if a result is "about" a keyword
RESTAURANT_TEXT_FIELDS = ("name", "type", "food_style", "vibe", "signatures")
RECIPE_TEXT_FIELDS = ("name", "cuisine")


def record_text(record: dict, fields: tuple) -> str:
    """Join the chosen fields of a record into one lowercase string."""
    parts = []
    for field in fields:
        value = record.get(field)
        parts.append(" ".join(value) if isinstance(value, list) else str(value or ""))
    return " ".join(parts).lower()


def is_relevant(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


# ---------------------------------------------------------------------------
# Retrieval metrics (classic information-retrieval scores)
# ---------------------------------------------------------------------------
def hit_at_k(relevance: list[bool]) -> float:
    """1 if at least one of the top-k results is relevant, else 0."""
    return 1.0 if any(relevance) else 0.0


def precision_at_k(relevance: list[bool]) -> float:
    """Share of the top-k results that are relevant."""
    return sum(relevance) / len(relevance) if relevance else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    """1 / position of the first relevant result (1.0 = first result was relevant)."""
    for position, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / position
    return 0.0


# ---------------------------------------------------------------------------
# Multi-agent output metrics
# ---------------------------------------------------------------------------
def completeness(final: dict, expected_each: int = 3) -> float:
    """Did the Recommendation Agent return 3 restaurants AND 3 recipes?"""
    got = min(len(final.get("restaurants", [])), expected_each) + min(len(final.get("recipes", [])), expected_each)
    return got / (2 * expected_each)


def groundedness(final: dict) -> float:
    """Share of recommended items that really exist in the retrieved data (no hallucinated names)."""
    picks = final.get("restaurants", []) + final.get("recipes", [])
    return sum(1 for p in picks if p.get("grounded")) / len(picks) if picks else 0.0


def location_match(restaurants: list[dict], expected_location: str) -> float:
    """Share of recommended restaurants located where the user asked."""
    if not restaurants:
        return 0.0
    hits = [expected_location.lower() in str(r.get("location", "")).lower() for r in restaurants]
    return sum(hits) / len(hits)


def price_match(restaurants: list[dict], max_price: int) -> float:
    """Share of recommended restaurants within budget (price_range 1..max_price)."""
    if not restaurants:
        return 0.0
    hits = [isinstance(r.get("price_range"), int) and 1 <= r["price_range"] <= max_price for r in restaurants]
    return sum(hits) / len(hits)


def profile_accuracy(profile: dict, case: dict) -> float | None:
    """Did the Profile Agent pick up the location / budget / diet mentioned in the request?"""
    checks = []
    if case.get("expected_location"):
        checks.append(case["expected_location"].lower() in str(profile.get("location") or "").lower())
    if case.get("expected_max_price"):
        budget = profile.get("budget")
        checks.append(isinstance(budget, int) and budget <= case["expected_max_price"])
    if case.get("expected_diet"):
        diets = " ".join(profile.get("dietary_restrictions") or []).lower()
        checks.append(case["expected_diet"].lower() in diets)
    return sum(checks) / len(checks) if checks else None   # None = nothing to check for this case


def llm_tool_use(tool_calls: list[dict]) -> float:
    """1.0 if the Retrieval Agent's LLM chose its own MCP tools, 0.0 if the code fallback was needed."""
    retrieval = [c for c in tool_calls if c.get("agent") == "retrieval"]
    return float(bool(retrieval) and all(c.get("mode") == "llm" for c in retrieval))


# ---------------------------------------------------------------------------
# LLM-as-a-judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are a strict evaluator of restaurant recommendation systems. "
    "Score how well the recommendations match the user's request. Respond in JSON only."
)


async def judge_relevance(request: str, final: dict) -> dict:
    """Ask the local LLM to grade the answer from 1 (useless) to 5 (perfect)."""
    compact = {
        "restaurants": [{k: r.get(k) for k in ("name", "location", "food_style", "price_range")}
                        for r in final.get("restaurants", [])],
        "recipes": [{k: r.get(k) for k in ("name", "cuisine")} for r in final.get("recipes", [])],
    }
    prompt = f"""User request: "{request}"
Recommendations: {json.dumps(compact)}

Score 1-5:
5 = every item clearly matches the request (cuisine, location, budget, diet)
3 = some items match, some are off
1 = items are unrelated to the request
Return JSON: {{"score": int, "reason": "one short sentence"}}"""
    try:
        result = parse_json(await ask_llm_async(JUDGE_SYSTEM, prompt, json_mode=True, temperature=0))
        score = result.get("score")
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError(f"Judge score must be an integer from 1 to 5, got {score!r}")
        return {"score": score, "reason": result.get("reason", "")}
    except Exception as error:   # the judge should never crash the whole eval run
        return {"score": None, "reason": f"judge failed: {error}"}


# ---------------------------------------------------------------------------
# Data quality (Module 1 output)
# ---------------------------------------------------------------------------
def data_quality_report(restaurants: list[dict]) -> dict:
    """Count common problems in the structured restaurant data produced by the LLM."""
    names = [str(r.get("name") or "").strip().lower() for r in restaurants]
    problems = {
        "missing_name": sum(1 for n in names if not n),
        "missing_location": sum(1 for r in restaurants if not str(r.get("location") or "").strip()),
        "invalid_price": sum(1 for r in restaurants
                             if not (isinstance(r.get("price_range"), int) and 1 <= r["price_range"] <= 4)),
        "invalid_rating": sum(1 for r in restaurants
                              if not (isinstance(r.get("rating"), (int, float)) and 0 < r["rating"] <= 5)),
        "no_signature_dishes": sum(1 for r in restaurants if not r.get("signatures")),
        "duplicate_names": len(names) - len(set(names)),
    }
    bad_records = sum(
        1 for r in restaurants
        if not str(r.get("name") or "").strip()
        or not str(r.get("location") or "").strip()
        or not (isinstance(r.get("price_range"), int) and 1 <= r["price_range"] <= 4)
    )
    return {
        "total_records": len(restaurants),
        "clean_record_rate": round(1 - bad_records / len(restaurants), 3) if restaurants else 0.0,
        **problems,
    }
