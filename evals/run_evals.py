"""
Evaluation runner - measures how well the system works.

Three levels of evaluation:
  1. Data quality   - is the LLM-structured restaurant data clean?        (Module 1)
  2. Retrieval      - does vector search return relevant results?         (Module 2)
  3. Multi-agent    - are the final answers complete, grounded, on-topic? (Module 3 + 4)

Usage:
    python -m evals.run_evals                  # everything (agents take ~1 min per case)
    python -m evals.run_evals --limit 2        # quick run: only the first 2 agent cases
    python -m evals.run_evals --skip-agents    # only data quality + retrieval (fast, no LLM)

Results are printed as tables and saved to evals/results/eval_<timestamp>.json
so you can compare runs over time (e.g. after changing a prompt or a model).
If Phoenix is running, every agent run is also traced there.
"""

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime


from evals import metrics
from restaurant_ai import config
from restaurant_ai.data_pipeline.store import load_json
from restaurant_ai.rag import retriever

GOLDEN_SET = config.EVAL_DIR / "golden_set.json"
RESULTS_DIR = config.EVAL_RESULTS_DIR
TOP_K = 5


# ---------------------------------------------------------------------------
# Small printing helper
# ---------------------------------------------------------------------------
def print_table(title: str, rows: list[dict]) -> None:
    """Print a list of dicts as a simple aligned text table."""
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    headers = list(rows[0].keys())
    cells = [[_fmt(row.get(h)) for h in headers] for row in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths)))


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _average(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(statistics.mean(values), 3) if values else None


# ---------------------------------------------------------------------------
# 1. Data quality
# ---------------------------------------------------------------------------
def eval_data_quality() -> dict:
    report = metrics.data_quality_report(load_json(config.RESTAURANTS_JSON))
    print_table("1. Data quality (restaurants.json)", [report])
    return report


# ---------------------------------------------------------------------------
# 2. Retrieval (no LLM involved -> fast and deterministic)
# ---------------------------------------------------------------------------
def eval_retrieval(cases: list[dict]) -> dict:
    restaurants_by_id = {r["itemId"]: r for r in load_json(config.RESTAURANTS_JSON)}
    recipes_by_id = {r["id"]: r for r in load_json(config.RECIPES_JSON)}

    rows = []
    for case in cases:
        query = case["request"]

        # Restaurants: is each hit about the expected cuisine?
        hits = retriever.search_restaurants(query, k=TOP_K)
        rel_r = [metrics.is_relevant(metrics.record_text(restaurants_by_id.get(h["item_id"], h),
                                                         metrics.RESTAURANT_TEXT_FIELDS),
                                     case["restaurant_keywords"]) for h in hits]

        # Recipes: multimodal late-fusion search
        recipe_hits = retriever.search_recipes(query, k=TOP_K)
        rel_p = [metrics.is_relevant(metrics.record_text(recipes_by_id.get(h["recipe_id"], h),
                                                         metrics.RECIPE_TEXT_FIELDS),
                                     case["recipe_keywords"]) for h in recipe_hits]

        row = {
            "case": case["id"],
            "rest_hit@5": metrics.hit_at_k(rel_r),
            "rest_P@5": metrics.precision_at_k(rel_r),
            "rest_MRR": metrics.reciprocal_rank(rel_r),
            "recipe_hit@5": metrics.hit_at_k(rel_p),
            "recipe_P@5": metrics.precision_at_k(rel_p),
            "recipe_MRR": metrics.reciprocal_rank(rel_p),
        }

        # Metadata filters must be respected 100% of the time
        if case.get("expected_location"):
            filtered = retriever.search_restaurants(query, k=TOP_K, location=case["expected_location"])
            row["filter_ok"] = metrics.location_match(filtered, case["expected_location"]) if filtered else None
        elif case.get("expected_max_price"):
            filtered = retriever.search_restaurants(query, k=TOP_K, max_price=case["expected_max_price"])
            row["filter_ok"] = metrics.price_match(filtered, case["expected_max_price"]) if filtered else None
        else:
            row["filter_ok"] = None
        rows.append(row)

    summary = {key: _average(rows, key) for key in rows[0] if key != "case"}
    print_table(f"2. Retrieval (top-{TOP_K}, keyword relevance)", rows + [{"case": "AVERAGE", **summary}])
    return {"per_case": rows, "average": summary}


# ---------------------------------------------------------------------------
# 3. Multi-agent workflow (end-to-end, uses the LLM + MCP server)
# ---------------------------------------------------------------------------
async def eval_agents(cases: list[dict]) -> dict:
    # Imported here so "--skip-agents" doesn't need Ollama running
    from restaurant_ai.agents.graph import run_recommendation

    rows, judge_notes = [], {}
    for number, case in enumerate(cases, start=1):
        print(f"  [{number}/{len(cases)}] running agents for: {case['request']!r} ...", flush=True)
        start = time.time()
        try:
            state = await run_recommendation(case["request"])
        except Exception as error:
            print(f"     failed: {error}")
            rows.append({"case": case["id"], "error": str(error)})
            continue
        latency = time.time() - start

        final = state.get("final_recommendations", {})
        picked_restaurants = final.get("restaurants", [])
        judge = await metrics.judge_relevance(case["request"], final)
        judge_notes[case["id"]] = judge["reason"]

        rows.append({
            "case": case["id"],
            "complete": metrics.completeness(final),
            "grounded": metrics.groundedness(final),
            "on_topic": metrics.precision_at_k([
                metrics.is_relevant(metrics.record_text(r, metrics.RESTAURANT_TEXT_FIELDS), case["restaurant_keywords"])
                for r in picked_restaurants]),
            "location_ok": (metrics.location_match(picked_restaurants, case["expected_location"])
                            if case.get("expected_location") else None),
            "price_ok": (metrics.price_match(picked_restaurants, case["expected_max_price"])
                         if case.get("expected_max_price") else None),
            "profile_ok": metrics.profile_accuracy(state.get("user_profile", {}), case),
            "llm_tool_use": metrics.llm_tool_use(state.get("tool_calls", [])),
            "mcp_calls": len(state.get("tool_calls", [])),
            "judge_1to5": judge["score"],
            "latency_s": round(latency, 1),
        })

    metric_keys = [k for k in rows[0] if k not in ("case", "error")] if rows else []
    summary = {key: _average(rows, key) for key in metric_keys}
    print_table("3. Multi-agent workflow (end-to-end)", rows + [{"case": "AVERAGE", **summary}])
    print("\nLLM judge comments:")
    for case_id, note in judge_notes.items():
        print(f"  - {case_id}: {note}")
    return {"per_case": rows, "average": summary, "judge_notes": judge_notes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the restaurant recommender")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N agent cases")
    parser.add_argument("--skip-agents", action="store_true", help="Skip the slow end-to-end agent evaluation")
    args = parser.parse_args()

    cases = json.loads(GOLDEN_SET.read_text())
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text_model": f"{config.TEXT_LLM_PROVIDER}:{config.TEXT_LLM_MODEL}",
        "data_quality": eval_data_quality(),
        "retrieval": eval_retrieval(cases),
    }
    if not args.skip_agents:
        print("\n=== 3. Multi-agent workflow ===")
        results["agents"] = asyncio.run(eval_agents(cases[:args.limit] if args.limit else cases))

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_file.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
