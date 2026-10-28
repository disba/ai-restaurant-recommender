"""Turn the workflow's final state into readable Markdown (used by the CLI and the Gradio app)."""

import json


def _price(value) -> str:
    return "$" * value if isinstance(value, int) and 1 <= value <= 4 else ""


def format_recommendations(state: dict) -> str:
    recs = state.get("final_recommendations", {})
    if "error" in recs:
        return f"Sorry, the recommendation agent failed: {recs['error']}"

    lines = [f"### Recommendations\n{recs.get('summary', '')}\n", "**Restaurants**"]
    for i, r in enumerate(recs.get("restaurants", []), 1):
        details = " · ".join(str(x) for x in [r.get("location"), _price(r.get("price_range")),
                                               f"★ {r['rating']}" if r.get("rating") else None] if x)
        lines.append(f"{i}. **{r.get('name')}** ({details})  \n   {r.get('reason', '')}")
        if not r.get("grounded"):
            lines.append("   **Warning: this name was not found in the retrieved records. Verify before using.**")
    lines.append("\n**Recipes to cook at home**")
    for i, r in enumerate(recs.get("recipes", []), 1):
        details = " · ".join(x for x in [r.get("cuisine"), r.get("total_time")] if x)
        lines.append(f"{i}. **{r.get('name')}** ({details})  \n   {r.get('reason', '')}")
        if not r.get("grounded"):
            lines.append("   **Warning: this name was not found in the retrieved records. Verify before using.**")
    return "\n".join(lines)


def format_agent_trace(state: dict) -> str:
    """Show what each agent did and which MCP tools were called."""
    lines = ["### Agent workflow trace"]
    for step in state.get("agent_steps", []):
        lines.append(f"- **{step['agent']}** ({step['seconds']}s): {step['summary']}")
    lines.append("\n**MCP tool calls**")
    for call in state.get("tool_calls", []):
        lines.append(f"- `{call['tool']}` by {call['agent']} agent [{call['mode']}] args={json.dumps(call['args'])}")
    profile = state.get("user_profile", {})
    lines.append(f"\n**User profile**\n```json\n{json.dumps(profile, indent=2)}\n```")
    return "\n".join(lines)
