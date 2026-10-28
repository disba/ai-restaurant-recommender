"""
Gradio web app - one UI for the whole project.

Tabs:
    🍽️ Recommend          -> the 5-agent LangGraph workflow (Module 3)
    💬 Ask the Concierge   -> single LLM agent that calls MCP tools (Module 4 host)
    📷 Food Photo Search   -> vision LLM caption + CLIP image search (Modules 1 & 2)
    🗂️ Manage Restaurants  -> add / edit / delete records (Module 1), index kept in sync
    ℹ️ About               -> architecture and links

Run:
    python -m restaurant_ai.app.gradio_app      (then open http://localhost:7860)
"""

import json

import gradio as gr

from restaurant_ai import config
from restaurant_ai.agents.assistant import ask_concierge
from restaurant_ai.agents.formatting import format_agent_trace, format_recommendations
from restaurant_ai.agents.graph import run_recommendation
from restaurant_ai.agents.mcp_client import call_tool_json, open_mcp_tools
from restaurant_ai.data_pipeline import store
from restaurant_ai.llm import ask_vision_llm
from restaurant_ai.observability import new_trace, setup_tracing
from restaurant_ai.rag import retriever

NO_USER = "(anonymous)"


def _user_ids() -> list[str]:
    return [NO_USER] + sorted({rv["userId"] for rv in store.load_json(config.REVIEWS_JSON)})


def _image(relative_path: str) -> str:
    return str(config.ROOT_DIR / relative_path)


# ---------------------------------------------------------------------------
# Tab 1 - Multi-agent recommendations
# ---------------------------------------------------------------------------
async def recommend(request: str, user_id: str):
    if not request.strip():
        return "Please describe what you are in the mood for.", [], ""
    # Each click = its own trace in Phoenix (see observability.new_trace)
    with new_trace("recommend-tab", request) as span:
        state = await run_recommendation(request, None if user_id == NO_USER else user_id)
        answer = format_recommendations(state)
        span.set_attribute("output.value", answer)
    recipes = state.get("final_recommendations", {}).get("recipes", [])
    gallery = [(_image(r["image_path"]), r["name"]) for r in recipes if r.get("image_path")]
    return answer, gallery, format_agent_trace(state)


# ---------------------------------------------------------------------------
# Tab 2 - Concierge chat (MCP host)
# ---------------------------------------------------------------------------
async def concierge_chat(message: str, history: list):
    # Groups this message's LLM calls + MCP tool calls into one trace
    with new_trace("concierge-chat", message) as span:
        answer, tool_log = await ask_concierge(message, history)
        span.set_attribute("output.value", answer)
    if tool_log:
        used = ", ".join(f"`{t['tool']}`" for t in tool_log)
        answer += f"\n\n<sub>🔧 MCP tools used: {used}</sub>"
    return answer


# ---------------------------------------------------------------------------
# Tab 3 - Food photo search (multimodal)
# ---------------------------------------------------------------------------
async def photo_search(image_path: str):
    if not image_path:
        return "Please upload a food photo.", []
    with new_trace("photo-search", image_path) as span:
        caption = ask_vision_llm(
            "You are a food expert. Describe only what you see.",
            "What dish is this? Describe its main ingredients, cooking style and presentation in 2-3 sentences.",
            image_path,
        )
        async with open_mcp_tools() as tools:
            result = await call_tool_json(tools, "find_recipes_by_image", {"image_path": image_path, "k": 6})
        span.set_attribute("output.value", caption)
    gallery = [(_image(r["image_path"]), f"{r['name']} (similarity {r['similarity']:.2f})") for r in result["recipes"]]
    return f"**What the vision model sees ({config.VISION_LLM_MODEL}):** {caption}", gallery


# ---------------------------------------------------------------------------
# Tab 4 - Manage restaurants
# ---------------------------------------------------------------------------
EDIT_FIELDS = ["name", "location", "type", "food_style", "vibe", "environment", "rating", "price_range"]


def _restaurant_choices():
    return [(f"{r.get('name')} - {r.get('location') or '?'} ({r['itemId']})", r["itemId"])
            for r in store.load_restaurants()]


def load_restaurant(item_id):
    record = next((r for r in store.load_restaurants() if r["itemId"] == item_id), None)
    if record is None:
        return [""] * len(EDIT_FIELDS) + ["{}"]
    return [record.get(f) if record.get(f) is not None else "" for f in EDIT_FIELDS] + [json.dumps(record, indent=2)]


def save_restaurant(item_id, *values):
    if item_id is None:
        return "Select a restaurant first.", gr.update()
    changes = dict(zip(EDIT_FIELDS, values))
    try:
        changes["rating"] = float(changes["rating"]) if str(changes["rating"]).strip() else None
        changes["price_range"] = int(changes["price_range"]) if str(changes["price_range"]).strip() else None
    except ValueError:
        return "Rating must be a number and price_range an integer 1-4.", gr.update()
    record = store.update_restaurant(item_id, changes)
    retriever.upsert_restaurant(record)
    return f"✅ Saved '{record['name']}' (backup created, search index updated).", gr.update(choices=_restaurant_choices())


def add_restaurant(paragraph: str):
    if not paragraph.strip():
        return "Paste a restaurant description first.", gr.update()
    with new_trace("add-restaurant", paragraph):
        record = store.add_restaurant_from_text(paragraph)
    retriever.upsert_restaurant(record)
    return (f"✅ Added via LLM structuring:\n```json\n{json.dumps(record, indent=2)}\n```",
            gr.update(choices=_restaurant_choices(), value=record["itemId"]))


def delete_restaurant(item_id, confirmed: bool):
    if item_id is None or not confirmed:
        return "Select a restaurant and tick the confirmation box.", gr.update()
    removed = store.delete_restaurant(item_id)
    retriever.delete_restaurant(item_id)
    return f"🗑️ Deleted '{removed['name']}' (backup created).", gr.update(choices=_restaurant_choices(), value=None)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Restaurant Recommender") as app:
        gr.Markdown("# 🍽️ AI Restaurant & Recipe Recommender\n"
                    f"Local LLMs via Ollama (`{config.TEXT_LLM_MODEL}`, `{config.VISION_LLM_MODEL}`) · "
                    "LangGraph multi-agent · Multimodal RAG · MCP tools")

        with gr.Tab("🍽️ Recommend"):
            with gr.Row():
                request = gr.Textbox(label="What are you in the mood for?", scale=4,
                                     placeholder="e.g. A cozy, affordable Japanese dinner with great noodles")
                user = gr.Dropdown(_user_ids(), value=NO_USER, label="Personalise with review history", scale=1)
            go = gr.Button("Get recommendations (5 agents)", variant="primary")
            gr.Examples(["Romantic Italian date night in Silver Lake, I'm vegetarian",
                         "Cheap and cheerful seafood tacos near the beach",
                         "Spicy Szechuan food and a recipe to cook at home",
                         "Healthy gluten-free lunch ideas"], inputs=request)
            # latex_delimiters=[] stops Gradio from treating "$$" prices as math formulas
            result = gr.Markdown(latex_delimiters=[])
            gallery = gr.Gallery(label="Recommended recipes", columns=3, height=260)
            with gr.Accordion("🔍 Agent workflow trace (what each agent did)", open=False):
                trace = gr.Markdown(latex_delimiters=[])
            go.click(recommend, [request, user], [result, gallery, trace])
            request.submit(recommend, [request, user], [result, gallery, trace])

        with gr.Tab("💬 Ask the Concierge"):
            gr.ChatInterface(
                concierge_chat,
                examples=["Tell me about Iron & Embers", "What do reviewers say about The Gilded Artichoke?",
                          "Find moody restaurants with a great wine list", "How do I make Pad Thai?"],
                description="One LLM agent that decides which MCP tools to call to answer your question.",
                chatbot=gr.Chatbot(latex_delimiters=[], height=450),
            )

        with gr.Tab("📷 Food Photo Search"):
            gr.Markdown("Upload a food photo: the **vision LLM** describes it and **CLIP** finds recipes that look similar.")
            photo = gr.Image(type="filepath", label="Food photo", height=300)
            photo_btn = gr.Button("Analyse photo", variant="primary")
            gr.Examples([str(config.RECIPE_IMAGES_DIR / "recipe1.png"), str(config.RECIPE_IMAGES_DIR / "recipe20.png")],
                        inputs=photo)
            caption = gr.Markdown()
            similar = gr.Gallery(label="Similar recipes", columns=3, height=260)
            photo_btn.click(photo_search, photo, [caption, similar])

        with gr.Tab("🗂️ Manage Restaurants"):
            status = gr.Markdown()
            picker = gr.Dropdown(_restaurant_choices(), label="Select a restaurant", value=None)
            with gr.Row():
                fields = [gr.Textbox(label=f) for f in EDIT_FIELDS[:4]]
            with gr.Row():
                fields += [gr.Textbox(label=f) for f in EDIT_FIELDS[4:]]
            with gr.Accordion("Raw JSON", open=False):
                raw = gr.Code(language="json")
            with gr.Row():
                save_btn = gr.Button("💾 Save changes", variant="primary")
                confirm = gr.Checkbox(label="Yes, I really want to delete")
                delete_btn = gr.Button("🗑️ Delete", variant="stop")
            gr.Markdown("### ➕ Add a new restaurant\nPaste a free-text description - the LLM turns it into structured JSON.")
            paragraph = gr.Textbox(lines=4, label="Restaurant description")
            add_btn = gr.Button("Add with LLM")

            picker.change(load_restaurant, picker, fields + [raw])
            save_btn.click(save_restaurant, [picker] + fields, [status, picker])
            delete_btn.click(delete_restaurant, [picker, confirm], [status, picker])
            add_btn.click(add_restaurant, paragraph, [status, picker])

        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
### How it works
```text
                                   ┌─> cuisine_agent ───┐
profile_agent ──> retrieval_agent ─┤   (in parallel)    ├─> recommendation_agent
                  (uses MCP tools) └─> nutrition_agent ─┘
```
The retrieval agent gets real data from the **MCP server** (vector search over ChromaDB + JSON files).
- **Observability:** open [Phoenix]({config.PHOENIX_URL}) to see every agent, LLM call and MCP tool call.
- **Evals:** run `make eval` to score retrieval, grounding, constraints and answer relevance.
- **LLM providers:** switch between local Ollama and any OpenAI-compatible cloud API in `.env`.
""")
    return app


def main():
    setup_tracing()
    build_app().launch(
        allowed_paths=[str(config.DATA_DIR)],
        theme=gr.themes.Soft(),
        # Don't trace Gradio's own web requests (page loads, polling); Phoenix
        # should only show our agent/LLM/tool traces.
        app_kwargs={"telemetry": {"tracing": False, "metrics": False, "logs": False}},
    )


if __name__ == "__main__":
    main()
