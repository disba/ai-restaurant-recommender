import json

import pytest
from pydantic import ValidationError

from evals import metrics
from restaurant_ai import config
from restaurant_ai.agents.graph import build_graph
from restaurant_ai.agents.formatting import format_recommendations
from restaurant_ai.agents.nodes import _attach_details, retrieval_agent
from restaurant_ai.data_pipeline import store
from restaurant_ai.data_pipeline.schemas import RestaurantRecord
from restaurant_ai.llm import _build_chat_model, parse_json
from restaurant_ai.mcp_server import server
from restaurant_ai.rag import retriever


@pytest.mark.parametrize("text", ['{"ok": true}', '```json\n{"ok": true}\n```',
                                 'Here is the result: {"ok": true}'])
def test_parse_json(text):
    assert parse_json(text) == {"ok": True}


def test_bad_json_is_an_error():
    with pytest.raises(ValueError):
        parse_json("not json")


def test_cloud_requires_key():
    with pytest.raises(ValueError, match="API key"):
        _build_chat_model("openai_compatible", "placeholder", "", "https://example.com", 0, False)


def test_schema_rejects_invalid_price():
    with pytest.raises(ValidationError):
        RestaurantRecord(name="Test", itemId=1, price_range=9)


def test_store_validates_and_backs_up(tmp_path, monkeypatch):
    path = tmp_path / "restaurants.json"
    original = [{"itemId": 1, "name": "Test", "price_range": 2}]
    path.write_text(json.dumps(original))
    monkeypatch.setattr(config, "RESTAURANTS_JSON", path)
    monkeypatch.setattr(config, "BACKUP_DIR", tmp_path / "backups")
    with pytest.raises(ValidationError):
        store.update_restaurant(1, {"price_range": 8})
    assert json.loads(path.read_text()) == original
    changes = {"itemId": 99, "name": "Updated"}
    assert store.update_restaurant(1, changes)["itemId"] == 1
    assert changes["itemId"] == 99
    backups = list(config.BACKUP_DIR.glob("*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == original
    store.delete_restaurant(1)
    assert store.load_restaurants() == []
    assert len(list(config.BACKUP_DIR.glob("*.json"))) == 2


def test_grounding_uses_authoritative_metadata():
    result = _attach_details(
        [{"name": "Real", "price_range": 1}, {"name": "Invented"}],
        [{"name": "Real", "price_range": 4}],
    )
    assert result[0]["price_range"] == 4
    assert result[0]["grounded"] is True
    assert result[1]["grounded"] is False
    text = format_recommendations({"final_recommendations": {"restaurants": result}})
    assert "Warning: this name was not found" in text


def test_graph_has_parallel_join():
    edges = {(edge.source, edge.target) for edge in build_graph().get_graph().edges}
    for analyst in ("cuisine_agent", "nutrition_agent"):
        assert ("retrieval_agent", analyst) in edges
        assert (analyst, "recommendation_agent") in edges


def test_mcp_tools_use_data_and_retriever(monkeypatch):
    monkeypatch.setattr(server, "load_json", lambda path: [{"name": "Cafe Test", "itemId": 1}])
    assert json.loads(server.get_restaurant_details("cafe test"))["status"] == "found"
    assert json.loads(server.get_restaurant_details("missing"))["status"] == "not_found"
    monkeypatch.setattr(retriever, "search_restaurants", lambda *args, **kwargs: [{"name": "Cafe Test"}])
    assert json.loads(server.search_restaurants("coffee"))["count"] == 1


def test_metrics():
    assert metrics.hit_at_k([False, True]) == 1
    assert metrics.precision_at_k([False, True]) == 0.5
    assert metrics.reciprocal_rank([False, True]) == 0.5
    assert metrics.completeness({"restaurants": [1] * 3, "recipes": [1] * 3}) == 1
    assert metrics.groundedness({"restaurants": [{"grounded": True}, {"grounded": False}]}) == 0.5
    assert metrics.llm_tool_use([]) == 0
    assert metrics.llm_tool_use([{"agent": "retrieval", "mode": "fallback"}]) == 0


def test_late_fusion_joins_by_recipe_id(monkeypatch):
    def hit(recipe_id, name, score):
        return {"recipe_id": recipe_id, "name": name, "similarity": score,
                "cuisine": "Test", "total_time": "10 minutes", "image_path": "test.png"}

    text = [hit(1, "First", 0.9), hit(2, "Second", 0.1)]
    images = [hit(2, "Second", 0.9), hit(1, "First", 0.1)]
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "embed_text_for_images", lambda query: [0.0])
    monkeypatch.setattr(retriever, "_query",
                        lambda collection, *args: text if collection == retriever.RECIPES_TEXT else images)
    result = retriever.search_recipes("test", k=2, w_text=0.6, w_image=0.4)
    assert [r["recipe_id"] for r in result] == [1, 2]
    assert [r["fused_score"] for r in result] == [0.6, 0.4]


def test_location_search_checks_full_candidate_pool(monkeypatch):
    class Collection:
        def count(self):
            return 211

    calls = []

    def query(collection, embedding, k, where):
        calls.append(k)
        return [{"name": "Outside", "location": "Downtown"},
                {"name": "Inside", "location": "Malibu Coast"}]

    monkeypatch.setattr(retriever, "get_collection", lambda name: Collection())
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever, "_query", query)
    assert retriever.search_restaurants("seafood", location="malibu", k=1)[0]["name"] == "Inside"
    assert calls == [211]


def test_retrieval_fallback_preserves_constraints(monkeypatch):
    import asyncio
    from restaurant_ai.agents import nodes

    calls = []

    async def no_tool_calls(*args, **kwargs):
        return "", []

    async def fake_call(tools, name, args):
        calls.append((name, args.copy()))
        return {"restaurants": [], "recipes": []}

    monkeypatch.setattr(nodes, "run_tool_calling_loop", no_tool_calls)
    monkeypatch.setattr(nodes, "get_text_llm", lambda **kwargs: None)
    monkeypatch.setattr(nodes, "call_tool_json", fake_call)
    tools = {name: None for name in nodes.RETRIEVAL_TOOLS}
    state = {"user_request": "cheap dinner in Malibu", "user_profile": {"location": "Malibu", "budget": 2}}
    result = asyncio.run(retrieval_agent(state, {"configurable": {"mcp_tools": tools}}))
    restaurant_calls = [args for name, args in calls if name == "search_restaurants"]
    assert restaurant_calls == [{"query": state["user_request"], "k": 6, "location": "Malibu", "max_price": 2}]
    assert result["retrieved_restaurants"] == []


def test_new_trace_starts_a_separate_trace():
    # Simulates Gradio: a long-lived parent span is active while a click is handled
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from restaurant_ai.observability import new_trace

    trace.set_tracer_provider(TracerProvider())  # a real (non no-op) provider, like Phoenix sets up
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("gradio-connection") as parent:
        with new_trace("recommend-tab", "hi") as child:
            pass
    assert child.get_span_context().is_valid
    assert child.get_span_context().trace_id != parent.get_span_context().trace_id
