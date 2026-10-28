"""
Module 4 - MCP server ("Restaurant Data Server").

The Model Context Protocol (MCP) lets any AI app discover and call tools over
a standard interface. This server wraps our data + vector search as MCP tools,
so the agents (Module 3) and the chat assistant get ALL their data from here.

Tools exposed:
    search_restaurants       semantic search + filters (vector DB)
    search_recipes           multimodal late-fusion recipe search (vector DB)
    find_recipes_by_image    image -> image recipe search (CLIP)
    get_restaurant_details   full record for one restaurant
    get_restaurant_reviews   user reviews (+ photo captions) for a restaurant
    get_user_dining_history  everything a user has reviewed (for profiling)
    get_recipe_details       ingredients + directions for a recipe
Resource:
    guide://california       the raw restaurant guide text

Run standalone (for testing with an MCP inspector):
    python -m restaurant_ai.mcp_server.server
Normally you don't run it yourself - the agents start it automatically (stdio).
"""

import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from restaurant_ai import config
from restaurant_ai.data_pipeline.store import load_json
from restaurant_ai.rag import retriever

# Keep the console quiet: only show warnings/errors from the server and libraries
for noisy in ("httpx", "sentence_transformers", "mcp", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

mcp = FastMCP("restaurant-data-server", log_level="WARNING")


def _to_json(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _find_restaurant(name: str) -> Optional[dict]:
    """Case-insensitive, partial-name lookup."""
    query = name.lower().strip()
    for r in load_json(config.RESTAURANTS_JSON):
        restaurant_name = (r.get("name") or "").lower()
        if restaurant_name and (query in restaurant_name or restaurant_name in query):
            return r
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def search_restaurants(query: str, location: Optional[str] = None, max_price: Optional[int] = None,
                       min_rating: Optional[float] = None, k: int = 5) -> str:
    """Find restaurants that match a natural-language request (cuisine, dish, vibe, occasion).

    Args:
        query: what the user wants, e.g. "romantic Italian dinner" or "cheap vegan brunch".
        location: optional neighborhood/city filter, e.g. "Santa Monica".
        max_price: optional max price level 1-4 (1 = $, 4 = $$$$).
        min_rating: optional minimum rating 0-5.
        k: number of results (default 5).
    """
    hits = retriever.search_restaurants(query, k=k, location=location,
                                        max_price=max_price, min_rating=min_rating)
    return _to_json({"query": query, "count": len(hits), "restaurants": hits})


@mcp.tool()
def search_recipes(query: str, cuisine: Optional[str] = None, k: int = 5) -> str:
    """Find home-cooking recipes for a dish, ingredient or craving (text + food-photo search).

    Args:
        query: e.g. "spicy noodles", "healthy vegetarian curry".
        cuisine: optional cuisine filter, e.g. "Italian", "Thai".
        k: number of results (default 5).
    """
    hits = retriever.search_recipes(query, k=k, cuisine=cuisine)
    return _to_json({"query": query, "count": len(hits), "recipes": hits})


@mcp.tool()
def find_recipes_by_image(image_path: str, k: int = 5) -> str:
    """Find recipes whose photo looks similar to the image at the given local file path."""
    hits = retriever.search_recipes_by_image(image_path, k=k)
    return _to_json({"count": len(hits), "recipes": hits})


@mcp.tool()
def get_restaurant_details(restaurant_name: str) -> str:
    """Get the full details (cuisine, price, rating, vibe, signature dishes) of one restaurant by name."""
    match = _find_restaurant(restaurant_name)
    if not match:
        return _to_json({"status": "not_found", "message": f"No restaurant matching '{restaurant_name}'."})
    return _to_json({"status": "found", "restaurant": match})


@mcp.tool()
def get_restaurant_reviews(restaurant_name: str) -> str:
    """Get user reviews (text, rating and photo descriptions) for a restaurant."""
    match = _find_restaurant(restaurant_name)
    if not match:
        return _to_json({"status": "not_found", "message": f"No restaurant matching '{restaurant_name}'."})
    reviews = [rv for rv in load_json(config.REVIEWS_JSON) if rv.get("itemId") == match["itemId"]]
    return _to_json({
        "status": "found" if reviews else "no_reviews",
        "restaurant": match["name"],
        "reviews": [{"title": rv["title"], "text": rv["text"], "rating": rv["rating"], "date": rv.get("date"),
                     "photo_descriptions": rv.get("image_captions", [])} for rv in reviews],
    })


@mcp.tool()
def get_user_dining_history(user_id: str) -> str:
    """Get all restaurants a user has reviewed, with their ratings and comments (used to build a user profile)."""
    restaurants = {r["itemId"]: r for r in load_json(config.RESTAURANTS_JSON)}
    history = []
    for rv in load_json(config.REVIEWS_JSON):
        if rv.get("userId") == user_id:
            r = restaurants.get(rv["itemId"], {})
            history.append({"restaurant": r.get("name"), "cuisine": r.get("food_style"),
                            "price_range": r.get("price_range"), "user_rating": rv["rating"],
                            "review": rv["text"]})
    return _to_json({"user_id": user_id, "visits": len(history), "history": history})


@mcp.tool()
def get_recipe_details(recipe_name: str) -> str:
    """Get ingredients, cooking time and directions of a recipe by name."""
    query = recipe_name.lower().strip()
    for rec in load_json(config.RECIPES_JSON):
        if query in rec["name"].lower() or rec["name"].lower() in query:
            return _to_json({"status": "found", "recipe": rec})
    return _to_json({"status": "not_found", "message": f"No recipe matching '{recipe_name}'."})


# ---------------------------------------------------------------------------
# Resource (read-only document)
# ---------------------------------------------------------------------------
@mcp.resource("guide://california")
def california_restaurant_guide() -> str:
    """The original free-text California restaurant guide."""
    return config.RESTAURANT_GUIDE_TXT.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run()  # stdio transport
