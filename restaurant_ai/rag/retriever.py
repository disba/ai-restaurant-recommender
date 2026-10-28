"""
Module 2 - Retrieval functions over the ChromaDB vector store.

Three collections live in vector_store/:
    restaurants    : MiniLM embedding of each restaurant's full description
    recipes_text   : MiniLM embedding of recipe name + ingredients + image caption
    recipe_images  : CLIP embedding of each recipe photo

Search features:
    - similarity search + metadata filtering (price, rating, location)
    - late fusion: recipe text score and recipe image score are combined
      into one ranking  ->  fused = w_text * text_score + w_image * image_score
    - image -> image search (find recipes that look like an uploaded photo)
"""

from functools import lru_cache
from pathlib import Path

import chromadb

from restaurant_ai import config
from restaurant_ai.rag.embeddings import embed_images, embed_text_for_images, embed_texts

RESTAURANTS = "restaurants"
RECIPES_TEXT = "recipes_text"
RECIPE_IMAGES = "recipe_images"


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))


def get_collection(name: str):
    # cosine distance -> similarity = 1 - distance
    return get_client().get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})


# ---------------------------------------------------------------------------
# Turning records into text + metadata (also used by build_index.py)
# ---------------------------------------------------------------------------
def clean_price(value) -> int:
    """Return 1-4, or 0 when the price is missing/invalid (e.g. None or 999)."""
    return value if isinstance(value, int) and 1 <= value <= 4 else 0


def restaurant_to_text(r: dict) -> str:
    return (
        f"Restaurant: {r.get('name', '')}. Type: {r.get('type', '')}. Cuisine: {r.get('food_style', '')}. "
        f"Location: {r.get('location', '')}. Vibe: {r.get('vibe') or ''}. "
        f"Signature dishes: {', '.join(r.get('signatures') or [])}. Setting: {r.get('environment', '')}."
    )


def restaurant_metadata(r: dict) -> dict:
    return {
        "item_id": r["itemId"],
        "name": r.get("name") or "",
        "location": r.get("location") or "",
        "food_style": r.get("food_style") or "",
        "vibe": r.get("vibe") or "",
        "price_range": clean_price(r.get("price_range")),
        "rating": float(r.get("rating") or 0.0),
    }


def recipe_to_text(rec: dict) -> str:
    return (
        f"Recipe: {rec.get('name', '')}. Cuisine: {rec.get('cuisine', '')}. "
        f"Ingredients: {', '.join(rec.get('ingredients', []))}. Looks like: {rec.get('image_description', '')}"
    )


def recipe_metadata(rec: dict) -> dict:
    image_path = config.RECIPE_IMAGES_DIR / f"recipe{rec['id']}.png"
    return {
        "recipe_id": rec["id"],
        "name": rec.get("name") or "",
        "cuisine": rec.get("cuisine") or "",
        "total_time": rec.get("total_time") or "",
        # stored relative to the project root so the DB works on any machine
        "image_path": str(image_path.relative_to(config.ROOT_DIR)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _query(collection_name: str, embedding: list[float], k: int, where: dict | None = None) -> list[dict]:
    """Run a Chroma query and return a simple list of {metadata..., similarity} dicts."""
    collection = get_collection(collection_name)
    k = min(k, collection.count())
    if k == 0:
        return []
    result = collection.query(query_embeddings=[embedding], n_results=k, where=where,
                              include=["metadatas", "distances"])
    hits = []
    for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
        hits.append({**metadata, "similarity": round(1 - distance, 4)})
    return hits


def _min_max(scores: dict) -> dict:
    """Scale scores to 0..1 so text and image scores are comparable before fusing."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-9:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


# ---------------------------------------------------------------------------
# Public search functions
# ---------------------------------------------------------------------------
def search_restaurants(query: str, k: int = 5, location: str | None = None,
                       max_price: int | None = None, min_rating: float | None = None) -> list[dict]:
    """Semantic restaurant search with optional metadata filters."""
    filters = []
    if max_price:
        filters += [{"price_range": {"$lte": int(max_price)}}, {"price_range": {"$gte": 1}}]
    if min_rating:
        filters.append({"rating": {"$gte": float(min_rating)}})
    where = None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}

    # Location names vary ("DTLA" vs "Arts District (DTLA)"), so we filter them with a
    # simple case-insensitive "contains" check in Python on a larger candidate pool.
    pool_size = get_collection(RESTAURANTS).count() if location else k
    hits = _query(RESTAURANTS, embed_texts([query])[0], pool_size, where)
    if location:
        hits = [h for h in hits if location.lower() in h["location"].lower()]
    return hits[:k]


def search_recipes(query: str, k: int = 5, cuisine: str | None = None,
                   w_text: float | None = None, w_image: float | None = None) -> list[dict]:
    """Multimodal recipe search using late fusion of text and image similarity."""
    w_text = config.FUSION_WEIGHT_TEXT if w_text is None else w_text
    w_image = config.FUSION_WEIGHT_IMAGE if w_image is None else w_image
    pool = k * 3

    text_hits = _query(RECIPES_TEXT, embed_texts([query])[0], pool)
    image_hits = _query(RECIPE_IMAGES, embed_text_for_images(query), pool)

    # Normalise each modality separately, then combine by recipe id
    text_scores = _min_max({h["recipe_id"]: h["similarity"] for h in text_hits})
    image_scores = _min_max({h["recipe_id"]: h["similarity"] for h in image_hits})
    info = {h["recipe_id"]: h for h in text_hits + image_hits}

    results = []
    for recipe_id, meta in info.items():
        t, i = text_scores.get(recipe_id, 0.0), image_scores.get(recipe_id, 0.0)
        results.append({
            "recipe_id": recipe_id, "name": meta["name"], "cuisine": meta["cuisine"],
            "total_time": meta["total_time"], "image_path": meta["image_path"],
            "text_score": round(t, 3), "image_score": round(i, 3),
            "fused_score": round(w_text * t + w_image * i, 3),
        })
    if cuisine:
        results = [r for r in results if cuisine.lower() in r["cuisine"].lower()]
    results.sort(key=lambda r: r["fused_score"], reverse=True)
    return results[:k]


def search_recipes_by_image(image_path: str | Path, k: int = 5) -> list[dict]:
    """Find recipes whose photo looks most similar to the given image (CLIP image -> image)."""
    return _query(RECIPE_IMAGES, embed_images([str(image_path)])[0], k)


# ---------------------------------------------------------------------------
# Keep the index in sync when the database is edited (CLI / Gradio)
# ---------------------------------------------------------------------------
def upsert_restaurant(record: dict) -> None:
    get_collection(RESTAURANTS).upsert(
        ids=[str(record["itemId"])],
        embeddings=embed_texts([restaurant_to_text(record)]),
        documents=[restaurant_to_text(record)],
        metadatas=[restaurant_metadata(record)],
    )


def delete_restaurant(item_id: int) -> None:
    get_collection(RESTAURANTS).delete(ids=[str(item_id)])
