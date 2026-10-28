"""
Module 2 - Build the multimodal vector index (ChromaDB).

Reads the processed JSON files from Module 1 + the recipe images and stores
their embeddings in vector_store/. Re-running it rebuilds everything.

Run:
    python -m restaurant_ai.rag.build_index
"""

import shutil

from restaurant_ai import config
from restaurant_ai.data_pipeline.store import load_json
from restaurant_ai.rag import retriever
from restaurant_ai.rag.embeddings import embed_images, embed_texts


def main():
    # Start from a clean folder so re-runs don't keep stale data
    if config.VECTOR_DB_DIR.exists():
        shutil.rmtree(config.VECTOR_DB_DIR)
    retriever.get_client.cache_clear()

    # 1) Restaurants (text)
    restaurants = [r for r in load_json(config.RESTAURANTS_JSON) if r.get("name")]
    docs = [retriever.restaurant_to_text(r) for r in restaurants]
    retriever.get_collection(retriever.RESTAURANTS).add(
        ids=[str(r["itemId"]) for r in restaurants],
        embeddings=embed_texts(docs),
        documents=docs,
        metadatas=[retriever.restaurant_metadata(r) for r in restaurants],
    )
    print(f"Indexed {len(restaurants)} restaurants")

    # 2) Recipes (text: name + ingredients + image caption)
    recipes = load_json(config.RECIPES_JSON)
    docs = [retriever.recipe_to_text(rec) for rec in recipes]
    retriever.get_collection(retriever.RECIPES_TEXT).add(
        ids=[str(rec["id"]) for rec in recipes],
        embeddings=embed_texts(docs),
        documents=docs,
        metadatas=[retriever.recipe_metadata(rec) for rec in recipes],
    )
    print(f"Indexed {len(recipes)} recipe texts")

    # 3) Recipe photos (CLIP image embeddings)
    with_images = [rec for rec in recipes if (config.RECIPE_IMAGES_DIR / f"recipe{rec['id']}.png").exists()]
    paths = [str(config.RECIPE_IMAGES_DIR / f"recipe{rec['id']}.png") for rec in with_images]
    retriever.get_collection(retriever.RECIPE_IMAGES).add(
        ids=[str(rec["id"]) for rec in with_images],
        embeddings=embed_images(paths),
        documents=[rec["name"] for rec in with_images],
        metadatas=[retriever.recipe_metadata(rec) for rec in with_images],
    )
    print(f"Indexed {len(with_images)} recipe images")
    print(f"Vector store ready at {config.VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
