"""
Module 1 / Step 2 - Caption food images with a vision LLM.

For every recipe in data/raw/recipes.json we look at its picture
(data/raw/recipe_images/recipe<ID>.png) and ask the vision model to describe
it. The caption is stored as a new field "image_description".

Run:
    python -m restaurant_ai.data_pipeline.caption_recipes --limit 3
"""

import argparse

from restaurant_ai import config
from restaurant_ai.data_pipeline.store import load_json, save_json_with_backup
from restaurant_ai.llm import ask_vision_llm

SYSTEM_MSG = (
    "You are an expert food image analyst. Describe only what you can see in the image. "
    "Be concise and factual. Do not invent ingredients."
)


def caption_prompt(food_name: str) -> str:
    return (
        f"The dish in this image is labeled '{food_name}'. First say whether the image matches the label, "
        "then describe it in 3-4 sentences: main visible ingredients and colors, cooking style and texture, "
        "and the plating / portion size. Return a single paragraph."
    )


def main():
    parser = argparse.ArgumentParser(description="Add image captions to recipes")
    parser.add_argument("--limit", type=int, default=None, help="Only caption the first N recipes")
    args = parser.parse_args()

    recipes = load_json(config.RECIPES_RAW_JSON)
    to_process = recipes[: args.limit]
    print(f"Captioning {len(to_process)} recipe images with {config.VISION_LLM_MODEL} ...")

    for i, recipe in enumerate(to_process):
        image_path = config.RECIPE_IMAGES_DIR / f"recipe{recipe['id']}.png"
        if not image_path.exists():
            print(f"[{i + 1}/{len(to_process)}] WARNING: missing image {image_path}")
            recipe["image_description"] = ""
            continue
        recipe["image_description"] = ask_vision_llm(SYSTEM_MSG, caption_prompt(recipe["name"]), image_path)
        print(f"[{i + 1}/{len(to_process)}] {recipe['name']}")

    # A --limit test run writes to a sample file so the full dataset is never overwritten by accident
    output = config.PROCESSED_DIR / "recipes_sample.json" if args.limit else config.RECIPES_JSON
    save_json_with_backup(output, to_process)
    print(f"Saved {len(to_process)} recipes to {output}")


if __name__ == "__main__":
    main()
