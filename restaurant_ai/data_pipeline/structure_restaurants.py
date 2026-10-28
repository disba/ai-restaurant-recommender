"""
Module 1 / Step 1 - Turn unstructured restaurant text into structured JSON.

Flow for each restaurant paragraph:
    paragraph -> LLM (few-shot prompt) -> Pydantic validation
              -> if invalid: LLM "auto-repair" prompt -> validate again (max 3 tries)

Run:
    python -m restaurant_ai.data_pipeline.structure_restaurants --limit 5
    python -m restaurant_ai.data_pipeline.structure_restaurants          # all (~10-15 min)
"""

import argparse
import json

from pydantic import ValidationError

from restaurant_ai import config
from restaurant_ai.data_pipeline.schemas import Restaurant, RestaurantRecord
from restaurant_ai.data_pipeline.store import save_json_with_backup
from restaurant_ai.llm import ask_llm

FIRST_ITEM_ID = 1000001  # matches the itemId values used in the user reviews
MAX_REPAIR_ATTEMPTS = 3

# One worked example ("few-shot") shows the LLM exactly what we want.
EXAMPLE_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** "
    "specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy "
    "of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making "
    "it a premier spot for open-air dining near the pier. Price range: $"
)
EXAMPLE_OUTPUT = {
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
    "vibe": "sun-drenched",
    "environment": "open-air dining near the pier",
    "shortcomings": [],
}


def load_restaurant_paragraphs() -> list[str]:
    """Read the raw guide and split it into one paragraph per restaurant."""
    text = config.RESTAURANT_GUIDE_TXT.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs[1:]  # the first block is just the title


def build_extraction_prompt(paragraph: str) -> tuple[str, str]:
    system_msg = (
        "You are a data extraction assistant. You convert restaurant descriptions into JSON. "
        "Return ONLY valid JSON, no explanations."
    )
    user_prompt = f"""Convert the restaurant description into JSON with exactly these keys:
name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings.

Rules:
- food_style is the cuisine (e.g. "Japanese", "Baja-style seafood"), not the mood.
- vibe is the mood/atmosphere (e.g. "moody", "cozy").
- price_range is the NUMBER of dollar signs ($ = 1 ... $$$$ = 4). Use null if not given.
- rating is a number between 0 and 5. Use null if not given.
- Use "" or [] when information is missing. Do not invent facts.

Example input:
{EXAMPLE_PARAGRAPH}

Example output:
{json.dumps(EXAMPLE_OUTPUT)}

Restaurant description:
{paragraph}"""
    return system_msg, user_prompt


def build_repair_prompt(bad_json: str, errors: str) -> tuple[str, str]:
    system_msg = (
        "You are a JSON repair assistant. Fix only the JSON syntax or schema errors listed. "
        "Keep the same field names and valid values. Return ONLY the corrected JSON."
    )
    user_prompt = f"JSON to repair:\n{bad_json}\n\nValidation errors:\n{errors}"
    return system_msg, user_prompt


def structure_restaurant(paragraph: str, item_id: int) -> dict:
    """Run extraction + validation + auto-repair for ONE paragraph and return a record dict."""
    system_msg, prompt = build_extraction_prompt(paragraph)
    candidate = ask_llm(system_msg, prompt, json_mode=True, temperature=0)

    for attempt in range(MAX_REPAIR_ATTEMPTS):
        try:
            restaurant = Restaurant.model_validate_json(candidate)
            record = RestaurantRecord(**restaurant.model_dump(), itemId=item_id)
            return record.model_dump()
        except ValidationError as error:
            print(f"   validation failed (attempt {attempt + 1}) - asking LLM to repair")
            repair_system, repair_prompt = build_repair_prompt(candidate, str(error))
            candidate = ask_llm(repair_system, repair_prompt, json_mode=True, temperature=0)

    raise ValueError(f"Could not produce valid JSON for item {item_id} after {MAX_REPAIR_ATTEMPTS} repairs")


def main():
    parser = argparse.ArgumentParser(description="Structure restaurant text into JSON with an LLM")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N restaurants")
    args = parser.parse_args()

    paragraphs = load_restaurant_paragraphs()[: args.limit]
    print(f"Structuring {len(paragraphs)} restaurants with {config.TEXT_LLM_MODEL} ...")

    records, failed = [], 0
    for i, paragraph in enumerate(paragraphs):
        try:
            records.append(structure_restaurant(paragraph, FIRST_ITEM_ID + i))
            print(f"[{i + 1}/{len(paragraphs)}] {records[-1]['name']}")
        except ValueError as error:
            failed += 1
            print(f"[{i + 1}/{len(paragraphs)}] SKIPPED: {error}")

    # A --limit test run writes to a sample file so the full dataset is never overwritten by accident
    output = config.PROCESSED_DIR / "restaurants_sample.json" if args.limit else config.RESTAURANTS_JSON
    save_json_with_backup(output, records)
    config.RESTAURANT_SCHEMA_JSON.write_text(json.dumps(RestaurantRecord.model_json_schema(), indent=2))
    print(f"Done. Saved {len(records)} records ({failed} failed) to {output}")


if __name__ == "__main__":
    main()
