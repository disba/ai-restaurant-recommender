"""
Simple JSON "database" helpers for the restaurant records.

Used by the command-line manager (Module 1) and the Gradio app (Module 3),
so both share exactly the same add / edit / delete logic.
Every write first copies the current file into data/processed/backups/.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from restaurant_ai import config
from restaurant_ai.data_pipeline.schemas import RestaurantRecord


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json_with_backup(path: Path, data: list[dict]) -> Path | None:
    """Back up the existing file (timestamped), then write the new data."""
    backup_path = None
    if path.exists():
        config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = config.BACKUP_DIR / f"{path.stem}_{stamp}{path.suffix}"
        shutil.copy2(path, backup_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return backup_path


# ---------------------------------------------------------------------------
# Restaurant-specific helpers
# ---------------------------------------------------------------------------
def load_restaurants() -> list[dict]:
    return load_json(config.RESTAURANTS_JSON)


def next_item_id(restaurants: list[dict]) -> int:
    return max((r.get("itemId", 0) for r in restaurants), default=1000000) + 1


def add_restaurant_from_text(paragraph: str) -> dict:
    """Use the LLM to structure a new description, save it and return the new record."""
    # Imported here to avoid a circular import (structure_restaurants imports this file)
    from restaurant_ai.data_pipeline.structure_restaurants import structure_restaurant

    restaurants = load_restaurants()
    record = structure_restaurant(paragraph, next_item_id(restaurants))
    restaurants.append(record)
    save_json_with_backup(config.RESTAURANTS_JSON, restaurants)
    return record


def update_restaurant(item_id: int, changes: dict) -> dict:
    """Update some fields of one restaurant (itemId cannot be changed)."""
    restaurants = load_restaurants()
    for index, record in enumerate(restaurants):
        if record.get("itemId") == item_id:
            updates = {key: value for key, value in changes.items() if key != "itemId"}
            record = RestaurantRecord.model_validate({**record, **updates}).model_dump()
            restaurants[index] = record
            save_json_with_backup(config.RESTAURANTS_JSON, restaurants)
            return record
    raise KeyError(f"No restaurant with itemId {item_id}")


def delete_restaurant(item_id: int) -> dict:
    restaurants = load_restaurants()
    for i, record in enumerate(restaurants):
        if record.get("itemId") == item_id:
            removed = restaurants.pop(i)
            save_json_with_backup(config.RESTAURANTS_JSON, restaurants)
            return removed
    raise KeyError(f"No restaurant with itemId {item_id}")


def format_restaurant_card(r: dict) -> str:
    """Pretty, human-readable text for one restaurant."""
    price = "$" * r["price_range"] if isinstance(r.get("price_range"), int) and 1 <= r["price_range"] <= 4 else "?"
    lines = [
        f"{r.get('name', 'N/A')}  (itemId {r.get('itemId')})",
        f"  Location : {r.get('location') or 'N/A'}",
        f"  Type     : {r.get('type') or 'N/A'} | Cuisine: {r.get('food_style') or 'N/A'}",
        f"  Rating   : {r.get('rating') if r.get('rating') is not None else 'N/A'} | Price: {price}",
        f"  Vibe     : {r.get('vibe') or 'N/A'}",
        f"  Signature: {', '.join(r.get('signatures') or []) or 'N/A'}",
        f"  Setting  : {r.get('environment') or 'N/A'}",
    ]
    return "\n".join(lines)
