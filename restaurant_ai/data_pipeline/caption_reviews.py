"""
Module 1 / Step 3 - Caption the photos attached to user reviews.

Each review has a list of image URLs. We download each image (cached in
data/processed/review_images/), then ask the vision LLM to describe it
*in the context of the written review*. Captions are stored in "image_captions".

Run:
    python -m restaurant_ai.data_pipeline.caption_reviews
"""

import ast
import time

import requests

from restaurant_ai import config
from restaurant_ai.data_pipeline.store import load_json, save_json_with_backup
from restaurant_ai.llm import ask_vision_llm

SYSTEM_MSG = (
    "You are a culinary expert analysing restaurant visit photos. Describe only what you can see, "
    "but interpret it in light of the user's written review. Be concise and factual."
)


def caption_prompt(review: dict) -> str:
    context = f"Title: {review.get('title', '')}. Text: {review.get('text', '')}. Rating: {review.get('rating', '')}"
    return (
        f'User review: "{context}"\n\n'
        "Describe this photo in 2-3 sentences: the dishes and presentation, the setting if visible, "
        "and how it matches the sentiment of the review. Return a single paragraph."
    )


def download_image(url: str, save_path, retries: int = 3) -> bool:
    """Download an image with a simple retry loop. Returns True on success."""
    if save_path.exists():
        return True  # already cached
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            save_path.write_bytes(response.content)
            return True
        except requests.RequestException as error:
            print(f"   download failed (attempt {attempt + 1}): {error}")
            time.sleep(2 ** attempt)
    return False


def main():
    reviews = load_json(config.USER_REVIEWS_RAW_JSON)
    config.REVIEW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for i, review in enumerate(reviews):
        # The "images" field is a string that looks like a Python list -> convert it
        urls = ast.literal_eval(review.get("images") or "[]")
        captions = []
        for j, url in enumerate(urls):
            image_path = config.REVIEW_IMAGES_DIR / f"review{i}_img{j}.jpg"
            if download_image(url, image_path):
                captions.append(ask_vision_llm(SYSTEM_MSG, caption_prompt(review), image_path))
        review["image_captions"] = captions
        print(f"[{i + 1}/{len(reviews)}] review {review['reviewId']}: {len(captions)} caption(s)")

    save_json_with_backup(config.REVIEWS_JSON, reviews)
    print(f"Saved {len(reviews)} reviews to {config.REVIEWS_JSON}")


if __name__ == "__main__":
    main()
