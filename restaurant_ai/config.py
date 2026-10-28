"""
Central configuration for the whole project.

All settings come from environment variables (loaded from the `.env` file),
with sensible defaults so the project runs out-of-the-box on a local machine
with Ollama installed.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains this package
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# Quieter console output from the ML libraries
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


def _env(name: str, default: str) -> str:
    """Read an environment variable, falling back to a default value."""
    value = os.getenv(name, "").strip()
    return value or default


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BACKUP_DIR = PROCESSED_DIR / "backups"

# Raw inputs (Module 1)
RESTAURANT_GUIDE_TXT = RAW_DIR / "restaurant_guide.txt"
RECIPES_RAW_JSON = RAW_DIR / "recipes.json"
RECIPE_IMAGES_DIR = RAW_DIR / "recipe_images"
USER_REVIEWS_RAW_JSON = RAW_DIR / "user_reviews.json"

# Processed outputs (Module 1) -> used by Module 2, 3 and 4
RESTAURANTS_JSON = PROCESSED_DIR / "restaurants.json"
RESTAURANT_SCHEMA_JSON = PROCESSED_DIR / "restaurant.schema.json"
RECIPES_JSON = PROCESSED_DIR / "recipes_enriched.json"
REVIEWS_JSON = PROCESSED_DIR / "reviews_enriched.json"
REVIEW_IMAGES_DIR = PROCESSED_DIR / "review_images"

# Vector database (Module 2)
VECTOR_DB_DIR = ROOT_DIR / "vector_store"

# Evaluation results
EVAL_DIR = ROOT_DIR / "evals"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ---------------------------------------------------------------------------
# LLM settings
# Provider can be "ollama" (local, free) or "openai_compatible" (any cloud API
# that speaks the OpenAI protocol: OpenAI, Groq, OpenRouter, Together, ...).
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(_env("OLLAMA_NUM_CTX", "8192"))

TEXT_LLM_PROVIDER = _env("TEXT_LLM_PROVIDER", "ollama")
TEXT_LLM_MODEL = _env("TEXT_LLM_MODEL", "qwen3:8b")
TEXT_LLM_API_KEY = _env("TEXT_LLM_API_KEY", "")
TEXT_LLM_BASE_URL = _env("TEXT_LLM_BASE_URL", "https://api.openai.com/v1")

VISION_LLM_PROVIDER = _env("VISION_LLM_PROVIDER", "ollama")
VISION_LLM_MODEL = _env("VISION_LLM_MODEL", "qwen2.5vl:3b")
VISION_LLM_API_KEY = _env("VISION_LLM_API_KEY", "")
VISION_LLM_BASE_URL = _env("VISION_LLM_BASE_URL", "https://api.openai.com/v1")

# ---------------------------------------------------------------------------
# Embedding models (free, run locally via sentence-transformers)
# ---------------------------------------------------------------------------
TEXT_EMBED_MODEL = _env("TEXT_EMBED_MODEL", "all-MiniLM-L6-v2")
CLIP_MODEL = _env("CLIP_MODEL", "openai/clip-vit-base-patch32")

# Late-fusion weights for recipe search (text vs image similarity)
FUSION_WEIGHT_TEXT = float(_env("FUSION_WEIGHT_TEXT", "0.6"))
FUSION_WEIGHT_IMAGE = float(_env("FUSION_WEIGHT_IMAGE", "0.4"))

# ---------------------------------------------------------------------------
# Observability (Arize Phoenix - open source, runs locally)
# ---------------------------------------------------------------------------
TRACING_ENABLED = _env("TRACING_ENABLED", "true").lower() == "true"
PHOENIX_URL = _env("PHOENIX_URL", "http://localhost:6006")
PHOENIX_PROJECT = _env("PHOENIX_PROJECT", "restaurant-recommender")
