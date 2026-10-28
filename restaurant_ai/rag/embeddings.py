"""
Embedding models (Module 2). Both are free and run locally.

- Text  : all-MiniLM-L6-v2  -> 384-number vector for any text
- Images: OpenAI CLIP ViT-B/32 -> 512-number vector for an image OR a text.
          CLIP puts images and text in the SAME vector space, which is what
          lets us search food photos with a text query ("spicy noodles").

Models are loaded once (lazily) and reused - loading takes a few seconds.
"""

from functools import lru_cache

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from restaurant_ai import config


@lru_cache(maxsize=1)
def _text_model() -> SentenceTransformer:
    return SentenceTransformer(config.TEXT_EMBED_MODEL)


@lru_cache(maxsize=1)
def _clip() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).eval()
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    return model, processor


def _normalize(features) -> list[list[float]]:
    # Newer transformers versions return an output object; the vectors are in "pooler_output"
    vectors = getattr(features, "pooler_output", features)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    return vectors.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """MiniLM text embeddings (normalized, so cosine similarity works well)."""
    return _text_model().encode(texts, normalize_embeddings=True).tolist()


@torch.no_grad()
def embed_images(image_paths: list[str], batch_size: int = 16) -> list[list[float]]:
    """CLIP image embeddings (processed in small batches to save memory)."""
    model, processor = _clip()
    vectors = []
    for start in range(0, len(image_paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in image_paths[start:start + batch_size]]
        inputs = processor(images=images, return_tensors="pt")
        vectors += _normalize(model.get_image_features(**inputs))
    return vectors


@torch.no_grad()
def embed_text_for_images(text: str) -> list[float]:
    """CLIP *text* embedding - used to compare a text query against food images."""
    model, processor = _clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    return _normalize(model.get_text_features(**inputs))[0]
