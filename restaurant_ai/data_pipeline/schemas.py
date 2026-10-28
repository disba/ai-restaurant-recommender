"""
Pydantic schema for one structured restaurant record.

Pydantic checks that the JSON produced by the LLM has the right fields and
types. If validation fails, we ask the LLM to repair its own output.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Restaurant(BaseModel):
    """One restaurant, extracted from a free-text paragraph by the LLM."""

    name: str
    location: str = ""
    type: str = Field("", description="e.g. 'casual taqueria', 'upscale bistro'")
    food_style: str = Field("", description="Cuisine, e.g. 'Baja-style seafood'")
    rating: Optional[float] = Field(None, ge=0, le=5)
    price_range: Optional[int] = Field(None, ge=1, le=4, description="Number of $ signs (1-4)")
    signatures: List[str] = Field(default_factory=list, description="Signature dishes")
    vibe: Optional[str] = None
    environment: str = ""
    shortcomings: List[str] = Field(default_factory=list)


class RestaurantRecord(Restaurant):
    """A Restaurant as stored on disk: the same fields plus an itemId assigned by code (never by the LLM)."""

    itemId: int
