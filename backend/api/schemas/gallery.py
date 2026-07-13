"""Gallery API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GalleryIdentityResponse(BaseModel):
    """Single gallery identity summary."""

    identity_id: str
    metadata: dict[str, Any] | None = None


class GalleryEntryResponse(BaseModel):
    """Gallery identity entry with metadata."""

    identity_id: str
    metadata: dict[str, Any] | None = None
    enrollment_count: int = 1


class GalleryListResponse(BaseModel):
    """Gallery listing response."""

    count: int
    identities: list[str]
    entries: list[GalleryEntryResponse] = Field(default_factory=list)


class GalleryDetailResponse(BaseModel):
    """Gallery detail response for a single identity."""

    identity: GalleryIdentityResponse


class GalleryRebuildResponse(BaseModel):
    """Gallery rebuild operation response."""

    message: str
    count: int
