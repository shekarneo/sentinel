"""Shared API schema types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard API error payload."""

    code: str
    message: str


class HealthResponse(BaseModel):
    """Basic service health response."""

    status: str = "ok"
    api_version: str


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


class TimingsResponse(BaseModel):
    """Per-stage execution timings in seconds."""

    stages: dict[str, float] = Field(default_factory=dict)


class MetadataResponse(BaseModel):
    """Opaque metadata bag for API responses."""

    data: dict[str, Any] = Field(default_factory=dict)
