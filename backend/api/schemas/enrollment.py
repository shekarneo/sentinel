"""Enrollment API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EnrollmentRequest(BaseModel):
    """Enrollment request contract."""

    identity_id: str = Field(description="Unique gallery identity identifier.")
    metadata: dict[str, Any] | None = None


class EnrollmentResponse(BaseModel):
    """Enrollment response contract."""

    identity_id: str
    embedding_id: int
    message: str = "Identity enrolled."
