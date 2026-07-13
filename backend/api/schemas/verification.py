"""Verification API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VerificationRequest(BaseModel):
    """Verification request contract.

    Probe faces are produced by recognition; search candidates are supplied
  by the search stage or a dedicated search call.
    """

    profile: str = Field(
        default="kyc",
        description="Pipeline profile that includes search and verification.",
    )


class VerificationResponse(BaseModel):
    """Verification response contract.

    Internal ``VerificationResult`` domain objects are never returned directly.
    """

    decision: str
    matched_identity_id: str | None = None
    similarity_score: float
    threshold: float
    is_verified: bool
    verification_time_ms: float
    metadata: dict[str, Any] | None = None
