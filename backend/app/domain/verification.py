"""
Verification stage domain models.

Stores outputs produced by the Verification Engine. Verification results
are kept separate from ``Face``; the pipeline attaches them to
``PipelineContext.metadata``.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class VerificationDecision(str, Enum):
    """Biometric verification decision."""

    UNKNOWN = "unknown"
    ACCEPT = "accept"
    REJECT = "reject"


def resolve_is_verified(decision: VerificationDecision) -> bool:
    """Derive ``is_verified`` from a verification decision.

    Invariant:
        - ``ACCEPT`` -> ``True``
        - ``REJECT`` -> ``False``
        - ``UNKNOWN`` -> ``False``
    """
    return decision is VerificationDecision.ACCEPT


class VerificationResult(BaseModel):
    """Verification output for a single probe face.

    Represents one accept/reject decision produced after comparing probe
    embeddings against search candidates. This model must not include probe
    embeddings, gallery vectors, or pipeline state.

    ``is_verified`` is derived from ``decision`` and must never diverge:

    - ``decision == ACCEPT`` -> ``is_verified is True``
    - ``decision == REJECT`` -> ``is_verified is False``
    - ``decision == UNKNOWN`` -> ``is_verified is False``
    """

    decision: VerificationDecision
    matched_identity_id: str | None = None
    similarity_score: float
    threshold: float
    is_verified: bool
    verification_time_ms: float
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional verification metadata for downstream decision stages.",
    )

    @model_validator(mode="after")
    def validate_decision_invariant(self) -> "VerificationResult":
        """Ensure ``is_verified`` always matches ``decision``."""
        expected = resolve_is_verified(self.decision)
        if self.is_verified is not expected:
            raise ValueError(
                "Verification invariant violated: "
                f"decision={self.decision.value!r} requires "
                f"is_verified={expected}, got {self.is_verified}."
            )
        return self
