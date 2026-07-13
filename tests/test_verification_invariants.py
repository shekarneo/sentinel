"""Verification domain invariant tests."""

import pytest

from backend.app.domain.verification import (
    VerificationDecision,
    VerificationResult,
    resolve_is_verified,
)


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (VerificationDecision.ACCEPT, True),
        (VerificationDecision.REJECT, False),
        (VerificationDecision.UNKNOWN, False),
    ],
)
def test_resolve_is_verified(decision: VerificationDecision, expected: bool) -> None:
    """Decision-to-flag mapping should remain stable."""
    assert resolve_is_verified(decision) is expected


def test_verification_result_rejects_mismatched_flag() -> None:
    """VerificationResult should enforce decision invariants."""
    with pytest.raises(ValueError, match="Verification invariant violated"):
        VerificationResult(
            decision=VerificationDecision.ACCEPT,
            matched_identity_id="user_a",
            similarity_score=0.99,
            threshold=0.45,
            is_verified=False,
            verification_time_ms=0.0,
        )
