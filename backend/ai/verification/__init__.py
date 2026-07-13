"""
Verification engine module.

Evaluates search candidates for probe faces and returns ``VerificationResult``
objects for downstream decision stages.
"""

from backend.ai.verification.engine import (
    ThresholdVerificationEngine,
    VerificationEngine,
)
from backend.ai.verification.verifier import FaceVerifier, verify

__all__ = [
    "FaceVerifier",
    "ThresholdVerificationEngine",
    "VerificationEngine",
    "verify",
]
