"""
Fraud detection stage domain model.

Stores outputs produced by the Biometric Fraud Detection module.
"""

from pydantic import BaseModel


class FraudData(BaseModel):
    """Outputs from the Biometric Fraud Detection stage.

    Populated by the fraud detection module and attached to ``Face.fraud``.
    Captures presentation attack and manipulation signals for a face.
    """

    fraud_score: float | None = None
    is_attack: bool | None = None
    attack_type: str | None = None
