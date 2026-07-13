"""
Face assessment module.

Evaluates aligned face quality and populates ``Face.assessment`` with
``AssessmentData`` for downstream fraud detection and embedding stages.
"""

from backend.ai.assessment.assessor import FaceAssessor, assess

__all__ = [
    "FaceAssessor",
    "assess",
]
