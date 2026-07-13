"""
Pipeline model and stage adapters.

Concrete stage adapters wrap frozen AI modules without modifying them.
Placeholder stages reserve extension points for future modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.ai.alignment.aligner import FaceAligner
from backend.ai.assessment.assessor import FaceAssessor
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.app.config.configuration import resolve_scrfd_model_path
from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.metadata import StageMetadata
from backend.app.pipeline.profile import PipelineProfile
from backend.app.pipeline.stage import PipelineStage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline:
    """A resolved sequence of stages for a pipeline profile."""

    profile: PipelineProfile
    stages: tuple[PipelineStage, ...]


def _stage_results(context: PipelineContext) -> dict[str, Any]:
    """Return the metadata bucket used for stage-level runtime outputs."""
    results = context.metadata.setdefault("results", {})
    if not isinstance(results, dict):
        raise ValueError("context.metadata['results'] must be a dictionary.")
    return results


class ScrfdDetectionStage(PipelineStage):
    """Adapter for the frozen SCRFD face detection module."""

    metadata = StageMetadata(
        name="scrfd",
        version="1.0.0",
        description="Detect faces and five-point landmarks using SCRFD.",
        requires=("image",),
        provides=("faces",),
        gpu_required=True,
        supports_batching=False,
    )

    def __init__(self, detector: SCRFDDetector | None = None) -> None:
        """Initialize the detection stage.

        Args:
            detector: Optional pre-initialized SCRFD detector.
        """
        self._detector = detector

    def execute(self, context: PipelineContext) -> PipelineContext:
        detector = self._detector or SCRFDDetector(resolve_scrfd_model_path())
        context.faces = detector.detect(context.image)
        logger.debug("SCRFD detected %d face(s).", len(context.faces))
        return context


class AlignmentStage(PipelineStage):
    """Adapter for the frozen face alignment module."""

    metadata = StageMetadata(
        name="alignment",
        version="1.0.0",
        description="Align detected faces to a canonical template.",
        requires=("faces",),
        provides=("alignment",),
        gpu_required=False,
        supports_batching=False,
    )

    def __init__(self, aligner: FaceAligner | None = None) -> None:
        """Initialize the alignment stage.

        Args:
            aligner: Optional pre-initialized face aligner.
        """
        self._aligner = aligner or FaceAligner()

    def execute(self, context: PipelineContext) -> PipelineContext:
        self._aligner.align(context.image, context.faces)
        logger.debug("Aligned %d face(s).", len(context.faces))
        return context


class AssessmentStage(PipelineStage):
    """Adapter for the frozen face assessment module."""

    metadata = StageMetadata(
        name="assessment",
        version="1.0.0",
        description="Evaluate aligned face quality for biometric suitability.",
        requires=("image", "faces", "alignment"),
        provides=("assessment",),
        gpu_required=False,
        supports_batching=False,
    )

    def __init__(self, assessor: FaceAssessor | None = None) -> None:
        """Initialize the assessment stage.

        Args:
            assessor: Optional pre-initialized face assessor.
        """
        self._assessor = assessor or FaceAssessor()

    def execute(self, context: PipelineContext) -> PipelineContext:
        self._assessor.assess(context.image, context.faces)
        logger.debug("Assessed %d face(s).", len(context.faces))
        return context


class FraudDetectionStage(PipelineStage):
    """Placeholder for the biometric fraud detection module."""

    metadata = StageMetadata(
        name="fraud",
        version="1.0.0",
        description="Detect presentation attacks and manipulated biometric samples.",
        requires=("image", "faces", "alignment"),
        provides=("fraud",),
        gpu_required=True,
        supports_batching=False,
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("Fraud detection stage is not yet implemented; skipping.")
        _stage_results(context)[self.metadata.name] = {"status": "not_implemented"}
        return context


class EmbeddingStage(PipelineStage):
    """Placeholder for the embedding service module."""

    metadata = StageMetadata(
        name="embedding",
        version="1.0.0",
        description="Generate biometric embeddings from aligned faces.",
        requires=("image", "faces", "alignment"),
        provides=("embedding",),
        gpu_required=True,
        supports_batching=True,
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("Embedding stage is not yet implemented; skipping.")
        _stage_results(context)[self.metadata.name] = {"status": "not_implemented"}
        return context


class SearchStage(PipelineStage):
    """Placeholder for the search and matching module."""

    metadata = StageMetadata(
        name="search",
        version="1.0.0",
        description="Search gallery embeddings and return match candidates.",
        requires=("faces", "embedding"),
        provides=("search",),
        gpu_required=False,
        supports_batching=False,
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("Search stage is not yet implemented; skipping.")
        _stage_results(context)[self.metadata.name] = {"status": "not_implemented"}
        return context


class VerificationStage(PipelineStage):
    """Placeholder for the verification and decision module."""

    metadata = StageMetadata(
        name="verification",
        version="1.0.0",
        description="Verify identity and produce an accept or reject decision.",
        requires=("faces", "embedding"),
        provides=("verification",),
        gpu_required=False,
        supports_batching=False,
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("Verification stage is not yet implemented; skipping.")
        _stage_results(context)[self.metadata.name] = {"status": "not_implemented"}
        return context
