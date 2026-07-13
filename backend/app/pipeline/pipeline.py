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
from backend.ai.embedding.embedder import FaceEmbedder
from backend.ai.search.searcher import FaceSearcher
from backend.ai.verification.verifier import FaceVerifier
from backend.app.config.configuration import (
    PipelineProfileDefinition,
    PipelineProfileSearchSettings,
    resolve_scrfd_model_path,
)
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
    search: PipelineProfileSearchSettings | None = None


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
        supports_warmup=False,
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
        supports_warmup=False,
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
        supports_warmup=False,
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
        supports_warmup=False,
        supports_batching=False,
    )

    def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("Fraud detection stage is not yet implemented; skipping.")
        _stage_results(context)[self.metadata.name] = {"status": "not_implemented"}
        return context


class EmbeddingStage(PipelineStage):
    """Adapter for the embedding engine module."""

    metadata = StageMetadata(
        name="embedding",
        version="1.0.0",
        description="Generate biometric embeddings from aligned faces.",
        requires=("image", "faces", "alignment"),
        provides=("embedding",),
        gpu_required=True,
        supports_warmup=True,
        supports_batching=True,
    )

    def __init__(self, embedder: FaceEmbedder | None = None) -> None:
        """Initialize the embedding stage.

        Args:
            embedder: Optional pre-initialized face embedder.
        """
        self._embedder = embedder or FaceEmbedder()
        self._initialized = False

    def initialize(self) -> None:
        """Load and warm up the embedding provider before execution."""
        if self._initialized:
            return

        self._embedder.warmup()
        self._initialized = True
        logger.debug("Embedding stage initialized.")

    def execute(self, context: PipelineContext) -> PipelineContext:
        self._embedder.embed(context.faces)
        logger.debug("Embedded %d face(s).", len(context.faces))
        return context


PROFILE_SEARCH_METADATA_KEY = "profile_search"


class SearchStage(PipelineStage):
    """Adapter for the search engine module."""

    metadata = StageMetadata(
        name="search",
        version="1.0.0",
        description="Search gallery embeddings and return match candidates.",
        requires=("faces", "embedding"),
        provides=("search",),
        gpu_required=False,
        supports_warmup=False,
        supports_batching=False,
    )

    def __init__(self, searcher: FaceSearcher | None = None) -> None:
        """Initialize the search stage.

        Args:
            searcher: Optional pre-initialized face searcher.
        """
        self._searcher = searcher or FaceSearcher()

    def execute(self, context: PipelineContext) -> PipelineContext:
        profile_search = context.metadata.get(PROFILE_SEARCH_METADATA_KEY)
        if profile_search is None:
            raise ValueError(
                "Search stage requires profile search configuration in "
                f"context.metadata['{PROFILE_SEARCH_METADATA_KEY}']."
            )

        if not profile_search.get("enabled", False):
            raise ValueError(
                "Search stage cannot execute when profile search is disabled."
            )

        top_k = profile_search.get("top_k")
        if top_k is None:
            raise ValueError(
                "Search stage requires profile search top_k configuration."
            )

        search_results = self._searcher.search(context.faces, top_k=int(top_k))
        context.metadata["search_results"] = search_results
        logger.debug(
            "Searched %d face(s) with top_k=%d; stored results in metadata.",
            len(search_results),
            top_k,
        )
        return context


class VerificationStage(PipelineStage):
    """Adapter for the verification engine module."""

    metadata = StageMetadata(
        name="verification",
        version="1.0.0",
        description="Verify identity and produce an accept or reject decision.",
        requires=("faces", "embedding", "search"),
        provides=("verification",),
        gpu_required=False,
        supports_warmup=False,
        supports_batching=False,
    )

    def __init__(self, verifier: FaceVerifier | None = None) -> None:
        """Initialize the verification stage.

        Args:
            verifier: Optional pre-initialized face verifier.
        """
        self._verifier = verifier or FaceVerifier()

    def execute(self, context: PipelineContext) -> PipelineContext:
        search_results = context.metadata.get("search_results")
        if search_results is None:
            raise ValueError(
                "Verification requires search_results in context.metadata."
            )

        verification_results = self._verifier.verify(context.faces, search_results)
        context.metadata["verification"] = verification_results
        logger.debug(
            "Verified %d face(s); stored results in metadata.",
            len(verification_results),
        )
        return context
