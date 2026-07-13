"""Recognition application service.

The API layer must call ``RecognitionService`` instead of the pipeline
orchestrator directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.pipeline.builder import PipelineBuilder
from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.executor import PipelineExecutor
from backend.app.pipeline.profile import PipelineProfile
from backend.app.pipeline.registry import PipelineRegistry, create_default_registry


class RecognitionService:
    """Runs biometric recognition through the pipeline orchestrator."""

    def __init__(
        self,
        builder: PipelineBuilder | None = None,
        executor: PipelineExecutor | None = None,
        *,
        registry: PipelineRegistry | None = None,
    ) -> None:
        """Initialize the recognition service.

        Args:
            builder: Optional pipeline builder for dependency injection.
            executor: Optional pipeline executor for dependency injection.
            registry: Optional stage registry used when ``builder`` is omitted.
        """
        resolved_registry = registry or create_default_registry()
        self._builder = builder or PipelineBuilder(resolved_registry)
        self._executor = executor or PipelineExecutor()

    def recognize(
        self,
        image: np.ndarray,
        profile: PipelineProfile,
    ) -> PipelineContext:
        """Execute a biometric pipeline for the supplied image.

        Args:
            image: Source image in HWC BGR layout.
            profile: Pipeline profile to execute.

        Returns:
            Completed pipeline context.
        """
        pipeline = self._builder.build(profile)
        context = PipelineContext(image=image, profile=profile)
        return self._executor.execute(pipeline, context)

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """Decode image bytes into an OpenCV BGR array.

        Args:
            image_bytes: Encoded image bytes.

        Returns:
            Decoded image array.

        Raises:
            ValueError: If the bytes do not represent a valid image.
        """
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image bytes.")
        return image
