"""
Pipeline stage interface.

Every biometric module is adapted to this interface by the pipeline layer.
AI modules must not invoke one another directly.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.metadata import StageMetadata


class PipelineStage(ABC):
    """Abstract pipeline stage executed by the orchestrator.

    Subclasses expose static ``metadata`` describing stage capabilities.
    Metadata is self-describing and may be queried without instantiation.
    """

    metadata: ClassVar[StageMetadata]

    @property
    def name(self) -> str:
        """Return the unique stage identifier from static metadata."""
        return self.metadata.name

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Run the stage and return the updated pipeline context.

        Args:
            context: Current pipeline execution state.

        Returns:
            Updated pipeline context.
        """
