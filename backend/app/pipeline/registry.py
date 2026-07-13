"""
Pipeline stage registry.

Central lookup for stage factories used by the pipeline builder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from backend.app.pipeline.metadata import StageMetadata
from backend.app.pipeline.pipeline import (
    AlignmentStage,
    AssessmentStage,
    EmbeddingStage,
    FraudDetectionStage,
    ScrfdDetectionStage,
    SearchStage,
    VerificationStage,
)
from backend.app.pipeline.stage import PipelineStage

StageFactory: TypeAlias = Callable[[], PipelineStage]
StageRegistration: TypeAlias = type[PipelineStage] | StageFactory


class PipelineRegistry:
    """Registers and instantiates pipeline stages by name."""

    def __init__(self) -> None:
        """Initialize an empty stage registry."""
        self._registrations: dict[str, StageRegistration] = {}

    def register(
        self,
        name: str,
        factory: StageRegistration,
    ) -> None:
        """Register a stage class or factory under a unique name.

        Args:
            name: Unique stage identifier used in pipeline profiles.
            factory: Stage class or callable that returns a ``PipelineStage``.

        Raises:
            ValueError: If a stage with the same name is already registered
                or if the registered metadata name does not match ``name``.
            TypeError: If ``factory`` is neither a stage class nor callable.
        """
        if name in self._registrations:
            raise ValueError(f"Stage '{name}' is already registered.")

        if not (isinstance(factory, type) and issubclass(factory, PipelineStage)):
            if not callable(factory):
                raise TypeError(
                    "Stage registration must be a PipelineStage subclass "
                    "or a zero-argument factory callable."
                )

        metadata = self._resolve_metadata(factory)
        if metadata.name != name:
            raise ValueError(
                f"Registered stage name '{name}' does not match "
                f"metadata name '{metadata.name}'."
            )

        self._registrations[name] = factory

    def lookup(self, name: str) -> StageRegistration | None:
        """Look up a registered stage factory by name.

        Args:
            name: Unique stage identifier.

        Returns:
            Registered stage class or factory, or ``None`` if not found.
        """
        return self._registrations.get(name)

    def get(self, name: str) -> StageRegistration:
        """Return a registered stage factory by name.

        Args:
            name: Unique stage identifier.

        Raises:
            KeyError: If the stage is not registered.
        """
        factory = self.lookup(name)
        if factory is None:
            raise KeyError(f"Stage '{name}' is not registered.")
        return factory

    def get_metadata(self, name: str) -> StageMetadata:
        """Return static metadata for a registered stage without instantiation.

        Args:
            name: Unique stage identifier.

        Returns:
            Static ``StageMetadata`` for the registered stage.

        Raises:
            KeyError: If the stage is not registered.
            TypeError: If metadata cannot be resolved for the registration.
        """
        return self._resolve_metadata(self.get(name))

    def metadata_for_all(self) -> dict[str, StageMetadata]:
        """Return static metadata for every registered stage.

        Returns:
            Mapping of stage name to ``StageMetadata``.
        """
        return {name: self.get_metadata(name) for name in self.names()}

    def create(self, name: str) -> PipelineStage:
        """Instantiate a registered stage.

        Args:
            name: Unique stage identifier.

        Returns:
            New ``PipelineStage`` instance.

        Raises:
            KeyError: If the stage is not registered.
        """
        return self._instantiate(self.get(name))

    def names(self) -> list[str]:
        """Return all registered stage names."""
        return list(self._registrations.keys())

    @staticmethod
    def _instantiate(factory: StageRegistration) -> PipelineStage:
        """Create a stage instance from a class or factory."""
        if isinstance(factory, type) and issubclass(factory, PipelineStage):
            return factory()
        return factory()

    @staticmethod
    def _resolve_metadata(factory: StageRegistration) -> StageMetadata:
        """Resolve static metadata from a stage class or factory."""
        if isinstance(factory, type) and issubclass(factory, PipelineStage):
            return factory.metadata

        metadata = getattr(factory, "metadata", None)
        if isinstance(metadata, StageMetadata):
            return metadata

        raise TypeError(
            "Cannot resolve stage metadata without a PipelineStage subclass "
            "or a factory exposing a StageMetadata 'metadata' attribute."
        )


def create_default_registry() -> PipelineRegistry:
    """Create a registry populated with default stage factories.

    Returns:
        Registry containing SCRFD, alignment, assessment, and placeholder stages.
    """
    registry = PipelineRegistry()
    registry.register("scrfd", ScrfdDetectionStage)
    registry.register("alignment", AlignmentStage)
    registry.register("assessment", AssessmentStage)
    registry.register("fraud", FraudDetectionStage)
    registry.register("embedding", EmbeddingStage)
    registry.register("search", SearchStage)
    registry.register("verification", VerificationStage)
    return registry
