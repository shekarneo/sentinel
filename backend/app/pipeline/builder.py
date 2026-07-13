"""
Pipeline builder.

Constructs executable pipelines from configuration-backed profiles or
custom stage lists.
"""

from backend.app.pipeline.config import load_pipeline_profile_settings
from backend.app.pipeline.pipeline import Pipeline
from backend.app.pipeline.profile import PipelineProfile
from backend.app.pipeline.registry import PipelineRegistry


class PipelineBuilder:
    """Builds pipelines from profiles using a stage registry."""

    def __init__(self, registry: PipelineRegistry) -> None:
        """Initialize the builder.

        Args:
            registry: Registry used to resolve and instantiate stage adapters.
        """
        self._registry = registry

    def build(
        self,
        profile: PipelineProfile,
        custom_stages: list[str] | None = None,
    ) -> Pipeline:
        """Build a pipeline for the given profile.

        Args:
            profile: Built-in or custom pipeline profile.
            custom_stages: Required when ``profile`` is ``CUSTOM``. Ordered
                list of registered stage names.

        Returns:
            Resolved pipeline ready for execution.

        Raises:
            ValueError: If a custom profile is requested without stages.
            KeyError: If a referenced profile or stage is not defined.
        """
        stage_names = self._resolve_stage_names(profile, custom_stages)
        stages = tuple(self._registry.create(name) for name in stage_names)
        return Pipeline(profile=profile, stages=stages)

    def _resolve_stage_names(
        self,
        profile: PipelineProfile,
        custom_stages: list[str] | None,
    ) -> tuple[str, ...]:
        """Resolve configured or custom stage names for a profile.

        Args:
            profile: Built-in or custom pipeline profile.
            custom_stages: Caller-defined stage list for ``CUSTOM`` profiles.

        Returns:
            Ordered stage names for the requested profile.

        Raises:
            ValueError: If a custom profile is requested without stages.
            KeyError: If the profile is not defined in configuration.
        """
        if profile is PipelineProfile.CUSTOM:
            if not custom_stages:
                raise ValueError(
                    "CUSTOM profile requires an explicit list of stage names."
                )
            return tuple(custom_stages)

        settings = load_pipeline_profile_settings()
        configured = settings.profiles.get(profile.value)
        if configured is None:
            raise KeyError(
                f"Profile '{profile.value}' is not defined in pipeline configuration."
            )
        return tuple(configured)
