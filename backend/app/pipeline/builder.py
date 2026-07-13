"""
Pipeline builder.

Constructs executable pipelines from configuration-backed profiles or
custom stage lists.
"""

from backend.app.config.configuration import PipelineProfileSearchSettings
from backend.app.pipeline.config import load_pipeline_profile_settings
from backend.app.pipeline.pipeline import Pipeline
from backend.app.pipeline.profile import PipelineProfile
from backend.app.pipeline.registry import PipelineRegistry

_DEFAULT_CUSTOM_SEARCH = PipelineProfileSearchSettings(enabled=True, top_k=1)


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
            ValueError: If a custom profile is requested without stages or if
                stage dependencies are not satisfied.
            KeyError: If a referenced profile or stage is not defined.
        """
        stage_names = self._resolve_stage_names(profile, custom_stages)
        self._validate_stage_dependencies(stage_names, profile=profile)
        stages = tuple(self._registry.create(name) for name in stage_names)
        search_settings = self._resolve_search_settings(profile)
        return Pipeline(profile=profile, stages=stages, search=search_settings)

    def _validate_stage_dependencies(
        self,
        stage_names: tuple[str, ...],
        *,
        profile: PipelineProfile,
    ) -> None:
        """Validate that each stage's requirements are satisfied by earlier stages.

        Args:
            stage_names: Ordered stage names for the pipeline.
            profile: Pipeline profile being constructed.

        Raises:
            ValueError: If a stage requires capabilities not provided earlier.
        """
        available_capabilities = {"image"}

        for stage_name in stage_names:
            metadata = self._registry.get_metadata(stage_name)
            missing = [
                requirement
                for requirement in metadata.requires
                if requirement not in available_capabilities
            ]
            if missing:
                raise ValueError(
                    f"Invalid pipeline configuration for profile "
                    f"'{profile.value}': stage '{stage_name}' requires "
                    f"{missing}, but earlier stages only provide "
                    f"{sorted(available_capabilities)}."
                )

            available_capabilities.update(metadata.provides)

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
        return tuple(configured.stages)

    def _resolve_search_settings(
        self,
        profile: PipelineProfile,
    ) -> PipelineProfileSearchSettings | None:
        """Resolve search policy for a built-in profile.

        Args:
            profile: Built-in pipeline profile.

        Returns:
            Profile search settings, or ``None`` for custom profiles.
        """
        if profile is PipelineProfile.CUSTOM:
            return _DEFAULT_CUSTOM_SEARCH

        settings = load_pipeline_profile_settings()
        configured = settings.profiles.get(profile.value)
        if configured is None:
            raise KeyError(
                f"Profile '{profile.value}' is not defined in pipeline configuration."
            )
        return configured.search
