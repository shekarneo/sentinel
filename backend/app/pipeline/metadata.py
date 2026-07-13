"""
Pipeline stage metadata.

Static, self-describing metadata for pipeline stages. This model must not
include latency, thresholds, configuration, or runtime state.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageMetadata:
    """Static metadata describing a pipeline stage.

    Attributes:
        name: Unique stage identifier used in profiles and the registry.
        version: Semantic version of the stage adapter.
        description: Human-readable summary of the stage responsibility.
        requires: Capability or context inputs required before execution.
        provides: Capability or context outputs produced after execution.
        gpu_required: Whether the stage expects GPU acceleration.
        supports_warmup: Whether the stage exposes a warm-up lifecycle hook.
        supports_batching: Whether the stage supports batched execution.
    """

    name: str
    version: str
    description: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    gpu_required: bool
    supports_warmup: bool
    supports_batching: bool
