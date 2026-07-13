"""
Pipeline execution context.

The context object flows through every pipeline stage. ``Face`` instances
remain the primary domain objects; they are referenced from the context
but are not replaced by pipeline-specific wrappers.

The public API is intentionally frozen. Only the following top-level fields
are permitted: ``image``, ``faces``, ``profile``, ``metadata``, ``timings``,
and ``errors``. Any additional runtime information must be stored inside
``metadata`` rather than introducing new top-level fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.app.domain.face import Face
from backend.app.pipeline.profile import PipelineProfile


@dataclass
class PipelineContext:
    """Mutable state passed through the biometric pipeline.

    Attributes:
        image: Source image in HWC BGR layout.
        faces: Detected and enriched ``Face`` objects flowing through stages.
        profile: Active pipeline profile for the current execution.
        metadata: Arbitrary execution metadata supplied by callers or stages.
        timings: Per-stage execution durations in seconds.
        errors: Recorded stage failures when graceful failure is applied.
    """

    image: np.ndarray
    faces: list[Face] = field(default_factory=list)
    profile: PipelineProfile = PipelineProfile.ENROLLMENT
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
