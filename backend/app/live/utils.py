"""Live camera helpers."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from backend.app.domain.search import SearchResults
from backend.app.domain.verification import VerificationResult
from backend.app.live.models import LiveFaceOverlay, LiveFrameOverlay, LiveTimelineEntry
from backend.app.pipeline.context import PipelineContext


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def create_session_id() -> str:
    """Generate a unique live session identifier."""
    return str(uuid.uuid4())


def pipeline_latency_ms(context: PipelineContext) -> float:
    """Return total pipeline latency in milliseconds."""
    return round(sum(context.timings.values()) * 1000.0, 2)


def build_overlay(context: PipelineContext) -> LiveFrameOverlay:
    """Build overlay data from a completed pipeline context."""
    search_results = context.metadata.get("search_results")
    verification_results = context.metadata.get("verification")

    faces: list[LiveFaceOverlay] = []
    for index, face in enumerate(context.faces):
        identity_id = None
        similarity = None
        verification_decision = None

        if isinstance(search_results, list) and index < len(search_results):
            result = search_results[index]
            if isinstance(result, SearchResults) and result.results:
                top = result.results[0]
                identity_id = top.identity_id
                similarity = top.score
        elif isinstance(search_results, SearchResults) and index == 0 and search_results.results:
            top = search_results.results[0]
            identity_id = top.identity_id
            similarity = top.score

        if isinstance(verification_results, list) and index < len(verification_results):
            verification = verification_results[index]
            if isinstance(verification, VerificationResult):
                verification_decision = verification.decision.value
                if identity_id is None:
                    identity_id = verification.matched_identity_id
                if similarity is None:
                    similarity = verification.similarity_score

        faces.append(
            LiveFaceOverlay(
                bounding_box={
                    "x": face.bounding_box.x,
                    "y": face.bounding_box.y,
                    "width": face.bounding_box.width,
                    "height": face.bounding_box.height,
                },
                confidence=face.confidence,
                identity_id=identity_id,
                similarity=similarity,
                verification_decision=verification_decision,
            )
        )

    return LiveFrameOverlay(
        faces=faces,
        pipeline_time_ms=pipeline_latency_ms(context),
        profile=context.profile.value,
        face_count=len(context.faces),
    )


def build_timeline_entry(context: PipelineContext) -> LiveTimelineEntry:
    """Create a timeline entry from a pipeline context."""
    overlay = build_overlay(context)
    identities = [face.identity_id for face in overlay.faces if face.identity_id]
    return LiveTimelineEntry(
        timestamp=utc_now(),
        face_count=overlay.face_count,
        pipeline_time_ms=overlay.pipeline_time_ms,
        recognized_identities=identities,
    )


class LatencyTracker:
    """Track rolling average pipeline latency."""

    def __init__(self) -> None:
        self._samples: list[float] = []
        self._last_sample_monotonic = 0.0

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        if len(self._samples) > 100:
            self._samples.pop(0)
        self._last_sample_monotonic = time.monotonic()

    @property
    def average_ms(self) -> float:
        if not self._samples:
            return 0.0
        return round(sum(self._samples) / len(self._samples), 2)

    def processing_fps(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        elapsed = max(time.monotonic() - self._last_sample_monotonic, 0.001)
        return round(1.0 / elapsed, 2)
