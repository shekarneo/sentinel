"""Default execution handler registration."""

from __future__ import annotations

from typing import Any

from backend.app.execution.models import ExecutionType
from backend.app.execution.registry import ExecutionRegistry
from backend.app.services.enrollment_service import EnrollmentService
from backend.app.services.gallery_service import GalleryService
from backend.app.services.recognition_service import RecognitionService


def _not_implemented(task_type: ExecutionType) -> dict[str, Any]:
    return {
        "status": "scaffolded",
        "message": f"{task_type.value} handler is registered but execution is disabled.",
    }


def register_default_handlers(
    registry: ExecutionRegistry,
    *,
    recognition: RecognitionService | None = None,
    enrollment: EnrollmentService | None = None,
    gallery: GalleryService | None = None,
) -> None:
    """Register scaffold handlers that delegate to application services.

    Handlers are placeholders until background execution is enabled. They are
    registered now so future worker activation requires no redesign.
    """
    registry.register(
        ExecutionType.RECOGNITION,
        lambda payload: _handler_recognition(recognition, payload),
    )
    registry.register(
        ExecutionType.ENROLLMENT,
        lambda payload: _handler_enrollment(enrollment, payload),
    )
    registry.register(
        ExecutionType.GALLERY_REBUILD,
        lambda payload: _handler_gallery_rebuild(gallery, payload),
    )
    registry.register(
        ExecutionType.BENCHMARK,
        lambda _payload: _not_implemented(ExecutionType.BENCHMARK),
    )
    registry.register(
        ExecutionType.VIDEO_PROCESSING,
        lambda _payload: _not_implemented(ExecutionType.VIDEO_PROCESSING),
    )
    registry.register(
        ExecutionType.DATASET_EVALUATION,
        lambda _payload: _not_implemented(ExecutionType.DATASET_EVALUATION),
    )
    registry.register(
        ExecutionType.FRAUD_DETECTION,
        lambda _payload: _not_implemented(ExecutionType.FRAUD_DETECTION),
    )


def _handler_recognition(
    recognition: RecognitionService | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if recognition is None:
        return _not_implemented(ExecutionType.RECOGNITION)
    return {
        "status": "scaffolded",
        "service": "RecognitionService",
        "payload_keys": sorted(payload.keys()),
    }


def _handler_enrollment(
    enrollment: EnrollmentService | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if enrollment is None:
        return _not_implemented(ExecutionType.ENROLLMENT)
    return {
        "status": "scaffolded",
        "service": "EnrollmentService",
        "payload_keys": sorted(payload.keys()),
    }


def _handler_gallery_rebuild(
    gallery: GalleryService | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if gallery is None:
        return _not_implemented(ExecutionType.GALLERY_REBUILD)
    return {
        "status": "scaffolded",
        "service": "GalleryService",
        "payload_keys": sorted(payload.keys()),
    }
