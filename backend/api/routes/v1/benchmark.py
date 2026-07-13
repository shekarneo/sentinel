"""Benchmark API routes."""

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.dependencies import get_recognition_service
from backend.api.exceptions import ValidationAPIError
from backend.api.schemas.benchmark import BenchmarkResponse, BenchmarkStageTimings
from backend.api.schemas.mappers import parse_profile
from backend.app.services.recognition_service import RecognitionService

router = APIRouter()

_STAGE_ALIASES = {
    "scrfd": "detection_time_ms",
    "detection": "detection_time_ms",
    "alignment": "alignment_time_ms",
    "assessment": "assessment_time_ms",
    "embedding": "embedding_time_ms",
    "search": "search_time_ms",
    "verification": "verification_time_ms",
}


@router.post("", response_model=BenchmarkResponse)
async def run_benchmark(
    profile: str = Form(default="search"),
    iterations: int = Form(default=1, ge=1, le=10),
    image: UploadFile = File(..., description="Benchmark probe image."),
    recognition_service: RecognitionService = Depends(get_recognition_service),
) -> BenchmarkResponse:
    """Run a pipeline benchmark and return per-stage timings."""
    try:
        pipeline_profile = parse_profile(profile)
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc

    image_bytes = await image.read()
    try:
        decoded_image = recognition_service.decode_image(image_bytes)
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc

    totals = {
        "detection_time_ms": 0.0,
        "alignment_time_ms": 0.0,
        "assessment_time_ms": 0.0,
        "embedding_time_ms": 0.0,
        "search_time_ms": 0.0,
        "verification_time_ms": 0.0,
        "total_pipeline_time_ms": 0.0,
    }

    for _ in range(iterations):
        context = recognition_service.recognize(decoded_image, pipeline_profile)
        for stage_name, seconds in context.timings.items():
            key = _STAGE_ALIASES.get(stage_name.lower())
            if key is not None:
                totals[key] += seconds * 1000.0
        totals["total_pipeline_time_ms"] += sum(context.timings.values()) * 1000.0

    if iterations > 1:
        for key in totals:
            totals[key] /= iterations

    return BenchmarkResponse(
        profile=pipeline_profile.value,
        iterations=iterations,
        timings=BenchmarkStageTimings(**totals),
        message="Benchmark completed.",
    )
