"""Search API routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.dependencies import (
    get_observability_service,
    get_recognition_service,
    get_search_service,
)
from backend.api.exceptions import ValidationAPIError
from backend.api.observability import record_pipeline_execution
from backend.api.schemas.mappers import map_search_results, parse_profile
from backend.api.schemas.search import SearchResponse
from backend.app.services.observability_service import ObservabilityService
from backend.app.services.recognition_service import RecognitionService
from backend.app.services.search_service import SearchService

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def search_gallery(
    top_k: int = Form(default=1, ge=1),
    profile: str = Form(default="search"),
    image: UploadFile = File(..., description="Probe image file."),
    recognition_service: RecognitionService = Depends(get_recognition_service),
    search_service: SearchService = Depends(get_search_service),
    observability: ObservabilityService = Depends(get_observability_service),
) -> SearchResponse:
    """Search the gallery for identities similar to a probe image."""
    try:
        pipeline_profile = parse_profile(profile)
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc

    image_bytes = await image.read()
    try:
        decoded_image = recognition_service.decode_image(image_bytes)
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc

    started_at = datetime.now(timezone.utc)
    context = recognition_service.recognize(decoded_image, pipeline_profile)
    record_pipeline_execution(
        observability,
        context,
        source="api.search",
        started_at=started_at,
    )
    if not context.faces:
        raise ValidationAPIError("No faces detected in probe image.")

    results = search_service.search(context.faces, top_k=top_k)
    if not results:
        raise ValidationAPIError("Search did not produce results.")
    return map_search_results(results[0])
