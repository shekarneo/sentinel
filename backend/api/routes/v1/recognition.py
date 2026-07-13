"""Recognition API routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.dependencies import get_observability_service, get_recognition_service
from backend.api.exceptions import ValidationAPIError
from backend.api.observability import record_pipeline_execution
from backend.api.schemas.mappers import map_recognition_context, parse_profile
from backend.api.schemas.recognition import RecognitionResponse
from backend.app.services.observability_service import ObservabilityService
from backend.app.services.recognition_service import RecognitionService

router = APIRouter()


@router.post("", response_model=RecognitionResponse)
async def recognize_faces(
    profile: str = Form(..., description="Pipeline profile name."),
    image: UploadFile = File(..., description="Probe image file."),
    recognition_service: RecognitionService = Depends(get_recognition_service),
    observability: ObservabilityService = Depends(get_observability_service),
) -> RecognitionResponse:
    """Run biometric recognition for an uploaded image."""
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
        source="api.recognition",
        started_at=started_at,
    )
    return map_recognition_context(context)
