"""Verification API routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.dependencies import (
    get_observability_service,
    get_recognition_service,
    get_verification_service,
)
from backend.api.exceptions import ValidationAPIError
from backend.api.observability import record_pipeline_execution
from backend.api.schemas.mappers import map_verification_result, parse_profile
from backend.api.schemas.verification import VerificationResponse
from backend.app.services.observability_service import ObservabilityService
from backend.app.services.recognition_service import RecognitionService
from backend.app.services.verification_service import VerificationService

router = APIRouter()


@router.post("", response_model=VerificationResponse)
async def verify_identity(
    profile: str = Form(default="kyc"),
    image: UploadFile = File(..., description="Probe image file."),
    recognition_service: RecognitionService = Depends(get_recognition_service),
    verification_service: VerificationService = Depends(get_verification_service),
    observability: ObservabilityService = Depends(get_observability_service),
) -> VerificationResponse:
    """Verify a probe image against gallery search candidates."""
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
        source="api.verification",
        started_at=started_at,
    )
    if not context.faces:
        raise ValidationAPIError("No faces detected in probe image.")

    verification_results = context.metadata.get("verification")
    if verification_results:
        return map_verification_result(verification_results[0])

    search_results = context.metadata.get("search_results")
    if search_results is None:
        raise ValidationAPIError(
            "Verification profile did not produce search results. "
            "Use a profile that includes search and verification stages."
        )

    results = verification_service.verify(context.faces, search_results)
    if not results:
        raise ValidationAPIError("Verification did not produce a result.")
    return map_verification_result(results[0])
