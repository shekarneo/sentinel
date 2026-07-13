"""Enrollment API routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.dependencies import (
    get_enrollment_service,
    get_observability_service,
    get_recognition_service,
)
from backend.api.exceptions import ValidationAPIError
from backend.api.observability import record_pipeline_execution
from backend.api.schemas.enrollment import EnrollmentResponse
from backend.app.pipeline.profile import PipelineProfile
from backend.app.services.enrollment_service import EnrollmentService
from backend.app.services.observability_service import ObservabilityService
from backend.app.services.recognition_service import RecognitionService

router = APIRouter()


@router.post("", response_model=EnrollmentResponse)
async def enroll_identity(
    identity_name: str = Form(..., description="Display name for the identity."),
    employee_id: str = Form(..., description="Unique employee identifier."),
    image: UploadFile = File(..., description="Enrollment image file."),
    recognition_service: RecognitionService = Depends(get_recognition_service),
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
    observability: ObservabilityService = Depends(get_observability_service),
) -> EnrollmentResponse:
    """Enroll a gallery identity from an uploaded image."""
    identity_id = employee_id.strip()
    if not identity_id:
        raise ValidationAPIError("employee_id is required.")

    image_bytes = await image.read()
    try:
        decoded_image = recognition_service.decode_image(image_bytes)
    except ValueError as exc:
        raise ValidationAPIError(str(exc)) from exc

    started_at = datetime.now(timezone.utc)
    context = recognition_service.recognize(decoded_image, PipelineProfile.ENROLLMENT)
    record_pipeline_execution(
        observability,
        context,
        source="api.enrollment",
        started_at=started_at,
    )
    if not context.faces:
        raise ValidationAPIError("No faces detected in enrollment image.")

    face = context.faces[0]
    if face.embedding is None or face.embedding.vector is None:
        raise ValidationAPIError("Enrollment image did not produce an embedding.")

    metadata = {
        "name": identity_name.strip(),
        "employee_id": employee_id.strip(),
    }
    embedding_id = enrollment_service.enroll_and_persist(
        identity_id,
        face.embedding.vector,
        metadata=metadata,
    )
    return EnrollmentResponse(
        identity_id=identity_id,
        embedding_id=embedding_id,
        message="Identity enrolled successfully.",
    )


@router.delete("/{identity_id}")
def delete_identity(
    identity_id: str,
    enrollment_service: EnrollmentService = Depends(get_enrollment_service),
) -> dict[str, str]:
    """Delete an enrolled gallery identity."""
    enrollment_service.delete_and_persist(identity_id)
    return {"message": f"Identity {identity_id!r} deleted."}
