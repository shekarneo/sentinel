"""Gallery API routes."""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_gallery_service
from backend.api.exceptions import ValidationAPIError
from backend.api.schemas.gallery import (
    GalleryDetailResponse,
    GalleryEntryResponse,
    GalleryIdentityResponse,
    GalleryListResponse,
    GalleryRebuildResponse,
)
from backend.app.services.gallery_service import GalleryService

router = APIRouter()


@router.get("", response_model=GalleryListResponse)
def list_gallery(
    gallery_service: GalleryService = Depends(get_gallery_service),
) -> GalleryListResponse:
    """List enrolled gallery identities."""
    entries = gallery_service.list_entries()
    identities = [entry["identity_id"] for entry in entries]
    return GalleryListResponse(
        count=len(identities),
        identities=identities,
        entries=[
            GalleryEntryResponse(
                identity_id=entry["identity_id"],
                metadata=entry["metadata"],
                enrollment_count=int(entry["enrollment_count"]),
            )
            for entry in entries
        ],
    )


@router.get("/{identity_id}", response_model=GalleryDetailResponse)
def get_gallery_identity(
    identity_id: str,
    gallery_service: GalleryService = Depends(get_gallery_service),
) -> GalleryDetailResponse:
    """Return metadata for a single gallery identity."""
    metadata = gallery_service.get_identity_metadata(identity_id)
    if metadata is None and identity_id not in gallery_service.list_identities():
        raise ValidationAPIError(f"Identity {identity_id!r} is not enrolled.")
    return GalleryDetailResponse(
        identity=GalleryIdentityResponse(
            identity_id=identity_id,
            metadata=metadata,
        )
    )


@router.post("/rebuild", response_model=GalleryRebuildResponse)
def rebuild_gallery(
    gallery_service: GalleryService = Depends(get_gallery_service),
) -> GalleryRebuildResponse:
    """Rebuild the vector search index from enrolled gallery vectors."""
    count = gallery_service.rebuild()
    return GalleryRebuildResponse(
        message="Gallery index rebuilt.",
        count=count,
    )
