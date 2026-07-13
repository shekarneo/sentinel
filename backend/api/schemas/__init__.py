"""API schema package."""

from backend.api.schemas.benchmark import BenchmarkRequest, BenchmarkResponse
from backend.api.schemas.common import ErrorResponse, HealthResponse, MessageResponse
from backend.api.schemas.configuration import (
    AppConfigResponse,
    ConfigurationResponse,
    PipelineProfileSummary,
)
from backend.api.schemas.enrollment import EnrollmentRequest, EnrollmentResponse
from backend.api.schemas.gallery import (
    GalleryDetailResponse,
    GalleryIdentityResponse,
    GalleryListResponse,
    GalleryRebuildResponse,
)
from backend.api.schemas.mappers import (
    map_face,
    map_recognition_context,
    map_search_results,
    map_verification_result,
    parse_profile,
)
from backend.api.schemas.recognition import (
    FaceResponse,
    RecognitionRequest,
    RecognitionResponse,
)
from backend.api.schemas.search import SearchCandidateResponse, SearchRequest, SearchResponse
from backend.api.schemas.system import SystemHealthResponse, SystemStatusResponse
from backend.api.schemas.verification import VerificationRequest, VerificationResponse

__all__ = [
    "AppConfigResponse",
    "BenchmarkRequest",
    "BenchmarkResponse",
    "ConfigurationResponse",
    "EnrollmentRequest",
    "EnrollmentResponse",
    "ErrorResponse",
    "FaceResponse",
    "GalleryDetailResponse",
    "GalleryIdentityResponse",
    "GalleryListResponse",
    "GalleryRebuildResponse",
    "HealthResponse",
    "MessageResponse",
    "PipelineProfileSummary",
    "RecognitionRequest",
    "RecognitionResponse",
    "SearchCandidateResponse",
    "SearchRequest",
    "SearchResponse",
    "SystemHealthResponse",
    "SystemStatusResponse",
    "VerificationRequest",
    "VerificationResponse",
    "map_face",
    "map_recognition_context",
    "map_search_results",
    "map_verification_result",
    "parse_profile",
]
