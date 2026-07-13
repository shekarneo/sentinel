"""API dependency injection container."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from backend.ai.search.searcher import create_search_engine_components
from backend.app.observability.store import InMemoryExecutionStore
from backend.app.services.dataset_service import DatasetService
from backend.app.services.configuration_service import ConfigurationService
from backend.app.services.enrollment_service import EnrollmentService
from backend.app.services.gallery_service import GalleryService
from backend.app.services.identity_service import IdentityService
from backend.app.services.job_service import JobService
from backend.app.services.live_camera_service import LiveCameraService
from backend.app.services.observability_service import ObservabilityService
from backend.app.services.recognition_service import RecognitionService
from backend.app.services.search_service import SearchService
from backend.app.services.verification_service import VerificationService


@dataclass(frozen=True)
class AppServices:
    """Shared application services instantiated once per process."""

    recognition: RecognitionService
    identity: IdentityService
    enrollment: EnrollmentService
    gallery: GalleryService
    search: SearchService
    verification: VerificationService
    configuration: ConfigurationService
    observability: ObservabilityService
    jobs: JobService
    live: LiveCameraService
    datasets: DatasetService


@lru_cache
def get_app_services() -> AppServices:
    """Create and cache shared application services."""
    repository, search_index = create_search_engine_components()
    identity = IdentityService(repository=repository, search_index=search_index)
    recognition = RecognitionService()
    enrollment = EnrollmentService(identity)
    gallery = GalleryService(identity)
    observability = ObservabilityService(store=InMemoryExecutionStore())
    jobs = JobService(
        recognition=recognition,
        enrollment=enrollment,
        gallery=gallery,
    )
    return AppServices(
        recognition=recognition,
        identity=identity,
        enrollment=enrollment,
        gallery=gallery,
        search=SearchService(repository=repository, search_index=search_index),
        verification=VerificationService(),
        configuration=ConfigurationService(),
        observability=observability,
        jobs=jobs,
        live=LiveCameraService(recognition=recognition),
        datasets=DatasetService(jobs=jobs, observability=observability),
    )


def get_recognition_service() -> RecognitionService:
    """FastAPI dependency for ``RecognitionService``."""
    return get_app_services().recognition


def get_enrollment_service() -> EnrollmentService:
    """FastAPI dependency for ``EnrollmentService``."""
    return get_app_services().enrollment


def get_gallery_service() -> GalleryService:
    """FastAPI dependency for ``GalleryService``."""
    return get_app_services().gallery


def get_search_service() -> SearchService:
    """FastAPI dependency for ``SearchService``."""
    return get_app_services().search


def get_verification_service() -> VerificationService:
    """FastAPI dependency for ``VerificationService``."""
    return get_app_services().verification


def get_configuration_service() -> ConfigurationService:
    """FastAPI dependency for ``ConfigurationService``."""
    return get_app_services().configuration


def get_observability_service() -> ObservabilityService:
    """FastAPI dependency for ``ObservabilityService``."""
    return get_app_services().observability


def get_job_service() -> JobService:
    """FastAPI dependency for ``JobService``."""
    return get_app_services().jobs


def get_live_camera_service() -> LiveCameraService:
    """FastAPI dependency for ``LiveCameraService``."""
    return get_app_services().live


def get_dataset_service() -> DatasetService:
    """FastAPI dependency for ``DatasetService``."""
    return get_app_services().datasets
