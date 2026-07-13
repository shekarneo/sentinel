"""Version 1 API routes."""

from fastapi import APIRouter

from backend.api.routes.v1.benchmark import router as benchmark_router
from backend.api.routes.v1.configuration import router as configuration_router
from backend.api.routes.v1.datasets import router as datasets_router
from backend.api.routes.v1.jobs import router as jobs_router
from backend.api.routes.v1.live import router as live_router
from backend.api.routes.v1.executions import router as executions_router
from backend.api.routes.v1.enrollment import router as enrollment_router
from backend.api.routes.v1.gallery import router as gallery_router
from backend.api.routes.v1.recognition import router as recognition_router
from backend.api.routes.v1.search import router as search_router
from backend.api.routes.v1.system import router as system_router
from backend.api.routes.v1.verification import router as verification_router

router = APIRouter()

router.include_router(recognition_router, prefix="/recognition", tags=["recognition"])
router.include_router(enrollment_router, prefix="/enrollment", tags=["enrollment"])
router.include_router(gallery_router, prefix="/gallery", tags=["gallery"])
router.include_router(search_router, prefix="/search", tags=["search"])
router.include_router(verification_router, prefix="/verification", tags=["verification"])
router.include_router(system_router, prefix="/system", tags=["system"])
router.include_router(benchmark_router, prefix="/benchmark", tags=["benchmark"])
router.include_router(configuration_router, prefix="/configuration", tags=["configuration"])
router.include_router(executions_router, prefix="/executions", tags=["executions"])
router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
router.include_router(live_router, prefix="/live", tags=["live"])
router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
