# 03_API_ARCHITECTURE

## Purpose

This document defines the REST API layer that sits between external
clients and the biometric platform. The backend is **frontend-agnostic**.
Any frontend (HTMX, React, Flutter, Electron, CLI, REST client) must
integrate exclusively through the REST API.

## Layered Architecture

```
Frontend (any client)
        ↓
REST API (`backend/api/`)
        ↓
Application Services (`backend/app/services/`)
        ↓
Pipeline Orchestrator (`backend/app/pipeline/`)
        ↓
AI Modules (`backend/ai/`)
```

### Access Rules

| Layer | May call | Must not call directly from API |
| --- | --- | --- |
| REST API | Application services | Pipeline, AI modules, repositories |
| Application services | Pipeline, AI orchestrators, repositories | — |
| Pipeline | AI modules through stage adapters | — |

The API must never import or invoke:

- `backend/app/pipeline/` (except indirectly through `RecognitionService`)
- `backend/ai/` modules
- `FaceSearcher`, `FaceVerifier`, or `IdentityService` from route handlers

Route handlers depend on application services injected through
`backend/api/dependencies.py`.

## API Versioning

All endpoints are mounted under a versioned prefix:

| Version | Prefix | Status |
| --- | --- | --- |
| v1 | `/api/v1/` | Current |
| v2 | `/api/v2/` | Future (parallel routers, unchanged business logic) |

Version constants live in `backend/api/version.py`. New API versions add
new route packages (`routes/v2/`) and DTOs without modifying frozen AI
modules or pipeline orchestration.

## Folder Structure

```
backend/api/
├── __init__.py
├── main.py                 # FastAPI app factory
├── dependencies.py         # Shared service container (DI)
├── version.py              # API version constants
├── exceptions/
│   ├── errors.py
│   └── handlers.py
├── middleware/
│   └── request_context.py
├── routes/
│   └── v1/
│       ├── recognition.py
│       ├── enrollment.py
│       ├── gallery.py
│       ├── search.py
│       ├── verification.py
│       ├── system.py
│       ├── benchmark.py
│       └── configuration.py
└── schemas/
    ├── common.py
    ├── recognition.py
    ├── enrollment.py
    ├── gallery.py
    ├── search.py
    ├── verification.py
    ├── system.py
    ├── benchmark.py
    ├── configuration.py
    └── mappers.py          # Domain → DTO conversion
```

## Application Services

| Service | Delegates to | Responsibility |
| --- | --- | --- |
| `RecognitionService` | `PipelineBuilder`, `PipelineExecutor` | Run biometric pipelines |
| `EnrollmentService` | `IdentityService` | Gallery enrollment writes |
| `GalleryService` | `IdentityService` | Gallery reads and rebuild |
| `SearchService` | `FaceSearcher` | 1:N gallery search |
| `VerificationService` | `FaceVerifier` | 1:1 threshold verification |
| `ConfigurationService` | `Configuration`, profile settings | Read-only config exposure |

Shared gallery components (`IdentityRepository`, `SearchIndex`) are
created once in `get_app_services()` and reused by `IdentityService`,
`SearchService`, `EnrollmentService`, and `GalleryService`.

## Service Boundaries

```
Recognition
    → RecognitionService
    → Pipeline

Enrollment
    → EnrollmentService
    → IdentityService

Gallery
    → GalleryService
    → IdentityService

Search
    → SearchService
    → FaceSearcher

Verification
    → VerificationService
    → FaceVerifier
```

## DTO Mapping

Internal domain models are never returned directly from the API.

| Domain model | API DTO |
| --- | --- |
| `Face` | `FaceResponse` (via `RecognitionResponse`) |
| `SearchResults` | `SearchResponse` |
| `VerificationResult` | `VerificationResponse` |

Mappers live in `backend/api/schemas/mappers.py`.

## REST Contracts (v1)

| Resource | Method | Path | Status |
| --- | --- | --- | --- |
| Recognition | POST | `/api/v1/recognition` | Implemented |
| Enrollment | POST | `/api/v1/enrollment` | Scaffolded (501) |
| Enrollment | DELETE | `/api/v1/enrollment/{identity_id}` | Scaffolded (501) |
| Gallery | GET | `/api/v1/gallery` | Implemented |
| Gallery | GET | `/api/v1/gallery/{identity_id}` | Implemented |
| Gallery | POST | `/api/v1/gallery/rebuild` | Implemented |
| Search | POST | `/api/v1/search` | Implemented |
| Verification | POST | `/api/v1/verification` | Implemented |
| System | GET | `/api/v1/system/health` | Implemented |
| System | GET | `/api/v1/system/status` | Implemented |
| Benchmark | POST | `/api/v1/benchmark` | Scaffolded (501) |
| Configuration | GET | `/api/v1/configuration` | Implemented |

## Running the API

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

OpenAPI documentation is available at `/docs` when the server is running.

## Future Frontend Compatibility

| Client type | Integration approach |
| --- | --- |
| HTMX / server-rendered UI | HTTP forms to multipart endpoints |
| React / Vue / Angular SPA | REST + OpenAPI client generation |
| Flutter / mobile | REST over HTTPS |
| Electron desktop | Local or remote REST client |
| CLI tools | `curl`, HTTP libraries, or SDK |
| External systems | Versioned REST contracts only |

No client should import Python backend modules or depend on internal
domain types. All coupling is through stable JSON/multipart contracts
under `/api/v1/`.
