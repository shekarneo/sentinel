# Sentinel

Open-Source Biometric Intelligence Platform

## Description

Sentinel is a production-oriented biometric intelligence platform for surveillance and identity verification. It combines SCRFD detection, ArcFace embeddings, FAISS gallery search, and threshold-based verification behind a config-driven pipeline orchestrator.

## Architecture

```
Frontend (any client)
        ↓
REST API (`backend/api/`)
        ↓
Application Services (`backend/app/services/`)
        ↓
Pipeline Orchestrator
        ↓
AI Modules
```

The backend is frontend-agnostic. Any frontend (HTMX, React, Flutter,
Electron, CLI, REST client) must integrate exclusively through the REST
API.

```
Image
  ↓
SCRFD Detection
  ↓
Face Alignment
  ↓
Face Assessment (optional)
  ↓
ArcFace Embedding
  ↓
FAISS Search (1:N)
  ↓
Threshold Verification (1:1 decision)
```

Detailed design: [`docs/architecture/01_SYSTEM_ARCHITECTURE.md`](docs/architecture/01_SYSTEM_ARCHITECTURE.md)

Development guide: [`docs/architecture/02_DEVELOPMENT_GUIDE.md`](docs/architecture/02_DEVELOPMENT_GUIDE.md)

API architecture: [`docs/architecture/03_API_ARCHITECTURE.md`](docs/architecture/03_API_ARCHITECTURE.md)

Observability: [`docs/architecture/04_OBSERVABILITY.md`](docs/architecture/04_OBSERVABILITY.md)

Execution engine: [`docs/architecture/05_EXECUTION_ENGINE.md`](docs/architecture/05_EXECUTION_ENGINE.md)

Live camera: [`docs/architecture/06_LIVE_CAMERA.md`](docs/architecture/06_LIVE_CAMERA.md)

Dataset processing: [`docs/architecture/07_DATASET_PROCESSING.md`](docs/architecture/07_DATASET_PROCESSING.md)

## Repository Layout

```
backend/
  api/                 # REST API layer (FastAPI, versioned routes, DTOs)
  ai/                  # Frozen biometric engines
  app/
    config/            # Configuration loader (Pydantic + YAML)
    domain/            # Face, SearchResults, VerificationResult, etc.
    pipeline/          # Pipeline orchestrator and stage adapters
    repositories/      # Identity gallery mappings
    services/          # Application services (recognition, gallery, search, verification)
frontend/
  templates/           # Jinja2 layouts and pages
  static/              # CSS and Alpine.js console scripts
  web.py               # Web console page routes
configs/               # settings, models, thresholds, pipeline profiles, logging
scripts/               # Module tests, integration tests, visualizations
models/                # ONNX model weights (not committed)
indexes/               # FAISS gallery index + JSON mapping (generated)
```

## Requirements

- Python 3.10+
- See [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

| File | Purpose |
|------|---------|
| `configs/settings.yaml` | App name, version, environment, paths |
| `configs/models.yaml` | SCRFD, ArcFace, FAISS model paths |
| `configs/thresholds.yaml` | Detection, assessment, verification thresholds |
| `configs/pipeline_profiles.yaml` | Use-case stage sequences |
| `configs/logging.yaml` | Logging format (reserved for future wiring) |

## Quick Start

```bash
# Application entry point
python -m backend.app.main

# REST API server (includes Web Console)
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Automated test suite
pytest

# Full pipeline integration test
python scripts/test_pipeline.py image.jpg

# End-to-end recognition + verification
python scripts/test_verification.py image.jpg --reset-gallery
python scripts/visualize_verification.py image.jpg
```

See [`scripts/README.md`](scripts/README.md) for the full script catalog.

## Module Status

| Module | Status |
|--------|--------|
| SCRFD Detection | Frozen |
| Face Alignment | Frozen |
| Face Assessment | Frozen |
| Pipeline Orchestrator | Frozen |
| ArcFace Embedding | Frozen |
| Search Engine (FAISS) | Frozen |
| Verification Engine | Frozen |
| REST API Layer | Scaffolded |
| Fraud Detection | Planned |
| Decision Engine | Planned |

## License

See repository license file.
