# 02_DEVELOPMENT_GUIDE

## Purpose

Execution roadmap for building the Biometric Intelligence Platform.

## Table of Contents

1.  Project Overview
2.  Repository Structure
3.  Technology Stack
4.  Development Principles
5.  Development Order
6.  Module Development Template
7.  Testing Strategy
8.  Future Enhancements

## Project Overview

Execution roadmap for building the Biometric Intelligence Platform. This guide defines repository layout, technology choices, development order, and module conventions for a single-developer workflow.

## Repository Structure

docs/ - 01_SYSTEM_ARCHITECTURE.md - 02_DEVELOPMENT_GUIDE.md - ARCHITECTURE_DECISIONS.md - assets/

## Technology Stack

### Current Technology Stack

**Backend**

-   FastAPI

**Deep Learning**

-   PyTorch

**Inference**

-   ONNX Runtime

**Detection**

-   SCRFD

**Recognition**

-   ArcFace

**Vector Search**

-   FAISS

**OCR**

-   PaddleOCR

**Database**

-   PostgreSQL

**Cache**

-   Redis

**Frontend**

-   Gradio

### Future Technologies

-   DeepStream
-   Triton
-   Kubernetes
-   Kafka

## Development Principles

Develop modules in the order defined below. Complete each area before moving to dependent work. Every module follows the Module Development Template.

## Development Order

### 1. Project Foundation

-   Repository setup
-   Docker
-   Poetry
-   FastAPI
-   Logging
-   Configuration
-   Dependency Injection
-   Environment Management

### 2. Infrastructure

-   Model Loader
-   ONNX Runtime
-   TensorRT
-   PostgreSQL
-   Redis
-   FAISS
-   File Storage

### 3. Core Biometric Engine

Implement modules in this order. Each stage enriches the same ``Face``
object by populating its own nested stage model. Modules do **not**
replace ``Face`` with separate types such as ``AlignedFace`` or
``AssessmentResult``.

```
Image
  ↓
SCRFD Detection
  ↓
Face
├── Detection Data
├── AlignmentData
├── AssessmentData
├── FraudData
└── EmbeddingData
  ↓
Search / Matching
  ↓
Verification / Decision
```

**Implementation roadmap**

1.  Face Detection — **complete** (SCRFD, frozen)
2.  Face Alignment — **complete** (frozen)
3.  Face Assessment Engine — **complete** (frozen)
4.  Pipeline Orchestrator — **complete**
5.  Embedding Service — **complete** (frozen)
6.  Search Engine — **complete** (frozen)
7.  Biometric Fraud Detection
8.  Decision Engine

**Search engine architecture**

**Recognition**

```
Face (with EmbeddingData)
  ↓
FaceSearcher
  ↓
SearchIndex.search()
  ↓
RawSearchOutput
  ↓
IdentityRepository
  ↓
SearchResults
  ↓
PipelineContext.metadata["search_results"]
```

**Enrollment**

```
IdentityService
  ↓
IdentityRepository
  ↓
SearchIndex
```

| Component | Role |
| --- | --- |
| ``SearchIndex`` | Vector-only provider: ``load()``, ``add()``, ``remove()``, ``update()``, ``search()``, ``save()``, ``list_embedding_ids()``, ``rebuild()`` |
| ``FaceSearcher`` | Probe validation, raw search, identity resolution, ``SearchResults`` assembly |
| ``IdentityRepository`` | Identity mapping persistence and lookup; sole owner of identity metadata |
| ``JsonIdentityRepository`` | JSON mapping file implementation |
| ``IdentityService`` | Gallery lifecycle: ``initialize()``, enroll, update, delete, ``rebuild_gallery()``, load, save |
| ``SearchResult`` | Identity resolved after raw search; ``SearchIndex`` never knows identities |
| ``SearchResults`` | Per-face search output with timing and provider |
| ``FaissSearchIndex`` | FAISS ``IndexFlatIP`` provider |

**Metadata ownership**

``SearchIndex`` must never store, return, or modify identity metadata.
``IdentityRepository`` owns identity data and embedding-to-identity
mappings. ``FaceSearcher`` resolves metadata only after
``SearchIndex.search()`` returns ``RawSearchOutput``.

**Dependency injection**

Construct a shared repository and index with
``create_search_engine_components()``, then inject both into
``IdentityService(repository=..., search_index=...)`` and
``FaceSearcher(repository=..., search_index=...)``.

**Gallery lifecycle**

| Method | Owner | Purpose |
| --- | --- | --- |
| ``IdentityService.initialize()`` | ``IdentityService`` | Load assets and validate mapping/index consistency |
| ``SearchIndex.rebuild()`` | ``SearchIndex`` | Rebuild vector index from stored vectors |
| ``IdentityService.rebuild_gallery()`` | ``IdentityService`` | Call ``SearchIndex.rebuild()`` only; never provider-specific APIs |

``initialize()`` is explicit and not wired to application startup.
Inconsistencies during validation raise ``ValueError`` without silent repair.

Configuration is defined in ``configs/models.yaml`` under ``search``.
FAISS is the first provider. Future providers (Milvus, Qdrant, pgvector)
must implement ``SearchIndex`` without changing ``FaceSearcher``.

Gallery enrollment belongs exclusively to ``IdentityService``.
``FaceSearcher`` performs recognition only. ``SearchIndex`` stores vectors
only and returns raw ``embedding_id`` matches.

**Embedding engine architecture**

```
Face (with AlignmentData)
  ↓
FaceEmbedder
  ↓
EmbeddingModel
  ↓
EmbeddingData
  ↓
Face.embedding
```

| Component | Role |
| --- | --- |
| ``EmbeddingModel`` | Provider abstraction with ``load()``, ``warmup()``, ``embed()``, ``embed_batch()`` |
| ``FaceEmbedder`` | Validates aligned faces, warm-up, and populates ``Face.embedding`` |
| ``EmbeddingData`` | Domain model for embedding vectors only |
| ``ArcFaceEmbeddingModel`` | ArcFace provider with thread-safe loading |

Configuration is defined in ``configs/models.yaml`` under ``embedding``.
ArcFace is the first provider. Future providers (AdaFace, MagFace,
ElasticFace) must implement ``EmbeddingModel`` without changing
``FaceEmbedder``.

``EmbeddingData`` must not store similarity, identity, search results, or
gallery metadata. Embedding enriches ``Face.embedding`` only.

**Embedding invariant**

Every embedding produced by the Embedding Engine is unit-normalized
(L2-normalized). Similarity search, FAISS indexing, verification, and
clustering operate directly on ``Face.embedding.vector`` without additional
normalization. Providers are responsible for L2 normalization before
returning ``EmbeddingData``; ``normalized=True`` is guaranteed for
successful embeddings.

**Pipeline orchestrator**

AI modules never invoke one another directly. The pipeline layer in
``backend/app/pipeline/`` owns execution order and use-case selection.

| Component | Role |
| --- | --- |
| ``PipelineContext`` | Frozen API: image, faces, profile, metadata, timings, errors |
| ``PipelineProfile`` | Selects a built-in use case or ``CUSTOM`` stage list |
| ``PipelineStage`` | Adapter interface exposing static ``StageMetadata`` and lifecycle hooks |
| ``StageMetadata`` | Self-describing requires/provides contract for each stage |
| ``PipelineRegistry`` | Stores stage classes or factories; exposes metadata without instantiation |
| ``PipelineBuilder`` | Loads ``configs/pipeline_profiles.yaml`` and resolves stages |
| ``PipelineExecutor`` | Runs stages sequentially with timing and graceful failure |

Pipeline profiles are configuration-driven. The builder contains no
profile-specific business logic. Runtime-specific outputs belong in
``context.metadata`` (for example ``metadata['results']``). New AI modules
should register a stage factory without changing ``PipelineBuilder``.

Stages are self-describing through ``StageMetadata``. The registry exposes
metadata without instantiating stages. The builder may use this metadata
for future validation and dependency checking.

**Stage lifecycle**

Stages expose optional ``initialize()`` and ``shutdown()`` hooks. Defaults
are no-ops. ``EmbeddingStage.initialize()`` calls ``FaceEmbedder.warmup()``
to load the ONNX provider before execution. Lifecycle hooks are idempotent.

**Embedding warm-up and batching**

Call ``FaceEmbedder.warmup()`` during application or stage initialization to
avoid cold-start latency on the first embedding request. ``EmbeddingModel``
also exposes ``embed_batch()`` for future batched inference; the default
raises ``NotImplementedError`` and ArcFace does not implement batching yet.

Built-in profiles (from ``configs/pipeline_profiles.yaml``):

| Profile | Stages |
| --- | --- |
| ``ENROLLMENT`` | scrfd, alignment, assessment, fraud, embedding |
| ``ATTENDANCE`` | scrfd, alignment, fraud, embedding, search |
| ``SURVEILLANCE`` | scrfd, alignment, embedding, search |
| ``KYC`` | scrfd, alignment, assessment, fraud, embedding, verification |
| ``ACCESS_CONTROL`` | scrfd, alignment, fraud, embedding, verification |
| ``SEARCH`` | scrfd, alignment, embedding, search |
| ``CUSTOM`` | User-defined stage list |

**Face Assessment architecture**

```
Face (with AlignmentData)
  ↓
Face Assessment
  ↓
AssessmentData
  ↓
Embedding
```

| Analyzer | Result model | Fields |
| --- | --- | --- |
| Blur | ``BlurResult`` | ``variance``, ``score`` |
| Brightness | ``BrightnessResult`` | ``mean_brightness``, ``score`` |
| Pose | ``PoseResult`` | ``yaw``, ``pitch``, ``roll``, ``score`` (heuristic quality indicators, not calibrated pose) |
| Size | ``SizeResult`` | ``width``, ``height``, ``score`` |
| Visibility | ``VisibilityResult`` | ``visible_ratio``, ``score`` |
| Scoring | — | ``overall_score``, ``is_acceptable`` |

``AssessmentData`` nests analyzer results under ``blur``, ``brightness``,
``pose``, ``size``, and ``visibility``. Each analyzer owns one result model.
``FaceAssessor`` combines them before attaching to ``Face.assessment``.

Assessment enriches ``Face.assessment`` only. It never modifies detection,
alignment, or embedding data.

**Face Assessment Calibration**

Assessment thresholds and scoring weights live in
``configs/thresholds.yaml``. They are intentionally separated from
implementation code.

Recommended calibration workflow:

1.  Collect aligned faces from representative deployment data.
2.  Label face quality.
3.  Run assessment metrics on the labeled set.
4.  Analyze per-analyzer score distributions.
5.  Update ``configs/thresholds.yaml``.

No code changes should be required after calibration.

**Module summary**

| Module | Input | Output | Nested Model |
| --- | --- | --- | --- |
| Face Detection | Image | `list[Face]` | Detection fields only |
| Face Alignment | `list[Face]` + image | `list[Face]` | `Face.alignment` |
| Face Assessment | `list[Face]` (aligned) | `list[Face]` | `Face.assessment` |
| Biometric Fraud Detection | `list[Face]` | `list[Face]` | `Face.fraud` |
| Embedding Service | `list[Face]` | `list[Face]` | `Face.embedding` |
| Decision Engine | `list[Face]` + platform results | Decision | — |

**Face architecture rules**

1.  Detection fields (``bounding_box``, ``confidence``, ``landmarks``)
    are immutable after SCRFD.
2.  Each biometric module owns exactly one nested data model.
3.  Modules populate only their own nested section on ``Face``. Face
    Assessment must not modify detection, alignment, or embedding fields.
4.  ``SearchResult``, ``VerificationResult``, ``Identity``, ``Track``,
    and ``Gallery`` remain separate domain models and are not stored
    inside ``Face``.

### 4. Identity Platform

-   Enrollment
-   Gallery
-   FAISS Search
-   Identity Matching
-   Active Search

### 5. Surveillance Platform

-   Tracking
-   Investigation
-   Timeline
-   Alerts
-   Real-Time Processing

### 6. Identity Verification

-   Document Processing
-   OCR
-   Face ↔ Identity Matching
-   Verification

### 7. Frontend

-   Gradio
-   Investigation Dashboard
-   Enrollment UI
-   Verification UI
-   Camera Dashboard

### 8. Deployment (Optional)

-   Docker Compose
-   Monitoring
-   Metrics

### 9. Future Enhancements

See the Future Enhancements section below.

## Module Development Template

Every module should follow:

1.  Overview
2.  Architecture
3.  Folder Structure
4.  Interfaces
5.  Data Models
6.  Implementation
7.  Unit Tests
8.  Integration Tests
9.  Documentation
10. Demo

## Testing Strategy

-   Module Testing
-   Integration Testing
-   System Testing
-   Performance Testing
-   Benchmarking

## Future Enhancements

-   NVIDIA DeepStream Integration
-   Triton Inference Server
-   Kubernetes Deployment
-   Distributed Processing
-   Multi-GPU Scaling
