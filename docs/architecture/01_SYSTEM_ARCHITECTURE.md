# 01_SYSTEM_ARCHITECTURE

## Purpose

This document defines the overall architecture of the Biometric
Intelligence Platform.

## Table of Contents

1.  Introduction
    -   Vision
    -   Goals
    -   Scope
    -   Design Principles
2.  Platform Overview
    -   Surveillance Intelligence
    -   Identity Verification
    -   Employee Access Control
    -   Visitor Management
    -   Future Applications
3.  Overall System Architecture
    -   High-Level Architecture
    -   Layered Architecture
    -   End-to-End Data Flow
    -   Architectural Decisions
4.  Core Data Models
    -   Frame
    -   Face
        -   Detection Data
        -   AlignmentData
        -   AssessmentData
        -   FraudData
        -   EmbeddingData
    -   Track
    -   Document
    -   Identity Object
    -   Biometric Profile
    -   Verification Context
5.  Core Biometric Engine
    -   Biometric Pipeline
    -   Face Detection
    -   Face Alignment
    -   Face Assessment Engine
        -   Image Quality Assessment
        -   Blur Estimation
        -   Pose Estimation
        -   Face Visibility Assessment
        -   Brightness / Exposure
        -   Face Attribute Assessment (Future)
        -   Occlusion Detection (Future)
    -   Biometric Fraud Detection
        -   Presentation Attack Detection
        -   Deepfake Detection
        -   Replay Attack Detection
        -   Image Tampering Detection
    -   Embedding Service
    -   Decision Engine
6.  Identity Services
    -   Enrollment & Gallery Management
    -   Identity Search Engine (FAISS)
    -   Identity Matching Engine
    -   Gallery Architecture
    -   Active Search Sessions
7.  Identity Document Processing Engine
    -   Document Detection
    -   Perspective Correction
    -   Image Enhancement
    -   OCR Engine
    -   Field Extraction
    -   Document Validation
8.  Real-Time Surveillance Platform
    -   Real-Time Processing Pipeline
    -   Future Production Deployment (NVIDIA DeepStream)
    -   Camera Manager
    -   Real-Time Processing Engine
    -   Cross-Camera Tracking
    -   Investigation Engine
    -   Timeline Reconstruction
    -   Alert Engine
9.  Identity Verification Platform
    -   Verification Flow
    -   Face ↔ Identity Matching
    -   Live Camera Verification
    -   Verification Policies
    -   Verification Decision
10. Backend Architecture
    -   FastAPI
    -   Platform Services
    -   Core AI Services
    -   Infrastructure Layer
11. Data & Storage Architecture
    -   PostgreSQL
    -   FAISS
    -   Redis
    -   File Storage
    -   Model Repository
12. Deployment Architecture
    -   Local Development
    -   Production Deployment
    -   Docker
    -   Triton Inference Server (Future)
    -   Kubernetes (Future)
13. Testing & Validation
    -   Module Testing
    -   Integration Testing
    -   System Testing
    -   Performance Testing
    -   Benchmarking
14. Future Roadmap
    -   Occlusion Detection
    -   Face Attribute Analysis
    -   Gait Recognition
    -   Voice Biometrics
    -   Distributed Search
    -   Multi-GPU Scaling

## Notes

This document contains: - Architecture diagrams - Component diagrams -
Sequence diagrams - High-level workflows - Design decisions - Technology
choices

It intentionally excludes implementation details, source code, folder
structure, and API implementations.

The architecture has been designed to support NVIDIA DeepStream for
production-scale deployments. The current implementation uses a simpler
backend pipeline while keeping the architecture compatible with future
DeepStream integration.

## 4. Core Data Models

### Face

``Face`` is the canonical object that flows through the entire biometric
engine. Subsequent pipeline modules do **not** replace ``Face`` with new
domain types such as ``AlignedFace``, ``AssessmentResult``, or
``EmbeddingResult``.

Each module populates its own nested stage model on the same ``Face``
instance:

```
Face
├── Detection Data      (bounding_box, confidence, landmarks)
├── AlignmentData       (populated by Face Alignment)
├── AssessmentData      (populated by Face Assessment)
├── FraudData           (populated by Biometric Fraud Detection)
└── EmbeddingData       (populated by Embedding Service)
```

### Face Pipeline Flow

```
Image
  ↓
SCRFD Detection
  ↓
Face
  ↓
Face Alignment        → Face.alignment
  ↓
Face Assessment       → Face.assessment
  ↓
Biometric Fraud       → Face.fraud
  ↓
Embedding             → Face.embedding
  ↓
Search / Matching
  ↓
Verification / Decision
```

### Architecture Rules

1.  **Detection fields are immutable.** After SCRFD, the following
    fields must never be modified:

    -   ``bounding_box``
    -   ``confidence``
    -   ``landmarks``

2.  **Each biometric module owns exactly one nested data model.**

    | Module | Nested Model |
    | --- | --- |
    | Face Alignment | ``AlignmentData`` |
    | Face Assessment | ``AssessmentData`` |
    | Biometric Fraud Detection | ``FraudData`` |
    | Embedding Service | ``EmbeddingData`` |

3.  **Modules populate only their own section.** A module must not
    modify another module's nested data. Face Assessment enriches
    ``Face.assessment`` only and must not modify detection, alignment,
    or embedding fields.

4.  **Platform models remain separate.** ``SearchResult``,
    ``VerificationResult``, ``Identity``, ``Track``, and ``Gallery`` are
    not stored inside ``Face``.

## 5. Core Biometric Engine

The Core Biometric Engine processes an input image through a fixed,
sequential pipeline. Each stage consumes the output of the previous
stage. This order is frozen and serves as the single source of truth for
all future implementation.

### Biometric Pipeline

```
Image
  ↓
Face Detection
  ↓
Face Alignment
  ↓
Face Assessment
  ↓
Biometric Fraud Detection
  ↓
Embedding
  ↓
Search / Matching
  ↓
Decision Engine
```

### Pipeline Orchestrator

**Purpose**

Select and execute biometric pipelines by use case. AI modules never
invoke one another directly. Execution order and stage selection are
owned by the pipeline layer in ``backend/app/pipeline/``.

**Core components**

| Component | Responsibility |
| --- | --- |
| ``PipelineContext`` | Frozen execution state: image, faces, profile, metadata, timings, errors |
| ``PipelineProfile`` | Built-in use-case profile (enrollment, attendance, surveillance, and others) |
| ``PipelineStage`` | Stage interface exposing static ``StageMetadata`` |
| ``StageMetadata`` | Self-describing stage contract: requires, provides, capabilities |
| ``PipelineRegistry`` | Register stage classes or factories; query metadata without instantiation |
| ``PipelineBuilder`` | Load profile configuration and resolve stage names only |
| ``PipelineExecutor`` | Run stages sequentially with timing and graceful failure |

``Face`` objects remain the primary domain objects. ``PipelineContext``
references ``context.faces`` but does not replace the ``Face`` model.

**Configuration**

Pipeline profile stage sequences are defined in
``configs/pipeline_profiles.yaml`` and loaded through
``PipelineProfileSettings``. The builder contains no profile-specific
business logic.

**Frozen context API**

``PipelineContext`` exposes only these top-level fields: ``image``,
``faces``, ``profile``, ``metadata``, ``timings``, and ``errors``.
Runtime-specific information must be stored inside ``metadata`` rather
than adding new top-level fields.

**Registry and dependency injection**

The registry stores stage classes or zero-argument factory callables.
Stages are instantiated only when ``PipelineBuilder`` constructs a
pipeline. New AI modules register themselves without modifying the
builder. Registered stage metadata can be queried through
``PipelineRegistry.get_metadata()`` without creating stage instances.

**Stage metadata**

Pipeline stages are self-describing through ``StageMetadata``:

| Field | Purpose |
| --- | --- |
| ``name`` | Unique stage identifier |
| ``version`` | Adapter version |
| ``description`` | Human-readable responsibility |
| ``requires`` | Required capabilities or context inputs |
| ``provides`` | Produced capabilities or context outputs |
| ``gpu_required`` | Whether GPU acceleration is expected |
| ``supports_warmup`` | Whether the stage exposes a warm-up lifecycle hook |
| ``supports_batching`` | Whether batched execution is supported |

**Stage lifecycle**

Pipeline stages may override optional lifecycle hooks:

| Method | Purpose |
| --- | --- |
| ``initialize()`` | Prepare resources before execution (for example model warm-up) |
| ``execute()`` | Run the stage against the pipeline context |
| ``shutdown()`` | Release resources after execution |

Default lifecycle methods are no-ops. ``EmbeddingStage.initialize()`` calls
``FaceEmbedder.warmup()`` to load the ONNX provider before the first request.
Warm-up is idempotent and does not repeat expensive initialization.

Metadata is static. It must not include latency, thresholds, configuration,
or runtime state. ``PipelineBuilder`` uses metadata to validate stage
dependencies during pipeline construction.

**Execution rule**

Stages are thin adapters around AI modules. A stage receives
``PipelineContext``, invokes exactly one module, updates ``context.faces``
or ``context.metadata``, and returns the same context object.

**Built-in profiles**

Profile definitions live in ``configs/pipeline_profiles.yaml``:

| Profile | Stages | Search depth |
| --- | --- | --- |
| ``ENROLLMENT`` | SCRFD → Alignment → Assessment → Fraud → Embedding | Disabled |
| ``ATTENDANCE`` | SCRFD → Alignment → Fraud → Embedding → Search | Top-5 |
| ``SURVEILLANCE`` | SCRFD → Alignment → Embedding → Search | Top-1 |
| ``KYC`` | SCRFD → Alignment → Assessment → Fraud → Embedding → Search → Verification | Top-5 |
| ``ACCESS_CONTROL`` | SCRFD → Alignment → Fraud → Embedding → Search → Verification | Top-3 |
| ``SEARCH`` | SCRFD → Alignment → Embedding → Search | Top-1 |
| ``CUSTOM`` | Caller-defined ordered stage list | Top-1 default |

Search depth is a business-level policy configured per pipeline profile in
``configs/pipeline_profiles.yaml``. ``SearchIndex`` is unaware of business
intent. ``SearchStage`` reads the resolved profile search policy and passes
``top_k`` to ``FaceSearcher.search()``, which forwards it to
``SearchIndex.search()`` without choosing depth itself.

| Use case | ``top_k`` | Rationale |
| --- | --- | --- |
| Surveillance | 1 | Low latency, continuous tracking |
| Attendance | 5 | Verification over multiple enrollment templates |
| KYC | 5 | Identity verification with candidate evaluation |
| Access Control | 3 | Fast verification with limited candidate fallback |
| Search / Investigation | 1 today (20 future) | Analyst workflows may request deeper candidate lists |

Fraud and Verification policy stages may remain placeholders until their
modules are fully implemented. Search and Embedding stages are implemented.

**Stage dependency chain**

The pipeline builder validates ``StageMetadata.requires`` against
capabilities provided by earlier stages during construction. Invalid
profiles fail before execution.

```
Alignment
  ↓
Embedding
  ↓
Search
  ↓
Verification
```

| Dependency | Required capability | Provided by |
| --- | --- | --- |
| Embedding | ``alignment`` | Alignment stage |
| Search | ``embedding`` | Embedding stage |
| Verification | ``search`` | Search stage |

Verification must never appear before Search in a profile. Search must
never appear before Embedding. Embedding must never appear before
Alignment.

### Face Detection

**Purpose**

-   Detect faces in an input image
-   Detect five facial landmarks per face
-   Return detected faces as domain objects

**Output**

`list[Face]` with detection fields populated. Nested stage models
(``alignment``, ``assessment``, ``fraud``, ``embedding``) are ``None`` at
this stage.

Each ``Face`` is created with immutable detection data: bounding box,
confidence, and five landmarks. SCRFD does not populate alignment,
assessment, fraud, embedding, identity, or tracking data.

**Status:** Implemented and frozen (SCRFD).

### Face Alignment

**Purpose**

-   Normalize detected faces using landmark geometry
-   Compute a similarity transform from detected landmarks
-   Warp each face to a canonical face size
-   Produce aligned face crops for downstream processing

**Consumes**

`list[Face]` and the source image

**Produces**

The same ``Face`` objects with ``Face.alignment`` populated
(``AlignmentData``).

**Shared module**

Face Alignment is shared by both Face Assessment and Embedding. All
quality, fraud, and recognition stages operate on aligned faces rather
than raw detections.

### Face Assessment Engine

**Purpose**

Evaluate whether an aligned face is suitable for biometric processing.

Face Assessment operates on **aligned faces**, not raw detections.
Alignment is performed upstream by the Face Alignment module.

**Pipeline**

```
Face (with AlignmentData)
  ↓
Face Assessment
  ↓
AssessmentData
  ↓
Embedding
```

**Consumes**

`list[Face]` where ``Face.alignment`` is populated

**Produces**

The same ``Face`` objects with ``Face.assessment`` populated
(``AssessmentData``).

Face Assessment enriches ``Face`` only. It must not modify detection,
alignment, or embedding data.

**Status:** Architecture frozen. Implementation complete.

**Configuration**

All assessment thresholds and scoring weights are loaded from
``configs/thresholds.yaml``. Implementation modules must not define
production threshold defaults.

| Group | Configuration keys |
| --- | --- |
| Blur | ``assessment.blur.warning``, ``acceptable``, ``excellent`` |
| Brightness | ``assessment.brightness.too_dark``, ``acceptable_low``, ``acceptable_high``, ``too_bright`` |
| Pose | ``assessment.pose.max_yaw``, ``max_pitch``, ``max_roll`` |
| Size | ``assessment.size.min_face_width``, ``min_face_height`` |
| Visibility | ``assessment.visibility.minimum_visible_ratio`` |
| Overall | ``assessment.overall.*_weight``, ``minimum_acceptable_score`` |

**Analyzers**

Each analyzer returns its own result model. ``FaceAssessor`` combines them
into ``AssessmentData``.

| Analyzer | Result model | Fields |
| --- | --- | --- |
| Blur | ``BlurResult`` | ``variance``, ``score`` |
| Brightness | ``BrightnessResult`` | ``mean_brightness``, ``score`` |
| Pose | ``PoseResult`` | ``yaw``, ``pitch``, ``roll``, ``score`` (heuristic quality indicators, not calibrated pose) |
| Size | ``SizeResult`` | ``width``, ``height``, ``score`` |
| Visibility | ``VisibilityResult`` | ``visible_ratio``, ``score`` |
| Scoring | — | ``overall_score``, ``is_acceptable`` |

**AssessmentData structure**

``AssessmentData`` contains nested stage results:

| Field | Type |
| --- | --- |
| ``blur`` | ``BlurResult \| None`` |
| ``brightness`` | ``BrightnessResult \| None`` |
| ``pose`` | ``PoseResult \| None`` |
| ``size`` | ``SizeResult \| None`` |
| ``visibility`` | ``VisibilityResult \| None`` |
| ``overall_score`` | ``float \| None`` |
| ``is_acceptable`` | ``bool \| None`` |

**Responsibilities**

-   Image quality assessment
-   Blur estimation
-   Pose estimation
-   Face visibility
-   Brightness / exposure
-   Face attribute assessment (future)
-   Occlusion detection (future)

**Face Assessment Calibration**

Assessment thresholds are configuration-driven. Threshold values are
intentionally separated from implementation code so deployments can be
tuned without modifying Python modules.

Recommended calibration workflow:

1.  Collect aligned faces from the target deployment environment.
2.  Label face quality (acceptable / unacceptable) with domain experts.
3.  Run assessment metrics on the labeled set.
4.  Analyze score distributions per analyzer.
5.  Update ``configs/thresholds.yaml`` with calibrated values.

No code changes should be required after calibration. Restart the
application or clear the in-process threshold cache in tests if
configuration is reloaded at runtime.

### Biometric Fraud Detection

**Purpose**

Detect presentation attacks and synthetic or manipulated biometric
samples.

Biometric Fraud Detection operates on **aligned faces** when applicable.
Alignment ensures consistent spatial normalization before liveness and
anti-spoofing analysis.

**Consumes**

`list[Face]` where ``Face.alignment`` is populated

**Produces**

The same ``Face`` objects with ``Face.fraud`` populated (``FraudData``).

**Responsibilities**

-   Presentation attack detection
-   Deepfake detection
-   Replay attack detection
-   Image tampering detection

### Embedding Service

**Purpose**

Extract biometric feature vectors from aligned face crops.

Embeddings are always extracted from **aligned faces** produced by the
Face Alignment module. Raw detection crops are not used for recognition.

**Architecture**

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

**Module layout**

| Component | Responsibility |
| --- | --- |
| ``EmbeddingModel`` | Provider abstraction: ``load()``, ``warmup()``, ``embed()``, ``embed_batch()`` |
| ``FaceEmbedder`` | Validates aligned faces, warm-up, and populates ``Face.embedding`` |
| ``EmbeddingData`` | Domain model for vector output only |
| ``ArcFaceEmbeddingModel`` | ArcFace provider with thread-safe model loading |

**Warm-up**

``FaceEmbedder.warmup()`` ensures the configured provider is loaded before
the first embedding request. ``EmbeddingModel.warmup()`` defaults to
``load()``; providers may override it for additional one-time setup.
Warm-up is idempotent.

**Future batching**

``EmbeddingModel.embed_batch()`` is reserved for future batched inference.
The default implementation raises ``NotImplementedError``. ArcFace currently
embeds faces sequentially through ``embed()``.

**Configuration**

Embedding provider settings live in ``configs/models.yaml`` under
``embedding``:

| Key | Purpose |
| --- | --- |
| ``provider`` | Active provider (``arcface`` first) |
| ``model`` | Relative ONNX model path |
| ``input_size`` | Canonical aligned input size |

Future providers (AdaFace, MagFace, ElasticFace) must implement
``EmbeddingModel`` without changing ``FaceEmbedder``.

**Consumes**

`list[Face]` where ``Face.alignment`` is populated

**Produces**

The same ``Face`` objects with ``Face.embedding`` populated
(``EmbeddingData``).

``EmbeddingData`` stores only embedding output:

| Field | Purpose |
| --- | --- |
| ``vector`` | Unit-length L2-normalized feature vector |
| ``dimension`` | Vector length |
| ``normalized`` | Guaranteed ``True`` for successful embeddings; providers normalize before return |
| ``model_name`` | Provider model identifier |
| ``inference_time_ms`` | Optional inference latency |
| ``confidence`` | Optional provider confidence |

Similarity, identity, search results, and gallery metadata remain outside
``EmbeddingData``.

**Embedding invariant**

Every embedding produced by the Embedding Engine is unit-normalized
(L2-normalized). Similarity search, FAISS indexing, verification, and
clustering operate directly on ``Face.embedding.vector`` without additional
normalization. Providers (ArcFace, AdaFace, MagFace, and future models)
are responsible for normalization before returning ``EmbeddingData``.

**Status:** Architecture frozen. ArcFace provider complete.

### Search Engine

**Purpose**

Query a gallery of enrolled identity embeddings and return ranked match
candidates for each probe face. Search operates on unit-normalized vectors
produced by the Embedding Engine; probe vectors must not be re-normalized.

**Architecture**

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

Search depth is configured per pipeline profile in
``configs/pipeline_profiles.yaml`` and injected into
``PipelineContext.metadata['profile_search']`` by ``PipelineExecutor``.
``FaceSearcher`` forwards the requested ``top_k`` to ``SearchIndex`` without
choosing business search depth itself.

**Enrollment**

```
IdentityService
  ↓
IdentityRepository
  ↓
SearchIndex
```

**Module layout**

| Component | Responsibility |
| --- | --- |
| ``SearchIndex`` | Vector-only provider: ``load()``, ``add()``, ``remove()``, ``update()``, ``search()``, ``save()``, ``list_embedding_ids()``, ``rebuild()`` |
| ``FaceSearcher`` | Probe validation, raw search, identity resolution, ``SearchResults`` assembly |
| ``IdentityRepository`` | Identity mapping persistence and lookup; sole owner of identity metadata |
| ``JsonIdentityRepository`` | JSON mapping file implementation |
| ``IdentityService`` | Gallery lifecycle owner: ``initialize()``, enroll, update, delete, ``rebuild_gallery()``, load, save |
| ``SearchResult`` | Identity resolved by ``IdentityRepository`` after raw search |
| ``SearchResults`` | Per-face search output with timing and provider metadata |
| ``FaissSearchIndex`` | FAISS ``IndexFlatIP`` provider |

**Responsibility ownership**

| Concern | Owner |
| --- | --- |
| Vector storage and nearest-neighbor search | ``SearchIndex`` |
| Identity mapping persistence | ``IdentityRepository`` |
| Gallery enroll / update / delete / rebuild | ``IdentityService`` |
| Probe search and ``SearchResult`` assembly | ``FaceSearcher`` |
| Identity metadata | ``IdentityRepository`` (not ``SearchIndex``) |

**Metadata ownership rule**

``SearchIndex`` must never store, return, or modify identity metadata. It
manages embedding identifiers, vectors, and raw distances only.
``IdentityRepository`` is the sole owner of identity metadata and
embedding-to-identity mappings. ``FaceSearcher`` resolves metadata only
after receiving ``RawSearchOutput`` from ``SearchIndex``.

**Dependency injection**

``IdentityService`` and ``FaceSearcher`` accept the same ``repository`` and
``search_index`` instances through constructor injection. Use
``create_search_engine_components()`` to construct a shared pair without
introducing a DI framework.

**Gallery lifecycle**

| Method | Owner | Purpose |
| --- | --- | --- |
| ``IdentityService.initialize()`` | ``IdentityService`` | Load repository and index, validate gallery consistency |
| ``SearchIndex.rebuild()`` | ``SearchIndex`` | Rebuild vector index from stored vectors (provider-specific) |
| ``IdentityService.rebuild_gallery()`` | ``IdentityService`` | Orchestrate gallery rebuild via ``SearchIndex.rebuild()`` only |

``initialize()`` is an explicit lifecycle hook. It is not invoked
automatically at startup. Callers must invoke it when production gallery
readiness should be verified. During ``initialize()``, missing mapping or
index files and mapping/index synchronization mismatches raise
``ValueError``; the service does not silently repair inconsistencies.

**Configuration**

Search provider settings live in ``configs/models.yaml`` under ``search``:

| Key | Purpose |
| --- | --- |
| ``provider`` | Active provider (``faiss`` first) |
| ``index_path`` | Relative path to the gallery index file |
| ``metric`` | Similarity metric (``cosine`` for unit-normalized vectors) |

Future providers (Milvus, Qdrant, pgvector) must implement ``SearchIndex``
without changing ``FaceSearcher``.

**Consumes**

`list[Face]` where ``Face.embedding`` is populated with a unit-normalized vector

**Produces**

``list[SearchResults]`` stored in ``PipelineContext.metadata["search_results"]``.
Search results are not attached to ``Face``.

| Field | Purpose |
| --- | --- |
| ``SearchResult.identity_id`` | Identity resolved by ``IdentityRepository`` after raw search |
| ``SearchResult.score`` | Similarity score for the match |
| ``SearchResult.rank`` | Rank within the result set (1-based) |
| ``SearchResult.metadata`` | Optional gallery metadata |
| ``SearchResults.search_time_ms`` | Query latency |
| ``SearchResults.provider`` | Active search provider name |

**Status:** Architecture frozen. FAISS provider complete.

### Verification Engine

**Purpose**

Evaluate search candidates for probe faces and produce accept or reject
verification decisions. Verification operates on ``SearchResults`` produced
by the Search Engine; it must not re-run gallery search or embedding
extraction.

**Architecture**

```
Face (with EmbeddingData)
  ↓
SearchResults
  ↓
FaceVerifier
  ↓
VerificationEngine.verify()
  ↓
VerificationResult
  ↓
PipelineContext.metadata["verification"]
```

**Module layout**

| Component | Responsibility |
| --- | --- |
| ``VerificationEngine`` | Verification policy abstraction: ``verify()`` |
| ``ThresholdVerificationEngine`` | Fixed-threshold verification provider |
| ``FaceVerifier`` | Input validation, engine orchestration, timing |
| ``VerificationResult`` | Per-face verification decision and scores |
| ``VerificationDecision`` | ``UNKNOWN``, ``ACCEPT``, ``REJECT`` |

**Responsibility ownership**

| Concern | Owner |
| --- | --- |
| Gallery search and candidate ranking | ``FaceSearcher`` / ``SearchIndex`` |
| Verification policy and decision logic | ``VerificationEngine`` |
| Probe validation and result assembly | ``FaceVerifier`` |
| Verification output storage | ``PipelineContext.metadata["verification"]`` |

``FaceVerifier`` must not depend on FAISS, ArcFace, or other provider
implementations. It delegates policy to ``VerificationEngine`` only.

**Configuration**

Configuration files live in ``configs/``:

| File | Purpose | Consumed by |
| --- | --- | --- |
| ``settings.yaml`` | App metadata, paths | ``Configuration.load()`` |
| ``models.yaml`` | Model paths and providers | ``Configuration.load_models()`` |
| ``thresholds.yaml`` | Detection, assessment, verification thresholds | ``Configuration.load_thresholds()`` |
| ``pipeline_profiles.yaml`` | Pipeline stage sequences | ``Configuration.load_pipeline_profiles()`` |
| ``logging.yaml`` | Logging format/level | ``configure_logging()`` at startup |

Keys reserved for future runtime use: ``app.debug``, ``runtime.device``,
``paths.data_dir``.

Verification thresholds live in ``configs/thresholds.yaml`` under
``verification``:

| Key | Purpose |
| --- | --- |
| ``similarity_threshold`` | Minimum similarity score required for acceptance |

**Consumes**

One ``Face`` with ``EmbeddingData`` and one ``SearchResults`` object per
probe.

**Produces**

``list[VerificationResult]`` stored in
``PipelineContext.metadata["verification"]``. Verification results are not
attached to ``Face``.

| Field | Purpose |
| --- | --- |
| ``VerificationResult.decision`` | ``UNKNOWN``, ``ACCEPT``, or ``REJECT`` |
| ``VerificationResult.matched_identity_id`` | Accepted gallery identity, if any |
| ``VerificationResult.similarity_score`` | Score used for the decision |
| ``VerificationResult.threshold`` | Threshold applied by the engine |
| ``VerificationResult.is_verified`` | Convenience accept/reject flag |
| ``VerificationResult.verification_time_ms`` | Verification latency |
| ``VerificationResult.metadata`` | Optional downstream metadata |
| ``VerificationResult.metadata.policy`` | Active verification policy (``threshold``) |
| ``VerificationResult.metadata.engine_version`` | Verification engine version (``1.0.0``) |

**Threshold policy**

``ThresholdVerificationEngine`` orders search candidates by rank and
evaluates the best-ranked candidate against
``verification.similarity_threshold`` from ``configs/thresholds.yaml``:

| Condition | Decision |
| --- | --- |
| No candidates | ``UNKNOWN`` |
| Best similarity >= threshold | ``ACCEPT`` |
| Best similarity < threshold | ``REJECT`` |

The threshold policy evaluates only the best candidate today. Candidate
ordering is internal so future policies (adaptive, multi-template,
watchlist) can evaluate the same ordered list without interface changes.

**Verification invariants**

``VerificationResult`` enforces a strict relationship between
``decision`` and ``is_verified``:

| ``decision`` | ``is_verified`` |
| --- | --- |
| ``ACCEPT`` | ``True`` |
| ``REJECT`` | ``False`` |
| ``UNKNOWN`` | ``False`` |

These values must never diverge. Engines derive ``is_verified`` through
``resolve_is_verified()``, and ``VerificationResult`` rejects invalid
combinations at construction time.

**Future engines**

Future providers must implement ``VerificationEngine`` without changing
``FaceVerifier``:

- ``ThresholdVerificationEngine`` — fixed similarity threshold (**complete**)
- ``AdaptiveVerificationEngine`` — dynamic threshold calibration
- ``MultiTemplateVerificationEngine`` — multi-template identity matching
- ``WatchlistVerificationEngine`` — watchlist-specific acceptance rules

**Status:** Architecture frozen. Threshold policy complete.

Verification depends on Search. Search depends on Embedding. Embedding
depends on Alignment. The pipeline builder enforces this dependency chain
at construction time using ``StageMetadata.requires`` and
``StageMetadata.provides``.

### Decision Engine

**Purpose**

Combine assessment, fraud, embedding, and search results into a final
biometric decision for enrollment, verification, or surveillance
workflows.
