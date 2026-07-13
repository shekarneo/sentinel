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
    modify another module's nested data.

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

**Consumes**

`list[Face]` where ``Face.alignment`` is populated

**Produces**

The same ``Face`` objects with ``Face.assessment`` populated
(``AssessmentData``).

**Responsibilities**

-   Image quality assessment
-   Blur estimation
-   Pose estimation
-   Face visibility
-   Brightness / exposure
-   Face attribute assessment (future)
-   Occlusion detection (future)

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

**Consumes**

`list[Face]` where ``Face.alignment`` is populated

**Produces**

The same ``Face`` objects with ``Face.embedding`` populated
(``EmbeddingData``).

### Decision Engine

**Purpose**

Combine assessment, fraud, embedding, and search results into a final
biometric decision for enrollment, verification, or surveillance
workflows.
