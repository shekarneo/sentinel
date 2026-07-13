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
4.  Biometric Fraud Detection
5.  Embedding Service
6.  Decision Engine

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
