# 07_DATASET_PROCESSING

## Purpose

This document defines the Dataset Processing Engine architecture for batch
biometric workloads. The engine reuses the existing Execution Engine and
application services. No additional queue, threading, or multiprocessing is
introduced in the scaffolded release.

## Architecture

```
Dataset Source
      ↓
DatasetLoader
      ↓
DatasetManifest
      ↓
DatasetProcessor
      ↓
ExecutionManager / JobService
      ↓
RecognitionService / EnrollmentService / GalleryService
      ↓
Pipeline Orchestrator
      ↓
Observability
      ↓
DatasetResult / DatasetExporter
```

Loaders translate source-specific formats into a source-agnostic
``DatasetManifest``. The processor consumes manifests only and never knows
whether data originated from a folder, CSV, ZIP, or COCO export.

Each ``DatasetItem`` becomes one ``ExecutionTask``. Workers remain disabled
until a future release enables background execution.

## Domain Models

### DatasetManifest

Canonical loaded dataset representation.

| Field | Purpose |
| --- | --- |
| `dataset_id` | Unique manifest identifier |
| `dataset_type` | Source type enum |
| `root_path` | Dataset root path |
| `items` | `list[DatasetItem]` |
| `metadata` | Loader-level metadata |
| `statistics` | Aggregate item statistics |

### DatasetItem

One execution unit inside a manifest.

| Field | Purpose |
| --- | --- |
| `item_id` | Unique item identifier |
| `source_path` | Item source location |
| `metadata` | Item metadata |
| `ground_truth` | Optional labels for evaluation |
| `attributes` | Optional structured attributes |

### DatasetResult

Aggregate output populated by processing and future evaluators.

| Field | Purpose |
| --- | --- |
| `processed` | Successfully processed items |
| `failed` | Failed items |
| `duration` | Total duration in milliseconds |
| `exports` | Generated export paths |
| `metrics` | Evaluator-specific metrics |

## Folder Structure

```
backend/app/dataset/
├── __init__.py
├── manager.py          # DatasetManager
├── processor.py        # DatasetProcessor (manifest-only)
├── dataset.py          # Manifest builders
├── loaders.py          # DatasetLoader implementations
├── exporters.py        # DatasetExporter interfaces
├── evaluators.py       # DatasetEvaluator interfaces
├── models.py           # DatasetManifest, DatasetItem, DatasetResult
├── config.py           # DatasetProcessingConfig
├── registry.py         # Loader/exporter/evaluator registries
└── utils.py            # ID and time helpers
```

## Dataset Types

| Type | Status |
| --- | --- |
| `IMAGE_FOLDER` | Scaffold loader |
| `VIDEO_FOLDER` | Scaffold via generic builder |
| `ZIP_ARCHIVE` | Scaffold loader |
| `CSV_MANIFEST` | Scaffold loader |
| `COCO_DATASET` | Scaffold loader |
| `GALLERY_EXPORT` | Scaffold via generic builder |
| `RTSP_CAPTURE` | Reserved |
| `LIVE_RECORDING` | Reserved |

## Dataset Operations

| Operation | Execution Type | Status |
| --- | --- | --- |
| `RECOGNITION` | `recognition` | Scaffold |
| `ENROLLMENT` | `enrollment` | Scaffold |
| `GALLERY_IMPORT` | `enrollment` | Scaffold |
| `GALLERY_EXPORT` | `gallery_rebuild` | Scaffold |
| `BENCHMARK` | `benchmark` | Scaffold |
| `MODEL_EVALUATION` | `dataset_evaluation` | Scaffold |
| `DATASET_VALIDATION` | `dataset_evaluation` | Scaffold |
| `FRAUD_EVALUATION` | — | Reserved |
| `OCR_EVALUATION` | — | Reserved |
| `CALIBRATION` | — | Reserved |

## DatasetManager API

| Method | Purpose |
| --- | --- |
| `load()` | Load manifest via registered loader |
| `process()` | Load manifest and queue one execution per item |
| `export()` | Generate scaffold export metadata |
| `summary()` | Return job counters |
| `get_job()` | Fetch job by ID |
| `get_manifest()` | Fetch loaded manifest |
| `list_jobs()` | List all jobs |
| `get_results()` | Return scaffold result + execution linkage |

## Execution Engine Integration

- Uses existing ``JobService`` and ``ExecutionManager``
- No new queue introduced
- Each ``DatasetItem`` maps to one ``ExecutionTask`` with `source="api.datasets"`
- Observability service is wired for future per-item execution records

## Loaders

All loaders return ``DatasetManifest``.

| Loader | Dataset Type |
| --- | --- |
| `ImageFolderLoader` | `IMAGE_FOLDER` |
| `ZipLoader` | `ZIP_ARCHIVE` |
| `CSVLoader` | `CSV_MANIFEST` |
| `COCOLoader` | `COCO_DATASET` |

## Exporters

| Exporter | Status |
| --- | --- |
| `ScaffoldSummaryExporter` | Enabled (metadata only) |
| `CSVExporter` | Reserved |
| `JSONExporter` | Reserved |
| `ExcelExporter` | Reserved |
| `PDFExporter` | Reserved |

## Evaluators

Evaluators populate ``DatasetResult``.

| Evaluator | Status |
| --- | --- |
| `RecognitionEvaluator` | Reserved |
| `VerificationEvaluator` | Reserved |
| `BenchmarkEvaluator` | Reserved |
| `FraudEvaluator` | Reserved |
| `CalibrationEvaluator` | Reserved |

## REST API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/v1/datasets/process` | Queue dataset job |
| GET | `/api/v1/datasets/jobs` | List jobs |
| GET | `/api/v1/datasets/jobs/{id}` | Job detail |
| GET | `/api/v1/datasets/results/{id}` | Scaffold results + `DatasetResult` |

## Web Console

Route: `/datasets`

Displays dataset type, operation, progress, processed/failed counts, execution
time, export status, and results summary.

## Future Extension Points

| Extension | Approach |
| --- | --- |
| Benchmarking | `BenchmarkEvaluator` populates `DatasetResult.metrics` |
| Fraud evaluation | `FraudEvaluator` + `FRAUD_EVALUATION` operation |
| Calibration | `CalibrationEvaluator` + `CALIBRATION` operation |
| Real loaders | Implement `DatasetLoader.load()` → `DatasetManifest` |
| Export formats | Implement `DatasetExporter.export()` backends |

## Production Readiness

| Area | Status |
| --- | --- |
| Domain models (`DatasetManifest`, `DatasetItem`, `DatasetResult`) | Ready to freeze |
| Manifest-only processor boundary | Ready to freeze |
| Manager / execution integration | Ready to freeze |
| Loader / exporter / evaluator interfaces | Ready to freeze |
| REST API scaffold | Ready to freeze |
| Web console scaffold | Ready to freeze |
| Real processing | Not enabled |
| Distributed execution | Future |

The architecture is ready to freeze.
