# 04_OBSERVABILITY

## Purpose

This document defines the platform-wide observability layer for pipeline
execution tracking, metrics, and operational visibility.

## Architecture

```
REST API routes
      ↓
record_pipeline_execution()   ← API boundary hook only
      ↓
ObservabilityService
      ↓
build_execution_record()
      ↓
ExecutionStore (InMemoryExecutionStore)
      ↓
GET /api/v1/executions*
      ↓
Web Console · Execution History
```

Observability is **additive**. It does not modify AI modules, pipeline
stage logic, or recognition behavior. Recording occurs after a
``PipelineContext`` is returned from ``RecognitionService``.

## Execution Lifecycle

1. **PipelineStarted** — API receives a pipeline request.
2. **StageStarted** — Each executed stage begins (reconstructed timeline).
3. **StageCompleted** / **StageFailed** — Stage finishes with metrics.
4. **PipelineCompleted** — Full execution record persisted.

## Domain Models

| Model | Purpose |
| --- | --- |
| ``ExecutionRecord`` | One complete pipeline execution |
| ``ExecutionStage`` | Per-stage timing, status, warnings, errors |
| ``ExecutionStatus`` | ``running``, ``success``, ``warning``, ``failed``, ``skipped`` |
| ``PipelineEvent`` | Lifecycle event with timestamp and metadata |

## Execution Timeline

Each ``ExecutionRecord`` captures:

- ``started_at`` / ``ended_at`` / ``duration_ms`` (total)
- Per-stage ``started_at``, ``ended_at``, ``duration_ms``
- Ordered stage list derived from ``PipelineContext.timings``

## Stage Metrics

Each ``ExecutionStage`` records:

| Field | Description |
| --- | --- |
| ``stage`` | Stage identifier (e.g. ``scrfd``, ``embedding``) |
| ``status`` | Success, warning, failed, or skipped |
| ``duration_ms`` | Stage latency |
| ``warnings`` | Quality or policy warnings |
| ``errors`` | Stage error messages |
| ``metadata`` | Non-sensitive stage summary |

## Execution Store

``ExecutionStore`` abstract interface:

| Method | Purpose |
| --- | --- |
| ``add()`` | Persist a completed record |
| ``get()`` | Fetch by execution ID |
| ``latest()`` | Return recent executions |
| ``list_all()`` | Return all stored executions |
| ``clear()`` | Reset store (testing / admin) |

Initial implementation: ``InMemoryExecutionStore``.

A future ``PostgresExecutionStore`` or ``RedisExecutionStore`` can
implement the same interface without redesigning services or API routes.

## REST API

| Method | Path | Description |
| --- | --- | --- |
| GET | ``/api/v1/executions`` | List all executions |
| GET | ``/api/v1/executions/latest`` | Recent executions (`limit` query param) |
| GET | ``/api/v1/executions/{id}`` | Single execution detail |

## Web Console

Route: ``/executions``

Features:

- **Recent Runs** — latest executions from ``/api/v1/executions/latest``
- **Pipeline Timeline** — per-stage duration bars
- **Failure Details** — errors and failed stage breakdown
- **Stage Metrics** — tabular latency / warning / error counts
- **Pipeline Events** — event log per execution

## Future Integrations

| System | Integration approach |
| --- | --- |
| **OpenTelemetry** | Export ``PipelineEvent`` spans from ``ObservabilityService`` |
| **Prometheus** | Counter/histogram metrics per stage latency and status |
| **Grafana** | Dashboards over Prometheus metrics and execution store |
| **Persistent store** | Replace ``InMemoryExecutionStore`` with database backend |

Recommended future flow:

```
PipelineContext
      ↓
ObservabilityService
      ├─ ExecutionStore (durable)
      ├─ OpenTelemetry tracer
      └─ Prometheus metrics registry
```

## Production Readiness

| Area | Status |
| --- | --- |
| Domain models | Implemented |
| In-memory store | Implemented |
| REST API | Implemented |
| Web console | Implemented |
| API recording hooks | Recognition, search, verification, enrollment |
| Durable persistence | Future |
| Metrics export | Future |
| Distributed tracing | Future |

The layer is production-ready for **single-process observability and
debugging**. For multi-instance deployments, add a shared execution store
and metrics export before relying on execution history for SLO monitoring.
