# 05_EXECUTION_ENGINE

## Purpose

This document defines the asynchronous execution engine architecture for
future background workloads. The engine is **scaffolded only** in the
current release. No background workers, threading, multiprocessing, or
asyncio execution loops are active.

## Architecture

```
REST API (/api/v1/jobs)
        ↓
JobService
        ↓
ExecutionManager
        ├─ ExecutionQueue (InMemoryExecutionQueue)
        ├─ ExecutionWorker (disabled)
        └─ ExecutionRegistry
                ↓
        Application Services
                ↓
        Pipeline Orchestrator
                ↓
        AI Modules
```

Workers and handlers must **never** access AI modules directly. Handlers
delegate exclusively to application services such as
``RecognitionService``, ``EnrollmentService``, and ``GalleryService``.

## Folder Structure

```
backend/app/execution/
├── __init__.py
├── manager.py          # ExecutionManager
├── queue.py            # ExecutionQueue, InMemoryExecutionQueue
├── worker.py           # ExecutionWorker (scaffold)
├── task.py             # Task contracts
├── models.py           # ExecutionTask, ExecutionType, ExecutionState
├── registry.py         # ExecutionRegistry
├── handlers.py         # Default handler registration
├── config.py           # ExecutionEngineConfig
└── utils.py            # Task helpers
```

## Execution Models

| Model | Purpose |
| --- | --- |
| ``ExecutionTask`` | One queued or completed background job |
| ``ExecutionType`` | Workload type identifier |
| ``ExecutionPriority`` | Scheduling priority |
| ``ExecutionState`` | ``queued``, ``running``, ``completed``, ``failed``, ``cancelled`` |

### Supported Execution Types

| Type | Handler target |
| --- | --- |
| ``RECOGNITION`` | ``RecognitionService`` |
| ``ENROLLMENT`` | ``EnrollmentService`` |
| ``GALLERY_REBUILD`` | ``GalleryService`` |
| ``BENCHMARK`` | Reserved |
| ``VIDEO_PROCESSING`` | Reserved |
| ``DATASET_EVALUATION`` | Reserved |
| ``FRAUD_DETECTION`` | Reserved |

New types can be added by extending ``ExecutionType`` and registering a
handler. ``ExecutionManager`` does not require changes.

## Execution Queue

``ExecutionQueue`` abstract API:

| Method | Purpose |
| --- | --- |
| ``submit()`` | Enqueue a task |
| ``next()`` | Dequeue highest-priority task |
| ``cancel()`` | Remove a queued task |
| ``clear()`` | Empty the queue |
| ``size()`` | Queue depth |

Initial implementation: ``InMemoryExecutionQueue``.

## Execution Worker

``ExecutionWorker`` responsibilities (future):

1. Poll the queue
2. Resolve handler from ``ExecutionRegistry``
3. Execute through application services
4. Update task state
5. Notify observability

Current status: ``worker_enabled = False``. Calling ``poll_once()`` raises
``RuntimeError``.

## Execution Manager

Public API:

| Method | Purpose |
| --- | --- |
| ``submit()`` | Queue a new task |
| ``cancel()`` | Cancel a queued task |
| ``status()`` | Fetch task by ID |
| ``list()`` | List all known tasks |

## REST API (Scaffold)

| Method | Path | Status |
| --- | --- | --- |
| POST | ``/api/v1/jobs`` | Queues task only |
| GET | ``/api/v1/jobs`` | Lists tasks + summary |
| GET | ``/api/v1/jobs/{id}`` | Task detail |
| DELETE | ``/api/v1/jobs/{id}`` | Cancel queued task |

## Web Console

Route: ``/jobs``

Displays scaffold counts for:

- Queued
- Running
- Completed
- Failed

## Future Extension Points

| Extension | Approach |
| --- | --- |
| Background workers | Enable ``ExecutionEngineConfig.worker_enabled`` and add process/thread pool |
| Redis / RabbitMQ queue | Implement ``ExecutionQueue`` backend |
| Celery integration | Map ``ExecutionTask`` to Celery task payloads |
| Batch processing | New ``ExecutionType`` + handler |
| Video processing | ``VIDEO_PROCESSING`` handler via application service |
| Observability linkage | Worker calls ``ObservabilityService.record_execution()`` |

## Production Readiness

| Area | Status |
| --- | --- |
| Domain models | Ready to freeze |
| Queue abstraction | Ready to freeze |
| Manager / registry | Ready to freeze |
| Worker scaffold | Ready to freeze (disabled) |
| REST API scaffold | Ready to freeze |
| Web console scaffold | Ready to freeze |
| Background execution | Not enabled |
| Distributed queue | Future |
| Retry / dead-letter | Future |

The architecture is ready for review and freeze. Enable workers only after
persistence, retry policy, and observability hooks are implemented.
