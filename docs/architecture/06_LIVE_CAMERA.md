# 06_LIVE_CAMERA

## Purpose

This document defines the Live Camera architecture for real-time biometric
recognition. The module reuses the existing recognition pipeline through
``RecognitionService`` and never accesses AI modules directly.

## Architecture

```
Browser Webcam
      ↓
Web Console (/live)
      ↓
REST API (/api/v1/live/*)  +  WebSocket (/api/v1/live/ws)
      ↓
LiveCameraService
      ↓
LiveCameraManager
      ↓
RecognitionService
      ↓
Pipeline Orchestrator
      ↓
AI Modules
      ↓
OverlayRenderer → Canvas/Web Console
```

The live module stops at ``RecognitionService``. Pipeline and AI layers remain
unchanged.

## Folder Structure

```
backend/app/live/
├── __init__.py
├── manager.py          # LiveCameraManager
├── stream.py           # LiveStream implementations
├── session.py          # LiveSessionState helpers
├── models.py           # LiveSession, overlays, metrics
├── config.py           # LiveCameraConfig, extension hooks
├── policy.py           # RecognitionPolicy
├── overlay.py          # OverlayRenderer implementations
├── recording.py        # RecordingHook extension point
├── processor.py        # Shared frame processing
├── ws_handler.py       # WebSocket protocol handler
└── utils.py            # Overlay builders, latency tracking
```

## Session Lifecycle

| Action | API | Manager Method |
| --- | --- | --- |
| Start | `POST /api/v1/live/start` | `start_session()` |
| Stop | `POST /api/v1/live/stop` | `stop_session()` |
| Pause | `POST /api/v1/live/pause` | `pause_session()` |
| Resume | `POST /api/v1/live/resume` | `resume_session()` |
| Status | `GET /api/v1/live/status` | `status()` |
| Sessions | `GET /api/v1/live/sessions` | `list_sessions()` |
| Frame (REST) | `POST /api/v1/live/frame` | `process_frame_bytes()` |
| Frame (WS) | `WS /api/v1/live/ws` | `process_live_frame()` |

## Real-Time Communication

### REST Frame Upload

`POST /api/v1/live/frame` remains supported for compatibility and testing.

### WebSocket Streaming

Endpoint: `WS /api/v1/live/ws`

Supported messages:

| Type | Direction | Purpose |
| --- | --- | --- |
| `ping` | Client → Server | Keepalive |
| `pong` | Server → Client | Keepalive response |
| `frame` | Client → Server | Submit one JPEG frame |
| `frame_result` | Server → Client | Overlay + metrics + render spec |
| `status` | Client → Server | Request session status |
| `error` | Server → Client | Validation or runtime error |

Binary frame format:

```
<json-header>\n<jpeg-bytes>
```

Both REST and WebSocket call the same ``process_live_frame()`` helper. No
pipeline logic is duplicated.

## Recognition Policy

Implicit ``frame_skip`` and ``recognition_interval`` have been replaced by an
explicit ``RecognitionPolicy``.

| Policy | Status | Behavior |
| --- | --- | --- |
| `every_frame` | Enabled | Recognize every submitted frame |
| `every_n_frames` | Enabled | Recognize every Nth captured frame |
| `adaptive` | Reserved | Future latency-aware scheduling |
| `tracker_assisted` | Reserved | Future tracker-guided recognition |
| `motion_triggered` | Reserved | Future motion-based triggering |

Client submission rate is controlled separately by
``submission_interval_ms``.

Deprecated API fields ``frame_skip`` and ``recognition_interval_ms`` are still
accepted and mapped to the new policy model for backward compatibility.

## Overlay Rendering

``OverlayRenderer`` converts recognition overlays into renderer-specific draw
instructions. ``RecognitionService`` remains unaware of rendering.

| Renderer | Status |
| --- | --- |
| `CanvasOverlayRenderer` | Enabled for Web Console |
| `OpenCVOverlayRenderer` | Reserved |
| `VideoOverlayRenderer` | Reserved |

## Recording Hook

``RecordingHook`` is reserved for future evidence capture:

- MP4 recording
- Snapshots
- Evidence storage

The default release ships ``NoOpRecordingHook`` only.

## LiveStream Implementations

| Stream | Status | Description |
| --- | --- | --- |
| `BrowserWebcamStream` | Enabled | Browser captures and submits frames |
| `OpenCVCameraStream` | Reserved | Server-side local camera via OpenCV |
| `RTSPStream` | Reserved | Network RTSP camera source |
| `USBStream` | Reserved | Dedicated USB camera device |
| `VideoFileStream` | Reserved | Recorded video file playback |

All stream implementations conform to ``LiveStream``:

- `open(config)`
- `close()`
- `read_frame()`
- `is_open`

## Frame Processing

Each submitted frame follows:

```
Frame bytes
    ↓
RecognitionService.decode_image()
    ↓
RecognitionService.recognize(image, profile)
    ↓
PipelineContext
    ↓
build_overlay(context)
    ↓
OverlayRenderer.build_render_spec()
```

## Web Console

Route: `/live`

- WebSocket-first frame transport with REST fallback
- Canvas overlay driven by server render spec
- Recognition policy configuration
- Performance metrics and timeline

## Production Readiness

| Area | Status |
| --- | --- |
| Domain models | Ready to freeze |
| Session manager | Ready to freeze |
| Recognition integration | Ready to freeze |
| Recognition policy | Ready to freeze |
| REST API | Ready to freeze |
| WebSocket transport | Ready to freeze |
| Overlay renderer abstraction | Ready to freeze |
| Recording hook | Ready to freeze (no-op) |
| Reserved stream backends | Documented only |
| Server-side capture / RTSP | Future |

The architecture is ready for review and freeze.
