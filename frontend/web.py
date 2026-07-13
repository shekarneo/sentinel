"""Web console routes and HTMX partial endpoints."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from frontend.pipeline_view import (
    build_debug_context,
    build_stages,
    build_timeline,
    encode_image_base64,
)

FRONTEND_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = FRONTEND_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["console"])


class RecognitionRenderPayload(BaseModel):
    """Stored recognition payload for history replay."""

    result: dict[str, Any]
    original_image_base64: str
    configuration: dict[str, Any] | None = None
    config_files: dict[str, str] | None = None
    api_latency_ms: float = 0.0
    run_id: str | None = None
    timestamp: str | None = None


def _page_context(request: Request, page: str, title: str) -> dict:
    return {
        "request": request,
        "page": page,
        "title": title,
    }


def _profile_config_from_configuration(
    configuration: dict[str, Any] | None,
    profile_name: str,
) -> dict[str, Any] | None:
    if not configuration:
        return None
    for profile in configuration.get("pipeline_profiles", []):
        if profile.get("name") == profile_name:
            return {
                "stages": profile.get("stages", []),
                "search": {
                    "enabled": profile.get("search_enabled", False),
                    "top_k": profile.get("search_top_k"),
                },
            }
    return None


def _render_recognition_panel(
    request: Request,
    *,
    result: dict[str, Any],
    original_image_base64: str,
    configuration: dict[str, Any] | None,
    config_files: dict[str, str] | None,
    api_latency_ms: float,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> HTMLResponse:
    profile_name = result.get("profile", "")
    profile_config = _profile_config_from_configuration(configuration, profile_name)
    stages = build_stages(result, profile_config)
    timeline = build_timeline(result)
    debug = build_debug_context(
        result=result,
        profile_config=profile_config,
        configuration=configuration,
        config_files=config_files,
        api_latency_ms=api_latency_ms,
    )
    faces = result.get("faces", [])
    bounding_boxes = [face.get("bounding_box", {}) for face in faces]
    run_id = run_id or str(uuid.uuid4())
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    payload_json = json.dumps(
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "result": result,
            "original_image_base64": original_image_base64,
            "configuration": configuration,
            "config_files": config_files,
            "api_latency_ms": api_latency_ms,
            "timeline": timeline,
        }
    )

    return templates.TemplateResponse(
        request,
        "components/recognition/results_panel.html",
        {
            "request": request,
            "result": result,
            "faces": faces,
            "bounding_boxes": bounding_boxes,
            "original_image_base64": original_image_base64,
            "stages": stages,
            "timeline": timeline,
            "debug": debug,
            "payload_json": payload_json,
        },
    )


async def _call_recognition_api(
    request: Request,
    *,
    profile: str,
    image_bytes: bytes,
    filename: str,
) -> tuple[dict[str, Any], float]:
    """Call the versioned recognition REST API through ASGI transport."""
    start = time.perf_counter()
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://console") as client:
        response = await client.post(
            "/api/v1/recognition",
            data={"profile": profile},
            files={"image": (filename, image_bytes, "application/octet-stream")},
        )
    latency_ms = (time.perf_counter() - start) * 1000.0
    if response.status_code >= 400:
        detail = response.json().get("message", "Recognition request failed.")
        raise ValueError(detail)
    return response.json(), latency_ms


async def _fetch_debug_sources(request: Request) -> tuple[dict[str, Any], dict[str, str]]:
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://console") as client:
        config_response, files_response = await client.get("/api/v1/configuration"), await client.get(
            "/api/v1/configuration/files"
        )
    configuration = config_response.json() if config_response.status_code == 200 else {}
    config_files = files_response.json() if files_response.status_code == 200 else {}
    return configuration, config_files


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/dashboard.html",
        _page_context(request, "dashboard", "Dashboard"),
    )


@router.get("/recognition", response_class=HTMLResponse, include_in_schema=False)
async def recognition_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/recognition.html",
        _page_context(request, "recognition", "Recognition"),
    )


@router.get("/live", response_class=HTMLResponse, include_in_schema=False)
async def live_camera_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/live_camera.html",
        _page_context(request, "live_camera", "Live Camera"),
    )


@router.post("/partials/recognition/run", response_class=HTMLResponse, include_in_schema=False)
async def recognition_run_partial(
    request: Request,
    profile: str = Form(...),
    image: UploadFile = File(...),
) -> HTMLResponse:
    """HTMX endpoint: run recognition and return results partial only."""
    image_bytes = await image.read()
    try:
        result, api_latency_ms = await _call_recognition_api(
            request,
            profile=profile,
            image_bytes=image_bytes,
            filename=image.filename or "probe.jpg",
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "components/recognition/error.html",
            {"request": request, "message": str(exc)},
            status_code=400,
        )

    configuration, config_files = await _fetch_debug_sources(request)
    return _render_recognition_panel(
        request,
        result=result,
        original_image_base64=encode_image_base64(image_bytes),
        configuration=configuration,
        config_files=config_files,
        api_latency_ms=api_latency_ms,
    )


@router.post("/partials/recognition/render", response_class=HTMLResponse, include_in_schema=False)
async def recognition_render_partial(
    request: Request,
    payload: RecognitionRenderPayload,
) -> HTMLResponse:
    """HTMX/Alpine endpoint: re-render a stored recognition payload."""
    return _render_recognition_panel(
        request,
        result=payload.result,
        original_image_base64=payload.original_image_base64,
        configuration=payload.configuration,
        config_files=payload.config_files,
        api_latency_ms=payload.api_latency_ms,
        run_id=payload.run_id,
        timestamp=payload.timestamp,
    )


@router.get("/enrollment", response_class=HTMLResponse, include_in_schema=False)
async def enrollment_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/enrollment.html",
        _page_context(request, "enrollment", "Enrollment"),
    )


@router.get("/gallery", response_class=HTMLResponse, include_in_schema=False)
async def gallery_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/gallery.html",
        _page_context(request, "gallery", "Gallery"),
    )


@router.get("/pipeline", response_class=HTMLResponse, include_in_schema=False)
async def pipeline_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/pipeline_explorer.html",
        _page_context(request, "pipeline", "Pipeline Explorer"),
    )


@router.get("/benchmarks", response_class=HTMLResponse, include_in_schema=False)
async def benchmarks_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/benchmarks.html",
        _page_context(request, "benchmarks", "Benchmarks"),
    )


@router.get("/executions", response_class=HTMLResponse, include_in_schema=False)
async def executions_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/executions.html",
        _page_context(request, "executions", "Execution History"),
    )


@router.get("/jobs", response_class=HTMLResponse, include_in_schema=False)
async def jobs_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/jobs.html",
        _page_context(request, "jobs", "Jobs"),
    )


@router.get("/datasets", response_class=HTMLResponse, include_in_schema=False)
async def datasets_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/datasets.html",
        _page_context(request, "datasets", "Dataset Processing"),
    )


@router.get("/configuration", response_class=HTMLResponse, include_in_schema=False)
async def configuration_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/configuration.html",
        _page_context(request, "configuration", "Configuration"),
    )


@router.get("/about", response_class=HTMLResponse, include_in_schema=False)
async def about_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "pages/about.html",
        _page_context(request, "about", "About"),
    )
