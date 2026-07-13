"""View-model builders for recognition pipeline visualization."""

from __future__ import annotations

import base64
from typing import Any

STAGE_ORDER: list[tuple[str, str]] = [
    ("scrfd", "Detection"),
    ("alignment", "Alignment"),
    ("assessment", "Assessment"),
    ("embedding", "Embedding"),
    ("search", "Search"),
    ("verification", "Verification"),
]

STATUS_BADGES: dict[str, tuple[str, str]] = {
    "running": ("Running", "text-bg-primary"),
    "success": ("Success", "text-bg-success"),
    "warning": ("Warning", "text-bg-warning"),
    "failed": ("Failed", "text-bg-danger"),
    "skipped": ("Skipped", "text-bg-secondary"),
}


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


def _profile_stages(profile_config: dict[str, Any] | None) -> list[str]:
    if not profile_config:
        return [key for key, _ in STAGE_ORDER]
    return list(profile_config.get("stages", []))


def _search_payload(metadata: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]] | None:
    search = metadata.get("search_results")
    if isinstance(search, list) and search:
        return search[0]
    return search if isinstance(search, dict) else None


def _verification_payload(metadata: dict[str, Any]) -> dict[str, Any] | None:
    verification = metadata.get("verification")
    if isinstance(verification, list) and verification:
        return verification[0]
    return verification if isinstance(verification, dict) else None


def _stage_status(key: str, result: dict[str, Any], profile_stages: list[str]) -> str:
    if key not in profile_stages:
        return "skipped"

    timings = result.get("timings", {}).get("stages", {})
    faces = result.get("faces", [])
    metadata = result.get("metadata", {})

    if key == "scrfd":
        if result.get("face_count", 0) == 0:
            return "failed"
        return "success"

    if key == "alignment":
        if not any(face.get("has_alignment") for face in faces):
            return "failed" if "alignment" in profile_stages else "skipped"
        return "success"

    if key == "assessment":
        assessments = [face.get("assessment") for face in faces if face.get("assessment")]
        if not assessments:
            return "skipped"
        if any(item.get("is_acceptable") is False for item in assessments):
            return "warning"
        return "success"

    if key == "embedding":
        if not any(face.get("embedding") for face in faces):
            return "failed" if "embedding" in profile_stages else "skipped"
        return "success"

    if key == "search":
        search = _search_payload(metadata)
        if not search:
            return "skipped"
        if not search.get("results"):
            return "warning"
        return "success"

    if key == "verification":
        verification = _verification_payload(metadata)
        if not verification:
            return "skipped"
        if verification.get("decision") == "reject":
            return "warning"
        if verification.get("decision") == "unknown":
            return "warning"
        return "success"

    if key in timings:
        return "success"
    return "skipped"


def _stage_summary(key: str, result: dict[str, Any]) -> str:
    faces = result.get("faces", [])
    metadata = result.get("metadata", {})

    if key == "scrfd":
        return f"{result.get('face_count', 0)} face(s) detected"

    if key == "alignment":
        count = sum(1 for face in faces if face.get("has_alignment"))
        return f"{count} aligned crop(s) produced"

    if key == "assessment":
        acceptable = [
            face.get("assessment", {}).get("is_acceptable")
            for face in faces
            if face.get("assessment")
        ]
        if not acceptable:
            return "Assessment not run for profile"
        ok = sum(1 for value in acceptable if value)
        return f"{ok}/{len(acceptable)} acceptable"

    if key == "embedding":
        model = faces[0].get("embedding", {}).get("model_name") if faces else None
        dim = faces[0].get("embedding", {}).get("dimension") if faces else None
        if model and dim:
            return f"{model} · {dim}D vector"
        return "Embedding generated"

    if key == "search":
        search = _search_payload(metadata)
        if not search:
            return "Search disabled for profile"
        count = len(search.get("results", []))
        return f"{count} candidate(s) · {search.get('search_time_ms', 0):.1f} ms"

    if key == "verification":
        verification = _verification_payload(metadata)
        if not verification:
            return "Verification not run for profile"
        return f"{verification.get('decision', 'unknown')} · score {verification.get('similarity_score', 0):.3f}"

    return "Stage completed"


def build_stages(
    result: dict[str, Any],
    profile_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build pipeline stage cards for template rendering."""
    profile_stages = _profile_stages(profile_config)
    timings = result.get("timings", {}).get("stages", {})
    stages: list[dict[str, Any]] = []

    for key, label in STAGE_ORDER:
        status = _stage_status(key, result, profile_stages)
        badge_label, badge_class = STATUS_BADGES[status]
        stages.append(
            {
                "key": key,
                "label": label,
                "status": status,
                "badge_label": badge_label,
                "badge_class": badge_class,
                "time_ms": _ms(timings.get(key, 0.0)),
                "summary": _stage_summary(key, result),
                "details": build_stage_details(key, result),
            }
        )
    return stages


def build_stage_details(key: str, result: dict[str, Any]) -> dict[str, Any]:
    """Build structured detail payloads per stage."""
    faces = result.get("faces", [])
    metadata = result.get("metadata", {})

    if key == "scrfd":
        return {
            "face_count": result.get("face_count", 0),
            "faces": [
                {
                    "confidence": face.get("confidence"),
                    "bounding_box": face.get("bounding_box"),
                }
                for face in faces
            ],
        }

    if key == "alignment":
        return {
            "faces": [
                {
                    "has_alignment": face.get("has_alignment"),
                    "aligned_image_base64": face.get("aligned_image_base64"),
                    "quality": _alignment_quality(face),
                }
                for face in faces
            ],
        }

    if key == "assessment":
        return {
            "faces": [face.get("assessment") for face in faces if face.get("assessment")],
        }

    if key == "embedding":
        return {
            "faces": [
                {
                    **(face.get("embedding") or {}),
                    "vector_norm": 1.0 if (face.get("embedding") or {}).get("normalized") else None,
                }
                for face in faces
                if face.get("embedding")
            ],
        }

    if key == "search":
        search = _search_payload(metadata)
        return search or {}

    if key == "verification":
        verification = _verification_payload(metadata)
        if not verification:
            return {}
        policy = (verification.get("metadata") or {}).get("policy", "threshold")
        return {**verification, "policy": policy}

    return {}


def _alignment_quality(face: dict[str, Any]) -> str:
    assessment = face.get("assessment") or {}
    pose = assessment.get("pose_score")
    if pose is None:
        return "aligned" if face.get("has_alignment") else "missing"
    if pose >= 0.8:
        return "excellent"
    if pose >= 0.6:
        return "acceptable"
    return "low"


def build_timeline(result: dict[str, Any]) -> dict[str, Any]:
    """Build horizontal execution timeline data."""
    timings = result.get("timings", {}).get("stages", {})
    items: list[dict[str, Any]] = []
    total = 0.0

    for key, label in STAGE_ORDER:
        seconds = timings.get(key, 0.0)
        if seconds <= 0:
            continue
        ms = _ms(seconds)
        total += ms
        items.append({"key": key, "label": label, "time_ms": ms})

    max_ms = max((item["time_ms"] for item in items), default=1.0)
    for item in items:
        item["width_pct"] = max(12, int((item["time_ms"] / max_ms) * 100))

    return {
        "entries": items,
        "total_ms": round(total, 1),
    }


def build_debug_context(
    *,
    result: dict[str, Any],
    profile_config: dict[str, Any] | None,
    configuration: dict[str, Any] | None,
    config_files: dict[str, str] | None,
    api_latency_ms: float,
) -> dict[str, Any]:
    """Build debug panel context."""
    models = []
    thresholds_excerpt = ""
    if configuration:
        app_info = configuration.get("app", {})
        models = configuration.get("pipeline_profiles", [])
    if config_files:
        thresholds_excerpt = _threshold_excerpt(config_files.get("thresholds_yaml", ""))

    return {
        "profile": result.get("profile"),
        "profile_config": profile_config,
        "api_latency_ms": round(api_latency_ms, 1),
        "app": (configuration or {}).get("app", {}),
        "models_yaml": (config_files or {}).get("models_yaml", ""),
        "thresholds_excerpt": thresholds_excerpt,
        "metadata_keys": sorted(result.get("metadata", {}).keys()),
    }


def _threshold_excerpt(yaml_text: str, max_lines: int = 12) -> str:
    lines = yaml_text.splitlines()
    return "\n".join(lines[:max_lines])


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode uploaded image bytes for inline browser display."""
    return base64.b64encode(image_bytes).decode("ascii")
