"""
SCRFD detection postprocessing utilities.
"""

import numpy as np

from backend.ai.detection.common.nms import nms
from backend.app.config.configuration import ScrfdThresholdSettings


def postprocess(
    scores: np.ndarray,
    boxes: np.ndarray,
    landmarks: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    image_height: int,
    image_width: int,
    score_threshold: float | None = None,
    iou_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter and remap SCRFD detections to original image coordinates.

    Args:
        scores: Detection scores with shape ``(N, 1)`` or ``(N,)``.
        boxes: Decoded boxes with shape ``(N, 4)`` in letterboxed coordinates.
        landmarks: Decoded landmarks with shape ``(N, 5, 2)``.
        scale: Letterbox resize scale from preprocessing.
        pad_x: Horizontal padding applied during preprocessing.
        pad_y: Vertical padding applied during preprocessing.
        image_height: Original image height.
        image_width: Original image width.
        score_threshold: Minimum score required to keep a detection.
        iou_threshold: IoU threshold used for non-maximum suppression.

    Returns:
        Filtered ``scores``, ``boxes``, and ``landmarks`` in original image
        coordinates.
    """
    defaults = ScrfdThresholdSettings()

    if score_threshold is None:
        score_threshold = defaults.score_threshold

    if iou_threshold is None:
        iou_threshold = defaults.nms_iou_threshold

    flat_scores = np.squeeze(scores).astype(np.float32)
    filtered_indices = np.where(flat_scores >= score_threshold)[0]

    if filtered_indices.size == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
        )

    filtered_scores = flat_scores[filtered_indices]
    filtered_boxes = boxes[filtered_indices].astype(np.float32)
    filtered_landmarks = landmarks[filtered_indices].astype(np.float32)

    keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold)

    filtered_scores = filtered_scores[keep_indices]
    filtered_boxes = filtered_boxes[keep_indices]
    filtered_landmarks = filtered_landmarks[keep_indices]

    filtered_boxes = _undo_letterbox_boxes(
        filtered_boxes,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    filtered_landmarks = _undo_letterbox_landmarks(
        filtered_landmarks,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )

    filtered_boxes = _clip_boxes(
        filtered_boxes,
        image_height=image_height,
        image_width=image_width,
    )
    filtered_landmarks = _clip_landmarks(
        filtered_landmarks,
        image_height=image_height,
        image_width=image_width,
    )

    return filtered_scores, filtered_boxes, filtered_landmarks


def _undo_letterbox_boxes(
    boxes: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Map boxes from letterboxed coordinates back to the original image."""
    remapped_boxes = boxes.copy()
    remapped_boxes[:, [0, 2]] = (remapped_boxes[:, [0, 2]] - pad_x) / scale
    remapped_boxes[:, [1, 3]] = (remapped_boxes[:, [1, 3]] - pad_y) / scale
    return remapped_boxes


def _undo_letterbox_landmarks(
    landmarks: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Map landmarks from letterboxed coordinates back to the original image."""
    remapped_landmarks = landmarks.copy()
    remapped_landmarks[:, :, 0] = (remapped_landmarks[:, :, 0] - pad_x) / scale
    remapped_landmarks[:, :, 1] = (remapped_landmarks[:, :, 1] - pad_y) / scale
    return remapped_landmarks


def _clip_boxes(
    boxes: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Clip box coordinates to image boundaries."""
    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = np.clip(
        clipped_boxes[:, [0, 2]],
        0,
        image_width,
    )
    clipped_boxes[:, [1, 3]] = np.clip(
        clipped_boxes[:, [1, 3]],
        0,
        image_height,
    )
    return clipped_boxes


def _clip_landmarks(
    landmarks: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Clip landmark coordinates to image boundaries."""
    clipped_landmarks = landmarks.copy()
    clipped_landmarks[:, :, 0] = np.clip(
        clipped_landmarks[:, :, 0],
        0,
        image_width,
    )
    clipped_landmarks[:, :, 1] = np.clip(
        clipped_landmarks[:, :, 1],
        0,
        image_height,
    )
    return clipped_landmarks
