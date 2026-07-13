"""
Non-maximum suppression utilities.
"""

import numpy as np


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Apply IoU-based non-maximum suppression.

    Args:
        boxes: Bounding boxes with shape ``(N, 4)`` in ``x1, y1, x2, y2`` format.
        scores: Confidence scores with shape ``(N,)``.
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.

    Returns:
        Indices of boxes to keep.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("Boxes and scores must contain the same number of detections.")

    if boxes.shape[0] == 1:
        return np.array([0], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []

    while order.size > 0:
        current_index = int(order[0])
        keep.append(current_index)

        if order.size == 1:
            break

        remaining_indices = order[1:]

        intersection_x1 = np.maximum(x1[current_index], x1[remaining_indices])
        intersection_y1 = np.maximum(y1[current_index], y1[remaining_indices])
        intersection_x2 = np.minimum(x2[current_index], x2[remaining_indices])
        intersection_y2 = np.minimum(y2[current_index], y2[remaining_indices])

        intersection_width = np.maximum(0.0, intersection_x2 - intersection_x1)
        intersection_height = np.maximum(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        union_area = (
            areas[current_index]
            + areas[remaining_indices]
            - intersection_area
        )
        iou = intersection_area / np.maximum(union_area, 1e-6)

        order = remaining_indices[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)
