"""
SCRFD decoder utilities.
"""

import numpy as np


def generate_center_priors(
    input_height: int,
    input_width: int,
    stride: int,
    num_anchors: int = 2,
) -> np.ndarray:
    """
    Generate center priors for one SCRFD feature level.

    Parameters
    ----------
    input_height : int
        Network input height.

    input_width : int
        Network input width.

    stride : int
        Feature map stride.

    num_anchors : int
        Number of anchors per feature location.

    Returns
    -------
    np.ndarray

        Shape:

        (N,3)

        columns:

        center_x

        center_y

        stride
    """

    feature_h = input_height // stride
    feature_w = input_width // stride

    x_coords = np.arange(feature_w, dtype=np.float32) * stride
    y_coords = np.arange(feature_h, dtype=np.float32) * stride

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    center_x = np.repeat(grid_x.reshape(-1), num_anchors)
    center_y = np.repeat(grid_y.reshape(-1), num_anchors)
    stride_column = np.full(center_x.shape, stride, dtype=np.float32)

    return np.column_stack(
        (
            center_x,
            center_y,
            stride_column,
        )
    )


def decode_bboxes(
    center_priors: np.ndarray,
    bbox_predictions: np.ndarray,
) -> np.ndarray:
    """
    Decode SCRFD bounding box predictions.

    Parameters
    ----------
    center_priors
        Shape: (N,3)

        Columns:
            cx
            cy
            stride

    bbox_predictions
        Shape: (N,4)

        Columns:
            left
            top
            right
            bottom

    Returns
    -------
    np.ndarray

        Shape: (N,4)

        Columns:

            x1
            y1
            x2
            y2
    """

    centers = center_priors[:, :2]

    strides = center_priors[:, 2:3]

    distances = bbox_predictions * strides

    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]

    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]

    return np.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        axis=1,
    )


def decode_landmarks(
    center_priors: np.ndarray,
    landmark_predictions: np.ndarray,
) -> np.ndarray:
    """
    Decode SCRFD facial landmark predictions.

    Parameters
    ----------
    center_priors
        Shape: (N,3)

        Columns:
            center_x
            center_y
            stride

    landmark_predictions
        Shape: (N,10)

        Columns:
            x1, y1, x2, y2, x3, y3, x4, y4, x5, y5

        Offset predictions that are scaled by stride before decoding.

    Returns
    -------
    np.ndarray
        Shape: (N,5,2)

        Decoded landmark coordinates as (x, y) pairs in this order:

        0. Left eye
        1. Right eye
        2. Nose
        3. Left mouth corner
        4. Right mouth corner
    """
    centers = center_priors[:, :2]
    strides = center_priors[:, 2:3]

    scaled_predictions = landmark_predictions * strides
    landmark_offsets = scaled_predictions.reshape(-1, 5, 2)

    return (centers[:, np.newaxis, :] + landmark_offsets).astype(np.float32)