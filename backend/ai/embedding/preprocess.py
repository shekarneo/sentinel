"""
Embedding preprocessing utilities.

Prepares aligned face crops for ONNX embedding inference.
"""

import numpy as np

from backend.ai.embedding.utils import validate_aligned_face

_PIXEL_MEAN = 127.5
_PIXEL_SCALE = 127.5


def preprocess_aligned_face(
    aligned_face: np.ndarray,
    *,
    input_size: int,
) -> np.ndarray:
    """Prepare an aligned face crop for embedding inference.

    Pipeline:
        validate image -> BGR to RGB -> float32 -> normalize
        -> HWC to CHW -> NCHW -> contiguous float32 tensor

    Args:
        aligned_face: Aligned face image in HWC BGR uint8 layout.
        input_size: Expected square input resolution (height and width).

    Returns:
        Model-ready NCHW ``float32`` tensor with shape ``(1, 3, H, W)``.

    Raises:
        ValueError: If the aligned face image is invalid.
    """
    validate_aligned_face(aligned_face, expected_size=input_size)

    rgb_image = aligned_face[:, :, ::-1].astype(np.float32)
    normalized_image = (rgb_image - _PIXEL_MEAN) / _PIXEL_SCALE
    channel_first = np.transpose(normalized_image, (2, 0, 1))
    batch_tensor = np.expand_dims(channel_first, axis=0)

    return np.ascontiguousarray(batch_tensor, dtype=np.float32)
