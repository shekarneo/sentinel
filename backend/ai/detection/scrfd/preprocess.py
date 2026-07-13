"""
SCRFD image preprocessing utilities.
"""

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    input_width: int = 640,
    input_height: int = 640,
) -> tuple[np.ndarray, float, int, int]:
    """Resize an image with aspect-ratio preservation and symmetric padding.

    The image is scaled to fit within a canvas of ``input_width`` x
    ``input_height`` while preserving its aspect ratio. Remaining space is
    filled with a neutral padding color.

    Args:
        image: Input image in HWC layout with shape ``(H, W, C)``.
        input_width: Target canvas width for the letterboxed image.
        input_height: Target canvas height for the letterboxed image.

    Returns:
        A tuple containing:
            - Letterboxed image with shape ``(input_height, input_width, C)``.
            - Resize scale applied to the original image dimensions.
            - Horizontal padding applied to the left side.
            - Vertical padding applied to the top side.
    """
    height, width = image.shape[:2]

    scale = min(input_height / height, input_width / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2

    letterboxed_image = cv2.copyMakeBorder(
        resized_image,
        pad_y,
        input_height - resized_height - pad_y,
        pad_x,
        input_width - resized_width - pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return letterboxed_image, scale, pad_x, pad_y


def normalize(image: np.ndarray) -> np.ndarray:
    """Convert an image array to SCRFD input range.

    SCRFD expects pixel values normalized as ``(pixel - 127.5) / 128``.

    Args:
        image: Input image array, typically ``uint8``.

    Returns:
        Normalized image as ``float32``.
    """
    return (image.astype(np.float32) - 127.5) / 128.0


def to_tensor(image: np.ndarray) -> np.ndarray:
    """Convert an HWC image into a contiguous NCHW tensor.

    Args:
        image: Input image in HWC layout.

    Returns:
        Contiguous tensor with shape ``(1, C, H, W)`` and dtype ``float32``.
    """
    channel_first = np.transpose(image, (2, 0, 1))
    batch_tensor = np.expand_dims(channel_first, axis=0)
    return np.ascontiguousarray(batch_tensor)


def preprocess(
    image: np.ndarray,
    input_size: int | tuple[int, int] = 640,
) -> tuple[np.ndarray, float, int, int]:
    """Run the full SCRFD preprocessing pipeline on an input image.

    Pipeline:
        letterbox -> normalize -> tensor conversion

    Args:
        image: Input image in HWC layout, typically loaded with OpenCV.
        input_size: Target model input size as a square side length or
            ``(width, height)`` tuple.

    Returns:
        A tuple containing:
            - Model input tensor with shape ``(1, 3, H, W)``.
            - Resize scale applied during letterboxing.
            - Horizontal padding applied to the left side.
            - Vertical padding applied to the top side.
    """
    if isinstance(input_size, int):
        input_width = input_height = input_size
    else:
        input_width, input_height = input_size

    letterboxed_image, scale, pad_x, pad_y = letterbox(
        image,
        input_width,
        input_height,
    )
    normalized_image = normalize(letterboxed_image)
    tensor = to_tensor(normalized_image)
    return tensor, scale, pad_x, pad_y
