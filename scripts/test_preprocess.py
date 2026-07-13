import tempfile
from pathlib import Path

import cv2
import numpy as np

import common
from backend.ai.detection.scrfd.preprocess import preprocess

INPUT_SIZE = 640
TEST_IMAGE_HEIGHT = 900
TEST_IMAGE_WIDTH = 1200


def load_test_image() -> np.ndarray:
    """Load a test image using OpenCV.

    A temporary image is created when no bundled fixture is available so the
    script can run without external assets.
    """
    image = np.zeros(
        (TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH, 3),
        dtype=np.uint8,
    )
    image[:, :] = (64, 128, 192)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    cv2.imwrite(str(temp_path), image)
    loaded_image = cv2.imread(str(temp_path))
    temp_path.unlink(missing_ok=True)

    if loaded_image is None:
        raise RuntimeError("Failed to load test image with OpenCV.")

    return loaded_image


def verify_tensor_layout(tensor: np.ndarray) -> None:
    """Verify that the tensor matches the expected NCHW layout."""
    expected_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)

    if tensor.shape != expected_shape:
        raise ValueError(
            f"Unexpected tensor shape: {tensor.shape}. "
            f"Expected: {expected_shape}."
        )

    if tensor.dtype != np.float32:
        raise ValueError(
            f"Unexpected tensor dtype: {tensor.dtype}. "
            "Expected: float32."
        )

    if not tensor.flags["C_CONTIGUOUS"]:
        raise ValueError("Tensor memory is not C-contiguous.")


def main() -> None:
    image = load_test_image()
    tensor, scale, pad_x, pad_y = preprocess(image, input_size=INPUT_SIZE)

    verify_tensor_layout(tensor)

    print("----------------------------------------")
    print()
    print(f"Tensor Shape : {tensor.shape}")
    print(f"Tensor Dtype : {tensor.dtype}")
    print(f"Tensor Min   : {tensor.min():.1f}")
    print(f"Tensor Max   : {tensor.max():.1f}")
    print(f"Scale        : {scale:.4f}")
    print(f"Pad X        : {pad_x}")
    print(f"Pad Y        : {pad_y}")
    print()
    print("----------------------------------------")


if __name__ == "__main__":
    main()
