import sys
from pathlib import Path

import cv2
import numpy as np

import common
from common import load_image
from backend.ai.common.onnx_engine import ONNXRuntimeEngine
from backend.ai.detection.scrfd.preprocess import preprocess

EXPECTED_OUTPUT_COUNT = 9




def validate_outputs(outputs: list[np.ndarray]) -> None:
    """Validate SCRFD inference outputs."""
    if len(outputs) != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTPUT_COUNT} outputs, got {len(outputs)}."
        )

    for index, tensor in enumerate(outputs):
        if tensor.dtype != np.float32:
            raise RuntimeError(
                f"Output {index} dtype is {tensor.dtype}, expected float32."
            )

        if np.isnan(tensor).any():
            raise RuntimeError(f"Output {index} contains NaN values.")

        if np.isinf(tensor).any():
            raise RuntimeError(f"Output {index} contains Inf values.")


def print_output_statistics(outputs: list[np.ndarray]) -> None:
    """Print statistics for each output tensor."""
    for index, tensor in enumerate(outputs):
        print("----------------------------------------")
        print()
        print(f"Tensor Index : {index}")
        print(f"Tensor Shape : {tensor.shape}")
        print(f"Tensor Dtype : {tensor.dtype}")
        print(f"Tensor Min   : {tensor.min():.6f}")
        print(f"Tensor Max   : {tensor.max():.6f}")
        print()
        print("----------------------------------------")


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python scripts/test_scrfd_inference.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    engine = ONNXRuntimeEngine(common.SCRFD_MODEL_PATH)
    engine.load()

    image = load_image(image_path)
    tensor, _, _, _ = preprocess(image)

    inputs = {engine.input_names[0]: tensor}
    outputs = engine.infer(inputs)

    validate_outputs(outputs)
    print_output_statistics(outputs)


if __name__ == "__main__":
    main()
