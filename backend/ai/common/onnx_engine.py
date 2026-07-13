"""
ONNX Runtime inference engine.
"""

from pathlib import Path
from typing import Any

import onnxruntime as ort

from backend.ai.common.inference_engine import InferenceEngine


class ONNXRuntimeEngine(InferenceEngine):
    """
    ONNX Runtime implementation of the inference engine.
    """

    def __init__(self, model_path: Path):

        self.model_path = model_path

        self.session: ort.InferenceSession | None = None

        self.input_names: list[str] = []

        self.output_names: list[str] = []

    def load(self) -> None:
        """
        Load the ONNX model and initialize the inference session.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_names = [
            tensor.name
            for tensor in self.session.get_inputs()
        ]

        self.output_names = [
            tensor.name
            for tensor in self.session.get_outputs()
        ]

    def infer(self, inputs: Any) -> list[Any]:
        """
        Run inference.
        """

        if self.session is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.session.run(
            self.output_names,
            inputs,
        )