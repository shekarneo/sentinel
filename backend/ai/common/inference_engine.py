"""
Abstract inference engine interface.

Every inference backend (ONNX Runtime, TensorRT, OpenVINO, etc.)
must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class InferenceEngine(ABC):
    """Base interface for all inference engines."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, inputs: Any) -> Any:
        """Run inference."""
        raise NotImplementedError