"""Verify ArcFace embedding model loading without running inference."""

import common

from backend.ai.embedding.arcface.model import ArcFaceEmbeddingModel


def main() -> None:
    model = ArcFaceEmbeddingModel()
    model.load()

    engine = model._engine
    if engine is None or engine.session is None:
        raise RuntimeError("ArcFace model engine was not initialized.")

    print("\nArcFace Model Loaded Successfully\n")

    print(f"Provider: {model._provider}")
    print(f"Model: {model._model_name}")
    print(f"Input size: {model._input_size}")

    if len(engine.input_names) != 1:
        raise ValueError(
            f"Expected one input tensor, got {len(engine.input_names)}"
        )
    if len(engine.output_names) != 1:
        raise ValueError(
            f"Expected one output tensor, got {len(engine.output_names)}"
        )

    input_meta = engine.session.get_inputs()[0]
    output_meta = engine.session.get_outputs()[0]

    print(f"\nInput tensor: {input_meta.name}")
    print(f"Input shape: {input_meta.shape}")

    print(f"\nOutput tensor: {output_meta.name}")
    print(f"Output shape: {output_meta.shape}")

    input_shape = input_meta.shape
    if len(input_shape) != 4 or input_shape[1] != 3:
        raise ValueError(f"Expected NCHW input shape, got {input_shape}")
    if input_shape[2] != model._input_size or input_shape[3] != model._input_size:
        raise ValueError(
            "Input resolution does not match configuration: "
            f"shape={input_shape}, input_size={model._input_size}"
        )

    output_shape = output_meta.shape
    if len(output_shape) < 2 or not isinstance(output_shape[-1], int):
        raise ValueError(f"Could not resolve embedding dimension from {output_shape}")

    print(f"\nEmbedding dimension: {output_shape[-1]}")
    print("\nAll validations passed (no inference).\n")


if __name__ == "__main__":
    main()
