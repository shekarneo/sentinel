import common
from backend.ai.common.onnx_engine import ONNXRuntimeEngine


def main():
    engine = ONNXRuntimeEngine(common.SCRFD_MODEL_PATH)

    engine.load()

    print("\nModel Loaded Successfully\n")

    print("Inputs")

    for name in engine.input_names:
        print(f"  {name}")

    print("\nOutputs")

    for name in engine.output_names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
