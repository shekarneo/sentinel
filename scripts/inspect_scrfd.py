import common
from backend.ai.common.onnx_engine import ONNXRuntimeEngine


def main():
    engine = ONNXRuntimeEngine(common.SCRFD_MODEL_PATH)
    engine.load()

    print("\nINPUTS\n")

    for tensor in engine.session.get_inputs():
        print(f"Name : {tensor.name}")
        print(f"Shape: {tensor.shape}")
        print(f"Type : {tensor.type}")
        print()

    print("\nOUTPUTS\n")

    for tensor in engine.session.get_outputs():
        print(f"Name : {tensor.name}")
        print(f"Shape: {tensor.shape}")
        print(f"Type : {tensor.type}")
        print()


if __name__ == "__main__":
    main()
