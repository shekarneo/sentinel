import numpy as np

import common
from backend.ai.alignment.template import get_reference_landmarks


def validate_template(template: np.ndarray) -> None:
    """Validate the reference landmark template layout."""
    if template.shape != (5, 2):
        raise RuntimeError(
            f"Unexpected template shape: {template.shape}. Expected: (5, 2)."
        )

    if template.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected template dtype: {template.dtype}. Expected: float32."
        )


def main() -> None:
    template = get_reference_landmarks()

    validate_template(template)

    print("--------------------------------")
    print("ArcFace Reference Landmark Template")
    print("--------------------------------")
    print()
    print(f"Shape : {template.shape}")
    print(f"Dtype : {template.dtype}")
    print()
    print("Coordinates:")
    landmark_names = [
        "Left Eye",
        "Right Eye",
        "Nose",
        "Left Mouth Corner",
        "Right Mouth Corner",
    ]

    for index, name in enumerate(landmark_names):
        x, y = template[index]
        print(f"  {index} {name:20s} : x={x:.4f}, y={y:.4f}")

    print()
    print("--------------------------------")


if __name__ == "__main__":
    main()
