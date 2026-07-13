import common
from backend.ai.detection.scrfd.decoder import decode_landmarks
import numpy as np


def main() -> None:
    priors = np.array(
        [
            [100, 200, 8],
        ],
        dtype=np.float32,
    )

    predictions = np.array(
        [
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        ],
        dtype=np.float32,
    )

    landmarks = decode_landmarks(
        priors,
        predictions,
    )

    print(f"Landmark tensor shape: {landmarks.shape}")
    print(landmarks)


if __name__ == "__main__":
    main()
