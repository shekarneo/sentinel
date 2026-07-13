import common
from backend.ai.detection.scrfd.decoder import decode_bboxes
import numpy as np


def main():
    priors = np.array(
        [
            [100, 200, 8],
        ],
        dtype=np.float32,
    )

    predictions = np.array(
        [
            [10, 20, 30, 40],
        ],
        dtype=np.float32,
    )

    boxes = decode_bboxes(
        priors,
        predictions,
    )

    print(boxes)


if __name__ == "__main__":
    main()
