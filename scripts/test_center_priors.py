import common
from backend.ai.detection.scrfd.decoder import generate_center_priors


def main():
    priors = generate_center_priors(
        input_height=640,
        input_width=640,
        stride=8,
    )

    print(priors.shape)
    print(priors[:5])


if __name__ == "__main__":
    main()
