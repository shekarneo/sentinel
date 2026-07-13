from pathlib import Path

import cv2

import common
from common import load_image
from backend.ai.detection.scrfd.detector import SCRFDDetector

LANDMARK_COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
]




def draw_faces(image, faces):
    """Draw bounding boxes, confidence scores, and landmarks."""
    output = image.copy()

    for face in faces:
        bbox = face.bounding_box
        x1 = int(round(bbox.x))
        y1 = int(round(bbox.y))
        x2 = int(round(bbox.x + bbox.width))
        y2 = int(round(bbox.y + bbox.height))

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{face.confidence:.2f}"
        label_y = max(y1 - 10, 20)
        cv2.putText(
            output,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        for landmark_index, landmark in enumerate(face.landmarks):
            center = (int(round(landmark.x)), int(round(landmark.y)))
            color = LANDMARK_COLORS[landmark_index % len(LANDMARK_COLORS)]
            cv2.circle(output, center, 3, color, -1)

    return output


def save_output(output, image_path: Path) -> Path:
    """Save the annotated image next to the source image."""
    output_path = image_path.with_name(f"{image_path.stem}_scrfd{image_path.suffix}")

    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Failed to save visualization: {output_path}")

    print(f"Saved visualization to: {output_path}")
    return output_path


def display_output(output, image_path: Path) -> None:
    """Display the annotated image or save it when GUI is unavailable."""
    window_name = "SCRFD Detection"

    try:
        cv2.imshow(window_name, output)
        print("Press ESC to exit. Press S to save.")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:
                break

            if key in (ord("s"), ord("S")):
                save_output(output, image_path)

        cv2.destroyAllWindows()
        return
    except cv2.error:
        save_output(output, image_path)
        print("OpenCV GUI is unavailable in this environment.")


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python scripts/visualize_scrfd.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    image = load_image(image_path)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    faces = detector.detect(image)

    print(f"Detected {len(faces)} face(s)")

    output = draw_faces(image, faces)
    display_output(output, image_path)


if __name__ == "__main__":
    main()
