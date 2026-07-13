from pathlib import Path

import cv2
import numpy as np

import common
from common import load_image
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.alignment.constants import CANONICAL_FACE_SIZE
from backend.ai.alignment.template import get_reference_landmarks
from backend.ai.alignment.utils import crop_face_region, landmarks_to_array
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.app.domain.face import Face

DISPLAY_SCALE = 3
PANEL_BORDER = 4

LANDMARK_COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
]




def build_naive_square_crop(crop: np.ndarray) -> np.ndarray:
    """Stretch a bbox crop to square size without alignment."""
    return cv2.resize(
        crop,
        (CANONICAL_FACE_SIZE, CANONICAL_FACE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


def landmarks_on_naive_crop(face: Face, crop: np.ndarray) -> np.ndarray:
    """Map detected landmarks into naive square crop coordinates."""
    bbox = face.bounding_box
    scale_x = CANONICAL_FACE_SIZE / crop.shape[1]
    scale_y = CANONICAL_FACE_SIZE / crop.shape[0]

    points = []
    for landmark in face.landmarks:
        x = (landmark.x - bbox.x) * scale_x
        y = (landmark.y - bbox.y) * scale_y
        points.append([x, y])

    return np.array(points, dtype=np.float32)


def landmarks_on_aligned_face(face: Face) -> np.ndarray:
    """Map detected landmarks into aligned face coordinates."""
    if face.alignment is None or face.alignment.transformation_matrix is None:
        raise RuntimeError("Face is missing alignment transform.")

    source = landmarks_to_array(face.landmarks)
    return cv2.transform(
        source.reshape(5, 1, 2),
        face.alignment.transformation_matrix,
    ).reshape(5, 2)


def draw_landmarks(
    image: np.ndarray,
    points: np.ndarray,
    colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw landmark points on an image."""
    output = image.copy()

    for index, point in enumerate(points):
        center = (int(round(point[0])), int(round(point[1])))
        color = colors[index] if colors else (0, 255, 0)
        cv2.circle(output, center, 3, color, -1)
        cv2.circle(output, center, 4, (0, 0, 0), 1)

    return output


def upscale_panel(panel: np.ndarray) -> np.ndarray:
    """Upscale a panel for easier visual inspection."""
    display_size = CANONICAL_FACE_SIZE * DISPLAY_SCALE
    return cv2.resize(
        panel,
        (display_size, display_size),
        interpolation=cv2.INTER_NEAREST,
    )


def add_panel_border(panel: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Add a colored border around a panel."""
    return cv2.copyMakeBorder(
        panel,
        PANEL_BORDER,
        PANEL_BORDER,
        PANEL_BORDER,
        PANEL_BORDER,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )


def landmark_alignment_error(face: Face) -> float:
    """Return the maximum landmark error against the ArcFace template."""
    if face.alignment is None or face.alignment.transformation_matrix is None:
        raise RuntimeError("Face is missing alignment transform.")

    reference_landmarks = get_reference_landmarks()
    transformed = landmarks_on_aligned_face(face)
    return float(np.max(np.abs(transformed - reference_landmarks)))


def build_comparison_panel(
    image: np.ndarray,
    face: Face,
) -> np.ndarray:
    """Build a side-by-side panel contrasting naive crop vs aligned face.

    Left panel:
        Bounding-box crop stretched to 112x112 with detected landmarks.
        No rotation or canonical normalization is applied.

    Right panel:
        Similarity-aligned 112x112 face with warped landmarks and the
        ArcFace reference template overlaid for comparison.
    """
    if face.alignment is None or face.alignment.aligned_image is None:
        raise RuntimeError("Face is missing aligned image.")

    crop = crop_face_region(image, face)
    naive_face = build_naive_square_crop(crop)
    aligned_face = face.alignment.aligned_image.copy()
    reference_landmarks = get_reference_landmarks()

    naive_points = landmarks_on_naive_crop(face, crop)
    aligned_points = landmarks_on_aligned_face(face)

    naive_panel = draw_landmarks(naive_face, naive_points, LANDMARK_COLORS)
    aligned_panel = draw_landmarks(aligned_face, aligned_points, LANDMARK_COLORS)
    aligned_panel = draw_landmarks(
        aligned_panel,
        reference_landmarks,
        [(255, 255, 255)] * 5,
    )

    naive_panel = add_panel_border(naive_panel, (0, 0, 255))
    aligned_panel = add_panel_border(aligned_panel, (0, 255, 0))

    naive_panel = upscale_panel(naive_panel)
    aligned_panel = upscale_panel(aligned_panel)

    panel = np.hstack([naive_panel, aligned_panel])

    cv2.putText(
        panel,
        "BBox crop (stretched)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        "Aligned (ArcFace 5-point)",
        (naive_panel.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        "Colored dots = detected landmarks",
        (12, panel.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        panel,
        "White dots = ArcFace template (eyes/nose/mouth)",
        (naive_panel.shape[1] + 12, panel.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )

    return panel


def save_aligned_faces(
    faces: list[Face],
    image_path: Path,
) -> list[Path]:
    """Save aligned face crops next to the source image."""
    saved_paths: list[Path] = []

    for index, face in enumerate(faces, start=1):
        if face.alignment is None or face.alignment.aligned_image is None:
            continue

        output_path = image_path.with_name(
            f"{image_path.stem}_aligned_{index}{image_path.suffix}"
        )

        if not cv2.imwrite(str(output_path), face.alignment.aligned_image):
            raise RuntimeError(f"Failed to save aligned face: {output_path}")

        saved_paths.append(output_path)
        print(f"Saved aligned face to: {output_path}")

    return saved_paths


def save_comparison_panels(
    image: np.ndarray,
    faces: list[Face],
    image_path: Path,
) -> list[Path]:
    """Save side-by-side comparison panels for each face."""
    saved_paths: list[Path] = []

    for index, face in enumerate(faces, start=1):
        panel = build_comparison_panel(image, face)
        output_path = image_path.with_name(
            f"{image_path.stem}_alignment_compare_{index}{image_path.suffix}"
        )

        if not cv2.imwrite(str(output_path), panel):
            raise RuntimeError(f"Failed to save comparison panel: {output_path}")

        saved_paths.append(output_path)
        print(f"Saved comparison panel to: {output_path}")

    return saved_paths


def display_faces(
    image: np.ndarray,
    faces: list[Face],
    image_path: Path,
) -> None:
    """Display naive crop vs aligned face for each detection."""
    window_name = "Face Alignment"

    try:
        for index, face in enumerate(faces, start=1):
            panel = build_comparison_panel(image, face)
            max_error = landmark_alignment_error(face)

            cv2.imshow(window_name, panel)
            print(
                f"Face {index}/{len(faces)} - "
                f"max landmark error: {max_error:.2f}px - "
                "Press ESC to exit, N for next face, S to save outputs."
            )

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == 27:
                    cv2.destroyAllWindows()
                    return

                if key in (ord("n"), ord("N")):
                    break

                if key in (ord("s"), ord("S")):
                    save_aligned_faces(faces, image_path)
                    save_comparison_panels(image, faces, image_path)

        cv2.destroyAllWindows()
    except cv2.error:
        save_aligned_faces(faces, image_path)
        save_comparison_panels(image, faces, image_path)
        print("OpenCV GUI is unavailable in this environment.")


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python scripts/visualize_alignment.py <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise RuntimeError(f"Image not found: {image_path}")

    image = load_image(image_path)
    detector = SCRFDDetector(common.SCRFD_MODEL_PATH)
    faces = detector.detect(image)

    print(f"Detected {len(faces)} face(s)")

    aligner = FaceAligner()
    aligner.align(image, faces)

    save_comparison_panels(image, faces, image_path)
    display_faces(image, faces, image_path)


if __name__ == "__main__":
    main()
