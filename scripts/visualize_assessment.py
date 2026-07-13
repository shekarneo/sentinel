"""Visualize face assessment metrics for every detected face."""

from pathlib import Path

import cv2
import numpy as np

import common
from common import load_image
from backend.ai.alignment.aligner import FaceAligner
from backend.ai.alignment.constants import CANONICAL_FACE_SIZE
from backend.ai.alignment.utils import crop_face_region
from backend.ai.assessment.assessor import FaceAssessor
from backend.ai.detection.scrfd.detector import SCRFDDetector
from backend.app.domain.face import Face

DISPLAY_SCALE = 3
PANEL_BORDER = 4




def build_square_panel(image: np.ndarray) -> np.ndarray:
    """Resize an image patch to the canonical face display size."""
    return cv2.resize(
        image,
        (CANONICAL_FACE_SIZE, CANONICAL_FACE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


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


def draw_metric_block(panel: np.ndarray, face: Face) -> np.ndarray:
    """Overlay assessment metrics on a visualization panel."""
    output = panel.copy()

    if face.assessment is None:
        raise RuntimeError("Face is missing assessment data.")

    assessment = face.assessment
    if any(
        section is None
        for section in (
            assessment.blur,
            assessment.brightness,
            assessment.pose,
            assessment.visibility,
            assessment.size,
        )
    ):
        raise RuntimeError("Face assessment is incomplete.")

    assert assessment.blur is not None
    assert assessment.brightness is not None
    assert assessment.pose is not None
    assert assessment.visibility is not None
    assert assessment.size is not None

    accept_label = "ACCEPT" if assessment.is_acceptable else "REJECT"
    accept_color = (0, 200, 0) if assessment.is_acceptable else (0, 0, 220)

    lines = [
        f"Blur: {assessment.blur.score:.2f}",
        f"Brightness: {assessment.brightness.score:.2f}",
        (
            "Pose: "
            f"y={assessment.pose.yaw:.1f} "
            f"p={assessment.pose.pitch:.1f} "
            f"r={assessment.pose.roll:.1f} "
            f"s={assessment.pose.score:.2f}"
        ),
        (
            "Visibility: "
            f"{assessment.visibility.visible_ratio:.2f} "
            f"s={assessment.visibility.score:.2f}"
        ),
        (
            "Size: "
            f"{assessment.size.width:.0f}x{assessment.size.height:.0f} "
            f"s={assessment.size.score:.2f}"
        ),
        f"Overall: {assessment.overall_score:.2f}",
        accept_label,
    ]

    y_offset = 18
    for line in lines[:-1]:
        cv2.putText(
            output,
            line,
            (8, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 16

    cv2.putText(
        output,
        lines[-1],
        (8, y_offset + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        accept_color,
        2,
        cv2.LINE_AA,
    )

    return output


def build_assessment_panel(image: np.ndarray, face: Face) -> np.ndarray:
    """Build a side-by-side original crop and aligned face assessment panel."""
    if face.alignment is None or face.alignment.aligned_image is None:
        raise RuntimeError("Face is missing aligned image.")

    original_crop = build_square_panel(crop_face_region(image, face))
    aligned_face = face.alignment.aligned_image.copy()

    original_panel = add_panel_border(original_crop, (0, 0, 255))
    aligned_panel = add_panel_border(aligned_face, (0, 255, 0))
    aligned_panel = draw_metric_block(aligned_panel, face)

    original_panel = upscale_panel(original_panel)
    aligned_panel = upscale_panel(aligned_panel)

    panel = np.hstack([original_panel, aligned_panel])

    cv2.putText(
        panel,
        "Original Face",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        "Aligned Face + Assessment",
        (original_panel.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return panel


def save_assessment_panels(
    image: np.ndarray,
    faces: list[Face],
    image_path: Path,
) -> list[Path]:
    """Save assessment visualization panels for each face."""
    saved_paths: list[Path] = []

    for index, face in enumerate(faces, start=1):
        panel = build_assessment_panel(image, face)
        output_path = image_path.with_name(
            f"{image_path.stem}_assessment_{index}{image_path.suffix}"
        )

        if not cv2.imwrite(str(output_path), panel):
            raise RuntimeError(f"Failed to save assessment panel: {output_path}")

        saved_paths.append(output_path)
        print(f"Saved assessment panel to: {output_path}")

    return saved_paths


def display_faces(
    image: np.ndarray,
    faces: list[Face],
    image_path: Path,
) -> None:
    """Display assessment visualization for each detected face."""
    window_name = "Face Assessment"

    try:
        for index, face in enumerate(faces, start=1):
            panel = build_assessment_panel(image, face)
            cv2.imshow(window_name, panel)

            assert face.assessment is not None
            print(
                f"Face {index}/{len(faces)} - "
                f"overall={face.assessment.overall_score:.4f} "
                f"acceptable={face.assessment.is_acceptable} - "
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
                    save_assessment_panels(image, faces, image_path)

        cv2.destroyAllWindows()
    except cv2.error:
        save_assessment_panels(image, faces, image_path)
        print("OpenCV GUI is unavailable in this environment.")


def main() -> None:
    """Run detection, alignment, assessment, and visualization."""
    import sys

    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python scripts/visualize_assessment.py <image_path>"
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

    assessor = FaceAssessor()
    assessor.assess(image, faces)

    save_assessment_panels(image, faces, image_path)
    display_faces(image, faces, image_path)


if __name__ == "__main__":
    main()
