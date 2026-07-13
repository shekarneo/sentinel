"""
SCRFD detector orchestration.
"""

import logging
from pathlib import Path

import numpy as np

from backend.ai.common.onnx_engine import ONNXRuntimeEngine
from backend.ai.detection.detector import FaceDetector
from backend.ai.detection.scrfd.constants import NUM_ANCHORS, STRIDES
from backend.ai.detection.scrfd.decoder import (
    decode_bboxes,
    decode_landmarks,
    generate_center_priors,
)
from backend.ai.detection.scrfd.postprocess import postprocess
from backend.ai.detection.scrfd.preprocess import preprocess
from backend.app.config.configuration import Configuration, resolve_scrfd_model_path
from backend.app.domain.face import BoundingBox, Face, Landmark

logger = logging.getLogger(__name__)


class SCRFDDetector(FaceDetector):
    """Orchestrates the SCRFD detection pipeline."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Initialize the detector and load the SCRFD ONNX model.

        Args:
            model_path: Path to the SCRFD ONNX model file. When omitted, the
                path is resolved from ``configs/models.yaml``.
        """
        configuration = Configuration()
        thresholds = configuration.load_thresholds().detection.scrfd
        scrfd_settings = configuration.load_models().detection.scrfd

        self._score_threshold = thresholds.score_threshold
        self._nms_iou_threshold = thresholds.nms_iou_threshold
        self._input_size = (scrfd_settings.input.width, scrfd_settings.input.height)

        if model_path is None:
            model_path = resolve_scrfd_model_path()

        self._engine = ONNXRuntimeEngine(model_path)
        self._engine.load()
        logger.info("Loaded SCRFD model from %s", model_path)

    def detect(self, image: np.ndarray) -> list[Face]:
        """Run the SCRFD detection pipeline on an input image.

        Args:
            image: Input image in HWC layout, typically loaded with OpenCV.

        Returns:
            Detected faces as domain objects.

        Raises:
            ValueError: If the input image is invalid.
        """
        _validate_image(image)

        image_height, image_width = image.shape[:2]
        tensor, scale, pad_x, pad_y = preprocess(image, input_size=self._input_size)
        _, _, input_height, input_width = tensor.shape

        outputs = self._engine.infer(
            {
                self._engine.input_names[0]: tensor,
            }
        )

        score_outputs = outputs[0:3]
        box_outputs = outputs[3:6]
        landmark_outputs = outputs[6:9]

        decoded_scores: list[np.ndarray] = []
        decoded_boxes: list[np.ndarray] = []
        decoded_landmarks: list[np.ndarray] = []

        for stride, score_output, box_output, landmark_output in zip(
            STRIDES,
            score_outputs,
            box_outputs,
            landmark_outputs,
            strict=True,
        ):
            center_priors = generate_center_priors(
                input_height=input_height,
                input_width=input_width,
                stride=stride,
                num_anchors=NUM_ANCHORS,
            )

            decoded_scores.append(score_output)
            decoded_boxes.append(decode_bboxes(center_priors, box_output))
            decoded_landmarks.append(
                decode_landmarks(center_priors, landmark_output)
            )

        scores = np.concatenate(decoded_scores, axis=0)
        boxes = np.concatenate(decoded_boxes, axis=0)
        landmarks = np.concatenate(decoded_landmarks, axis=0)

        filtered_scores, filtered_boxes, filtered_landmarks = postprocess(
            scores=scores,
            boxes=boxes,
            landmarks=landmarks,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            image_height=image_height,
            image_width=image_width,
            score_threshold=self._score_threshold,
            iou_threshold=self._nms_iou_threshold,
        )

        faces = self._create_faces(
            filtered_scores,
            filtered_boxes,
            filtered_landmarks,
        )
        logger.debug("Detected %d faces", len(faces))
        return faces

    def _create_faces(
        self,
        scores: np.ndarray,
        boxes: np.ndarray,
        landmarks: np.ndarray,
    ) -> list[Face]:
        """Convert filtered detections into Face domain objects."""
        faces: list[Face] = []

        for score, box, face_landmarks in zip(
            scores,
            boxes,
            landmarks,
            strict=True,
        ):
            x1, y1, x2, y2 = box

            faces.append(
                Face(
                    bounding_box=BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                    ),
                    confidence=float(score),
                    landmarks=[
                        Landmark(x=float(point[0]), y=float(point[1]))
                        for point in face_landmarks
                    ],
                )
            )

        return faces


def _validate_image(image: np.ndarray) -> None:
    """Validate detector input image layout."""
    if image is None:
        raise ValueError("Input image must not be None.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HWC BGR image with 3 channels.")

    if image.size == 0:
        raise ValueError("Input image must not be empty.")
