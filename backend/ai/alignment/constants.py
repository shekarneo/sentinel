"""
Face alignment constants.

Defines canonical face dimensions and landmark indices used across the
alignment pipeline.
"""

# Canonical output size for aligned face crops (width and height).
CANONICAL_FACE_SIZE: int = 112

# Number of landmarks required for alignment (matches SCRFD output).
NUM_LANDMARKS: int = 5

# Landmark indices — must match Face domain model ordering.
LANDMARK_LEFT_EYE: int = 0
LANDMARK_RIGHT_EYE: int = 1
LANDMARK_NOSE: int = 2
LANDMARK_LEFT_MOUTH: int = 3
LANDMARK_RIGHT_MOUTH: int = 4
