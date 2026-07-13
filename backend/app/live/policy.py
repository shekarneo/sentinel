"""Recognition policy for live camera sessions."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RecognitionPolicyType(str, Enum):
    """Supported recognition scheduling policies."""

    EVERY_FRAME = "every_frame"
    EVERY_N_FRAMES = "every_n_frames"
    ADAPTIVE = "adaptive"
    TRACKER_ASSISTED = "tracker_assisted"
    MOTION_TRIGGERED = "motion_triggered"


class RecognitionPolicy(BaseModel):
    """Configurable policy controlling when frames are recognized."""

    policy_type: RecognitionPolicyType = RecognitionPolicyType.EVERY_N_FRAMES
    frame_interval: int = Field(default=3, ge=1, description="Process every Nth captured frame.")

    def should_recognize(self, captured_frame_count: int, *, force: bool = False) -> bool:
        """Return whether the current frame should run through recognition."""
        if force:
            return True

        if self.policy_type == RecognitionPolicyType.EVERY_FRAME:
            return True

        if self.policy_type == RecognitionPolicyType.EVERY_N_FRAMES:
            return captured_frame_count % self.frame_interval == 0

        if self.policy_type in {
            RecognitionPolicyType.ADAPTIVE,
            RecognitionPolicyType.TRACKER_ASSISTED,
            RecognitionPolicyType.MOTION_TRIGGERED,
        }:
            raise NotImplementedError(
                f"Recognition policy {self.policy_type.value!r} is reserved for a future release."
            )

        return False

    @classmethod
    def from_legacy(cls, *, frame_skip: int | None) -> "RecognitionPolicy":
        """Map deprecated ``frame_skip`` to an ``every_n_frames`` policy."""
        if frame_skip is None:
            return cls()
        return cls(
            policy_type=RecognitionPolicyType.EVERY_N_FRAMES,
            frame_interval=max(frame_skip + 1, 1),
        )
