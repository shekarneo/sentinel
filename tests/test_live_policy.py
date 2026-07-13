"""Recognition policy tests."""

import pytest

from backend.app.live.policy import RecognitionPolicy, RecognitionPolicyType


def test_every_frame_policy() -> None:
    policy = RecognitionPolicy(policy_type=RecognitionPolicyType.EVERY_FRAME)
    assert policy.should_recognize(1) is True
    assert policy.should_recognize(2) is True


def test_every_n_frames_policy() -> None:
    policy = RecognitionPolicy(
        policy_type=RecognitionPolicyType.EVERY_N_FRAMES,
        frame_interval=3,
    )
    assert policy.should_recognize(1) is False
    assert policy.should_recognize(3) is True


def test_legacy_frame_skip_mapping() -> None:
    policy = RecognitionPolicy.from_legacy(frame_skip=2)
    assert policy.frame_interval == 3
    assert policy.should_recognize(3) is True


def test_reserved_policy_raises() -> None:
    policy = RecognitionPolicy(policy_type=RecognitionPolicyType.ADAPTIVE)
    with pytest.raises(NotImplementedError):
        policy.should_recognize(1)
