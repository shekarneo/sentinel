"""Run the Pipeline Orchestrator across all built-in profiles on an image."""

from pathlib import Path

import cv2

import common
from common import load_image
from backend.app.pipeline import (
    PipelineBuilder,
    PipelineContext,
    PipelineExecutor,
    PipelineProfile,
    create_default_registry,
    load_pipeline_profile_settings,
)
from backend.app.domain.face import Face




def count_faces_with_alignment(faces: list[Face]) -> int:
    """Count faces that have alignment data populated."""
    return sum(
        1
        for face in faces
        if face.alignment is not None and face.alignment.aligned_image is not None
    )


def count_faces_with_assessment(faces: list[Face]) -> int:
    """Count faces that have assessment data populated."""
    return sum(1 for face in faces if face.assessment is not None)


def validate_profile_result(
    profile: PipelineProfile,
    context: PipelineContext,
    stage_names: list[str],
) -> None:
    """Validate pipeline output for a profile.

    Args:
        profile: Executed pipeline profile.
        context: Resulting pipeline context.
        stage_names: Ordered stage names that were executed.

    Raises:
        ValueError: If pipeline output is invalid.
    """
    if context.errors:
        raise ValueError(
            f"Profile '{profile.value}' failed: {context.errors}"
        )

    if not context.faces:
        raise ValueError(f"Profile '{profile.value}' detected no faces.")

    if "alignment" in stage_names:
        aligned = count_faces_with_alignment(context.faces)
        if aligned != len(context.faces):
            raise ValueError(
                f"Profile '{profile.value}' expected all faces aligned, "
                f"got {aligned}/{len(context.faces)}."
            )

    if "assessment" in stage_names:
        assessed = count_faces_with_assessment(context.faces)
        if assessed != len(context.faces):
            raise ValueError(
                f"Profile '{profile.value}' expected all faces assessed, "
                f"got {assessed}/{len(context.faces)}."
            )

    expected_timings = set(stage_names)
    actual_timings = set(context.timings)
    if expected_timings != actual_timings:
        raise ValueError(
            f"Profile '{profile.value}' timing mismatch. "
            f"expected={sorted(expected_timings)} actual={sorted(actual_timings)}"
        )


def print_profile_result(
    profile: PipelineProfile,
    context: PipelineContext,
    stage_names: list[str],
) -> None:
    """Print pipeline execution summary for a profile.

    Args:
        profile: Executed pipeline profile.
        context: Resulting pipeline context.
        stage_names: Ordered stage names that were executed.
    """
    total_time = sum(context.timings.values())

    print(f"Profile: {profile.value}")
    print(f"Stages: {' -> '.join(stage_names)}")
    print(f"Faces: {len(context.faces)}")
    print(f"Aligned: {count_faces_with_alignment(context.faces)}")
    print(f"Assessed: {count_faces_with_assessment(context.faces)}")
    print(f"Total Time: {total_time:.4f}s")

    for stage_name in stage_names:
        elapsed = context.timings[stage_name]
        print(f"  {stage_name}: {elapsed:.4f}s")

    if context.metadata.get("results"):
        print(f"Stage Results: {context.metadata['results']}")

    print()


def run_profile(
    profile: PipelineProfile,
    image,
    builder: PipelineBuilder,
    executor: PipelineExecutor,
    custom_stages: list[str] | None = None,
) -> PipelineContext:
    """Build and execute a single pipeline profile.

    Args:
        profile: Pipeline profile to execute.
        image: Source image.
        builder: Pipeline builder.
        executor: Pipeline executor.
        custom_stages: Optional stage list for ``CUSTOM`` profiles.

    Returns:
        Resulting pipeline context.
    """
    if profile is PipelineProfile.CUSTOM:
        pipeline = builder.build(profile, custom_stages=custom_stages)
    else:
        pipeline = builder.build(profile)

    stage_names = [stage.name for stage in pipeline.stages]
    context = PipelineContext(image=image, profile=profile)
    context = executor.execute(pipeline, context)
    validate_profile_result(profile, context, stage_names)
    print_profile_result(profile, context, stage_names)
    return context


def main() -> None:
    """Run all configured pipeline profiles on an input image."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Pipeline Orchestrator profiles on an image."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--profile",
        choices=[p.value for p in PipelineProfile],
        help="Run a single profile instead of all built-in profiles",
    )
    args = parser.parse_args()

    image = load_image(args.image)
    registry = create_default_registry()
    builder = PipelineBuilder(registry)
    executor = PipelineExecutor()

    configured = load_pipeline_profile_settings().profiles
    print(f"Configured Profiles: {', '.join(configured.keys())}")
    print(f"Registered Stages: {', '.join(registry.names())}")
    print()

    if args.profile:
        profiles = [PipelineProfile(args.profile)]
    else:
        profiles = [p for p in PipelineProfile if p is not PipelineProfile.CUSTOM]

    for profile in profiles:
        run_profile(profile, image, builder, executor)

    if args.profile is None:
        run_profile(
            PipelineProfile.CUSTOM,
            image,
            builder,
            executor,
            custom_stages=["scrfd", "alignment", "assessment"],
        )

    print("All pipeline profiles passed.")


if __name__ == "__main__":
    main()
