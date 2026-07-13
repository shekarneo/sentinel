"""
Pipeline executor.

Runs pipeline stages sequentially and records timing and errors.
"""

import logging
import time

from backend.app.pipeline.context import PipelineContext
from backend.app.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes pipeline stages in order with graceful failure handling."""

    def execute(
        self,
        pipeline: Pipeline,
        context: PipelineContext,
        *,
        stop_on_error: bool = True,
    ) -> PipelineContext:
        """Execute all stages in the pipeline.

        Args:
            pipeline: Resolved pipeline to execute.
            context: Initial pipeline context. ``context.profile`` is updated
                to match the pipeline profile.
            stop_on_error: When ``True``, halt execution after the first
                stage failure. The failing stage is recorded in
                ``context.errors``.

        Returns:
            Updated pipeline context with faces, metadata, errors, and timings.
        """
        context.profile = pipeline.profile

        for stage in pipeline.stages:
            started = time.perf_counter()
            try:
                context = stage.execute(context)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                context.timings[stage.name] = elapsed
                context.errors.append(
                    {
                        "stage": stage.name,
                        "message": str(exc),
                    }
                )
                logger.exception(
                    "Pipeline stage '%s' failed after %.4fs.",
                    stage.name,
                    elapsed,
                )
                if stop_on_error:
                    break
                continue

            context.timings[stage.name] = time.perf_counter() - started
            logger.debug(
                "Pipeline stage '%s' completed in %.4fs.",
                stage.name,
                context.timings[stage.name],
            )

        return context
