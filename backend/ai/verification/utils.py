"""
Utility helpers for the verification engine.
"""

from backend.app.domain.face import Face
from backend.app.domain.search import SearchResult, SearchResults
from backend.app.domain.verification import resolve_is_verified

__all__ = [
    "resolve_is_verified",
    "validate_face_embedding",
    "validate_search_results",
    "validate_similarity_score",
    "validate_threshold",
    "validate_verification_input",
]

_SIMILARITY_MIN = -1.0
_SIMILARITY_MAX = 1.0
_THRESHOLD_MIN = 0.0
_THRESHOLD_MAX = 1.0
_SCORE_TOLERANCE = 1e-5


def validate_threshold(threshold: float) -> None:
    """Validate a configured verification similarity threshold.

    Args:
        threshold: Similarity threshold expected in ``[0, 1]``.

    Raises:
        ValueError: If the threshold is outside the supported range.
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError(
            f"Verification threshold must be numeric, got {type(threshold)!r}."
        )

    if not _THRESHOLD_MIN <= float(threshold) <= _THRESHOLD_MAX:
        raise ValueError(
            "Verification threshold must be within "
            f"[{_THRESHOLD_MIN}, {_THRESHOLD_MAX}], got {threshold}."
        )


def validate_similarity_score(score: float) -> None:
    """Validate a cosine similarity score from search results.

    Args:
        score: Inner-product similarity score for unit-normalized embeddings.

    Raises:
        ValueError: If the score is outside the supported cosine range.
    """
    if not isinstance(score, (int, float)):
        raise ValueError(
            f"Similarity score must be numeric, got {type(score)!r}."
        )

    if not (
        (_SIMILARITY_MIN - _SCORE_TOLERANCE)
        <= float(score)
        <= (_SIMILARITY_MAX + _SCORE_TOLERANCE)
    ):
        raise ValueError(
            "Similarity score must be within "
            f"[{_SIMILARITY_MIN}, {_SIMILARITY_MAX}], got {score}."
        )


def validate_face_embedding(face: Face, *, index: int | None = None) -> None:
    """Validate that a face has embedding data required for verification.

    Args:
        face: Probe face to validate.
        index: Optional face index for error messages.

    Raises:
        ValueError: If embedding data is missing or invalid.
    """
    prefix = f"Face at index {index}" if index is not None else "Face"

    if not isinstance(face, Face):
        raise ValueError(f"Expected Face, got {type(face)!r}.")

    if face.embedding is None:
        raise ValueError(f"{prefix} must have embedding data before verification.")

    if face.embedding.vector is None:
        raise ValueError(
            f"{prefix} must have an embedding vector before verification."
        )

    if not face.embedding.normalized:
        raise ValueError(
            f"{prefix} must have a normalized embedding before verification."
        )


def validate_search_results(search_results: SearchResults) -> None:
    """Validate search results used for verification.

    Args:
        search_results: Ranked gallery matches for one probe face.

    Raises:
        ValueError: If search results are invalid.
    """
    if not isinstance(search_results, SearchResults):
        raise ValueError(
            "Expected SearchResults, "
            f"got {type(search_results)!r}."
        )

    if not search_results.provider:
        raise ValueError("SearchResults.provider must be a non-empty string.")

    seen_identities: set[str] = set()
    seen_ranks: set[int] = set()

    for index, candidate in enumerate(search_results.results):
        if not isinstance(candidate, SearchResult):
            raise ValueError(
                "Expected SearchResult at index "
                f"{index}, got {type(candidate)!r}."
            )

        if not candidate.identity_id:
            raise ValueError(
                f"SearchResult at index {index} must have a non-empty identity_id."
            )

        if candidate.identity_id in seen_identities:
            raise ValueError(
                "Duplicate search candidate identity_id "
                f"{candidate.identity_id!r}."
            )
        seen_identities.add(candidate.identity_id)

        if candidate.rank in seen_ranks:
            raise ValueError(f"Duplicate search candidate rank {candidate.rank}.")
        seen_ranks.add(candidate.rank)

        validate_similarity_score(candidate.score)

    if search_results.results and 1 not in seen_ranks:
        raise ValueError(
            "Search results must include a rank-1 candidate when matches exist."
        )


def validate_verification_input(
    faces: list[Face],
    search_results: list[SearchResults],
) -> None:
    """Validate inputs to the verification pipeline.

    Args:
        faces: Probe faces to verify.
        search_results: Search outputs aligned with ``faces``.

    Raises:
        ValueError: If inputs are invalid or misaligned.
    """
    if not isinstance(faces, list):
        raise ValueError("Faces must be provided as a list.")

    if not isinstance(search_results, list):
        raise ValueError("Search results must be provided as a list.")

    if len(faces) != len(search_results):
        raise ValueError(
            "Faces and search results must have the same length: "
            f"faces={len(faces)} search_results={len(search_results)}."
        )

    for index, face in enumerate(faces):
        validate_face_embedding(face, index=index)

    for index, results in enumerate(search_results):
        if not isinstance(results, SearchResults):
            raise ValueError(
                "Expected SearchResults at index "
                f"{index}, got {type(results)!r}."
            )
        validate_search_results(results)
