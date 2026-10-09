"""Deterministic fingerprinting of Finalization handoff evidence."""

import hashlib

from domain.finalization import FinalizationArtifact


def fingerprint_finalization_evidence(
    artifact: FinalizationArtifact,
) -> str:
    """Return a SHA-256 digest of the complete handoff artifact."""

    canonical_json = artifact.model_dump_json(
        exclude_none=False,
        round_trip=True,
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()