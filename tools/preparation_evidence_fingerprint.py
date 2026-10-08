"""Deterministic fingerprints for data preparation evidence."""

from hashlib import sha256

from domain.data_preparation import DataPreparationArtifact


def fingerprint_preparation_evidence(
    artifact: DataPreparationArtifact,
) -> str:
    """Return a SHA-256 fingerprint of the complete preparation artifact."""

    canonical_json = artifact.model_dump_json(
        exclude_none=False,
    )

    # Re-serialize with sorted object keys so dictionary insertion order
    # cannot affect the fingerprint.
    import json

    canonical_payload = json.dumps(
        json.loads(canonical_json),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    return sha256(canonical_payload.encode("utf-8")).hexdigest()
