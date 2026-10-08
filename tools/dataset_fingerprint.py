"""Deterministic fingerprints for persisted dataset files."""

from hashlib import sha256
from pathlib import Path


def fingerprint_dataset_file(file_path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file's exact bytes."""

    path = Path(file_path)
    digest = sha256()

    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()
