from io import BytesIO
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a dataset format is not supported."""


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a supported CSV or Excel dataset from a filesystem path."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset '{file_path}' was not found.")

    return _load_by_extension(
        path.suffix.lower(),
        path,
    )


def load_uploaded_dataset(
    file_name: str,
    file_content: bytes,
) -> pd.DataFrame:
    """Load a supported CSV or Excel dataset from uploaded file content.

    The function uses the filename extension to determine the appropriate
    pandas reader while keeping file handling outside the Streamlit UI.
    """

    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported dataset format '{extension}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    buffer = BytesIO(file_content)

    if extension == ".csv":
        return pd.read_csv(buffer)

    return pd.read_excel(buffer)


def _load_by_extension(
    extension: str,
    path: Path,
) -> pd.DataFrame:
    """Load a filesystem dataset using its validated extension."""

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise UnsupportedFileTypeError(
        f"Unsupported dataset format '{extension}'. "
        f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )
