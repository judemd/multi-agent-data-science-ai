from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a dataset format is not supported."""


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a supported CSV or Excel dataset into a pandas DataFrame."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset '{file_path}' was not found.")

    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise UnsupportedFileTypeError(
        f"Unsupported dataset format '{extension}'. "
        f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )
