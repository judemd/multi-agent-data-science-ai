"""Basic dataset readiness checks before binary classification training."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tools.data_quality import NULL_MARKERS


@dataclass
class ModelingSanityResult:
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_modeling_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
) -> ModelingSanityResult:
    """Inspect modeling readiness without modifying the dataset."""

    errors: list[str] = []
    warnings: list[str] = []

    if dataframe.empty:
        errors.append("Dataset is empty.")

    if not dataframe.columns.is_unique:
        errors.append("Dataset contains duplicate column names.")
        return ModelingSanityResult("BLOCK", errors, warnings)

    if target_column not in dataframe.columns:
        errors.append(f"Target column '{target_column}' is missing.")
        return ModelingSanityResult("BLOCK", errors, warnings)

    target = dataframe[target_column]

    target_blank = target.map(
        lambda value: (
            isinstance(value, str)
            and value.strip().lower() in NULL_MARKERS
        )
    )

    invalid_target = target.isna() | target_blank

    if invalid_target.any():
        errors.append(
            f"Target contains {int(invalid_target.sum())} missing, "
            "blank, or textual-null values."
        )

    valid_target = target.loc[~invalid_target]
    class_counts = valid_target.value_counts()

    if len(class_counts) != 2:
        errors.append(
            "POC supports binary classification with exactly two target classes."
        )
    elif class_counts.min() < 2:
        errors.append(
            "Each target class must contain at least two observations."
        )

    features = dataframe.drop(columns=[target_column])

    if features.shape[1] == 0:
        errors.append("No feature columns are available.")

    for column in features.columns:
        series = features[column]

        if pd.api.types.is_numeric_dtype(series):
            numeric_values = pd.to_numeric(series, errors="coerce")

            if np.isinf(numeric_values.to_numpy(dtype=float)).any():
                errors.append(
                    f"Feature '{column}' contains infinite numeric values."
                )

        supported = (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

        if not supported:
            errors.append(
                f"Feature '{column}' has unsupported dtype '{series.dtype}'."
            )
            continue

        missing_count = int(series.isna().sum())

        blank_count = int(
            series.map(
                lambda value: (
                    isinstance(value, str)
                    and value.strip().lower() in NULL_MARKERS
                )
            ).sum()
        )

        if missing_count or blank_count:
            warnings.append(
                f"Feature '{column}': {missing_count} missing values, "
                f"{blank_count} blank or textual-null values."
            )

    status = (
        "BLOCK" if errors
        else "WARNING" if warnings
        else "PASS"
    )

    return ModelingSanityResult(
        status=status,
        errors=errors,
        warnings=warnings,
    )