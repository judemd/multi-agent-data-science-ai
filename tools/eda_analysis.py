from typing import Any

import pandas as pd


def summarize_numeric_columns(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Return deterministic descriptive statistics for numeric columns."""

    numeric = dataframe.select_dtypes(include="number")

    if numeric.empty:
        return {}

    summary = numeric.describe().transpose()

    results: dict[str, dict[str, float | int]] = {}

    for column, row in summary.iterrows():
        results[column] = {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "min": float(row["min"]),
            "25%": float(row["25%"]),
            "50%": float(row["50%"]),
            "75%": float(row["75%"]),
            "max": float(row["max"]),
        }

    return results


def summarize_categorical_columns(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Return deterministic frequency summaries for categorical columns."""

    categorical = dataframe.select_dtypes(
        include=["object", "category"]
    )

    results: dict[str, dict[str, Any]] = {}

    for column in categorical.columns:
        counts = dataframe[column].value_counts(
            dropna=False,
        )

        results[column] = {
            "unique_count": int(dataframe[column].nunique(dropna=True)),
            "top_values": {
                str(value): int(count)
                for value, count in counts.head(10).items()
            },
        }

    return results


def calculate_numeric_correlations(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Calculate Pearson correlations between numeric columns."""

    numeric = dataframe.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        return {}

    correlation_matrix = numeric.corr(method="pearson")

    results: dict[str, dict[str, float]] = {}

    for column in correlation_matrix.columns:
        results[column] = {
            other_column: float(value)
            for other_column, value in correlation_matrix[column].items()
            if other_column != column
        }

    return results
