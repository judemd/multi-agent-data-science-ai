from typing import Any

import pandas as pd


def summarize_numeric_target_relationships(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, float]:
    """Calculate Pearson correlations between numeric features and a numeric target."""

    if target_column not in dataframe.columns:
        raise KeyError(f"Column '{target_column}' does not exist.")

    if not pd.api.types.is_numeric_dtype(dataframe[target_column]):
        raise TypeError(
            f"Target column '{target_column}' must be numeric."
        )

    numeric = dataframe.select_dtypes(include="number")

    if target_column not in numeric.columns:
        return {}

    correlations = numeric.corr(method="pearson")[target_column]

    return {
        column: float(value)
        for column, value in correlations.items()
        if column != target_column and pd.notna(value)
    }


def summarize_categorical_target_relationships(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, dict[str, Any]]:
    """
    Summarize target rates across categorical feature values.

    The target must be binary and represented numerically.
    """

    if target_column not in dataframe.columns:
        raise KeyError(f"Column '{target_column}' does not exist.")

    target = dataframe[target_column]

    if not pd.api.types.is_numeric_dtype(target):
        raise TypeError(
            f"Target column '{target_column}' must be numeric."
        )

    unique_targets = set(target.dropna().unique())

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target_column}' must contain only 0 and 1."
        )

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category"]
    ).columns

    results: dict[str, dict[str, Any]] = {}

    for column in categorical_columns:
        grouped = (
            dataframe.groupby(
                column,
                dropna=False,
            )[target_column]
            .agg(
                count="count",
                target_rate="mean",
            )
        )

        results[column] = {
            str(value): {
                "count": int(row["count"]),
                "target_rate": float(row["target_rate"]),
            }
            for value, row in grouped.iterrows()
        }

    return results
