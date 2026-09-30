from typing import Final

import pandas as pd

NULL_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
    }
)


def count_blank_and_null_like_values(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """Count pandas nulls and common textual blank/null markers."""

    counts: dict[str, int] = {}

    for column in dataframe.columns:
        series = dataframe[column]
        missing_mask = series.isna()

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.astype("string").str.strip().str.lower()
            missing_mask = missing_mask | normalized.isin(NULL_MARKERS)

        count = int(missing_mask.sum())

        if count > 0:
            counts[column] = count

    return counts


def count_duplicate_rows(dataframe: pd.DataFrame) -> int:
    """Count rows that are exact duplicates of an earlier row."""

    return int(dataframe.duplicated().sum())


def find_duplicate_values(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    """Return repeated non-null values and their frequencies for one column."""

    if column not in dataframe.columns:
        raise KeyError(f"Column '{column}' does not exist.")

    counts = (
        dataframe[column]
        .dropna()
        .astype("string")
        .str.strip()
        .value_counts()
    )

    return {
        str(value): int(count)
        for value, count in counts.items()
        if count > 1
    }


def detect_categorical_inconsistencies(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, list[str]]:
    """Detect categorical values that differ only by whitespace or case."""

    if column not in dataframe.columns:
        raise KeyError(f"Column '{column}' does not exist.")

    series = dataframe[column].dropna().astype("string")

    groups: dict[str, set[str]] = {}

    for raw_value in series:
        raw = str(raw_value)
        normalized = raw.strip().lower()

        if normalized in NULL_MARKERS:
            continue

        groups.setdefault(normalized, set()).add(raw)

    return {
        normalized: sorted(values)
        for normalized, values in groups.items()
        if len(values) > 1
    }


def detect_iqr_outliers(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, float | int]:
    """Detect IQR-based outliers for a numeric column."""

    if column not in dataframe.columns:
        raise KeyError(f"Column '{column}' does not exist.")

    series = dataframe[column]

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"Column '{column}' must be numeric for IQR outlier detection."
        )

    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "outlier_count": 0,
        }

    q1 = float(clean_series.quantile(0.25))
    q3 = float(clean_series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_mask = (
        (clean_series < lower_bound)
        | (clean_series > upper_bound)
    )

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(outlier_mask.sum()),
    }


def detect_numeric_like_columns(
    dataframe: pd.DataFrame,
    threshold: float = 0.8,
) -> dict[str, dict[str, float | int]]:
    """
    Detect text columns where most non-null values can be interpreted as numeric.
    """

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1.")

    results: dict[str, dict[str, float | int]] = {}

    for column in dataframe.columns:
        series = dataframe[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        non_null = series.dropna().astype("string").str.strip()

        if non_null.empty:
            continue

        numeric_values = pd.to_numeric(
            non_null,
            errors="coerce",
        )

        numeric_count = int(numeric_values.notna().sum())
        total_count = len(non_null)
        proportion = numeric_count / total_count

        if proportion >= threshold:
            results[column] = {
                "numeric_like_count": numeric_count,
                "non_null_count": total_count,
                "numeric_like_proportion": proportion,
            }

    return results


def detect_string_formatting_issues(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    """
    Detect leading/trailing whitespace and inconsistent casing
    in string values.

    Returns counts of affected rows rather than modifying values.
    """

    if column not in dataframe.columns:
        raise KeyError(f"Column '{column}' does not exist.")

    series = dataframe[column]

    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        raise TypeError(
            f"Column '{column}' must contain string-like values."
        )

    values = series.dropna().astype("string")

    whitespace_count = int(
        (values != values.str.strip()).sum()
    )

    non_null_values = values[values.str.strip() != ""]

    if non_null_values.empty:
        casing_count = 0
    else:
        normalized = non_null_values.str.strip().str.lower()
        casing_groups = (
            pd.DataFrame(
                {
                    "raw": non_null_values,
                    "normalized": normalized,
                }
            )
            .groupby("normalized")["raw"]
            .nunique()
        )

        inconsistent_groups = casing_groups[casing_groups > 1]
        casing_count = int(
            sum(
                int(
                    (
                        normalized == normalized_value
                    ).sum()
                )
                for normalized_value in inconsistent_groups.index
            )
        )

    return {
        "leading_or_trailing_whitespace_count": whitespace_count,
        "case_inconsistency_count": casing_count,
    }
