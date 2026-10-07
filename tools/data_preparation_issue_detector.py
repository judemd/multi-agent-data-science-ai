import pandas as pd

from domain.data_preparation_issue import DataPreparationIssue
from tools.data_quality import (
    count_blank_and_null_like_values,
    count_duplicate_rows,
    detect_categorical_inconsistencies,
    detect_numeric_like_columns,
    detect_potential_identifier_columns,
)


def detect_missing_value_issues(
    dataframe: pd.DataFrame,
) -> list[DataPreparationIssue]:
    """Detect missing and null-like values with controlled treatments."""

    issues: list[DataPreparationIssue] = []

    missing_value_summary = count_blank_and_null_like_values(
        dataframe
    )

    for column, missing_count in missing_value_summary.items():
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            allowed_treatments = [
                "median",
                "mean",
                "mode",
                "drop_rows",
                "retain",
            ]
        else:
            allowed_treatments = [
                "mode",
                "constant",
                "drop_rows",
                "retain",
            ]

        issues.append(
            DataPreparationIssue(
                issue_type="missing_values",
                column=column,
                evidence=(
                    f"Column '{column}' contains "
                    f"{missing_count} missing or null-like values."
                ),
                allowed_treatments=allowed_treatments,
            )
        )

    return issues


def detect_categorical_inconsistency_issues(
    dataframe: pd.DataFrame,
) -> list[DataPreparationIssue]:
    """Detect case/whitespace variants in categorical columns."""

    issues: list[DataPreparationIssue] = []

    for column in dataframe.columns:
        series = dataframe[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue

        inconsistencies = detect_categorical_inconsistencies(
            dataframe,
            column,
        )

        if not inconsistencies:
            continue

        evidence_parts = [
            f"{normalized}: {variants}"
            for normalized, variants in inconsistencies.items()
        ]

        issues.append(
            DataPreparationIssue(
                issue_type="categorical_inconsistency",
                column=column,
                evidence=(
                    f"Column '{column}' contains categorical "
                    f"case or whitespace variants: "
                    f"{'; '.join(evidence_parts)}."
                ),
                allowed_treatments=[
                    "normalize_categories",
                    "retain",
                ],
            )
        )

    return issues



def detect_numeric_conversion_issues(
    dataframe: pd.DataFrame,
) -> list[DataPreparationIssue]:
    """Detect text columns that are predominantly numeric-like."""

    issues: list[DataPreparationIssue] = []

    numeric_like_columns = detect_numeric_like_columns(dataframe)

    for column, evidence in numeric_like_columns.items():
        numeric_count = int(evidence["numeric_like_count"])
        non_null_count = int(evidence["non_null_count"])
        proportion = float(evidence["numeric_like_proportion"])

        issues.append(
            DataPreparationIssue(
                issue_type="numeric_conversion",
                column=column,
                evidence=(
                    f"Column '{column}' is stored as text but "
                    f"{numeric_count} of {non_null_count} non-null values "
                    f"are numeric-like "
                    f"({proportion:.1%})."
                ),
                allowed_treatments=[
                    "convert_numeric",
                    "retain",
                ],
            )
        )

    return issues



def detect_exact_duplicate_issues(
    dataframe: pd.DataFrame,
) -> list[DataPreparationIssue]:
    """Detect exact duplicate rows."""

    duplicate_count = count_duplicate_rows(dataframe)

    if duplicate_count == 0:
        return []

    return [
        DataPreparationIssue(
            issue_type="exact_duplicates",
            column=None,
            evidence=(
                f"Dataset contains {duplicate_count} exact duplicate "
                f"row(s) after the first occurrence."
            ),
            allowed_treatments=[
                "remove_duplicates",
                "retain",
            ],
        )
    ]



def detect_identifier_issues(
    dataframe: pd.DataFrame,
) -> list[DataPreparationIssue]:
    """Surface potential identifier columns for explicit human review."""

    identifier_columns = detect_potential_identifier_columns(dataframe)

    return [
        DataPreparationIssue(
            issue_type="identifier",
            column=column,
            evidence=(
                f"Column '{column}' is a high-cardinality string column "
                "with at least 95% unique non-null values and may represent "
                "a business identifier."
            ),
            allowed_treatments=[
                "exclude_feature",
                "retain",
                "investigate",
            ],
            requires_explicit_human_decision=True,
        )
        for column in identifier_columns
    ]
