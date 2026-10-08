import pandas as pd

from domain.data_preparation import DataPreparationArtifact
from domain.data_validation_rule import DataValidationRule
from tools.data_quality import (
    detect_candidate_date_columns,
    detect_numeric_like_columns,
    detect_potential_identifier_columns,
)
from tools.data_validation_issue_detector import detect_validation_rule_issues
from tools.data_preparation_issue_detector import (
    detect_categorical_inconsistency_issues,
    detect_date_conversion_issues,
    detect_exact_duplicate_issues,
    detect_fake_null_issues,
    detect_feature_variability_issues,
    detect_formatting_issues,
    detect_identifier_issues,
    detect_missing_value_issues,
    detect_numeric_conversion_issues,
    detect_outlier_issues,
)


def build_data_preparation_evidence(
    dataframe: pd.DataFrame,
    file_name: str,
    approved_validation_rules: list[DataValidationRule] | None = None,
) -> DataPreparationArtifact:
    """Build deterministic preparation evidence without modifying the DataFrame."""

    missing_value_summary = {
        column: int(dataframe[column].isna().sum())
        for column in dataframe.columns
        if int(dataframe[column].isna().sum()) > 0
    }

    numeric_like_columns = detect_numeric_like_columns(dataframe)

    categorical_inconsistencies: dict[str, dict[str, list[str]]] = {}

    formatting_issues: dict[str, dict[str, int]] = {}

    constant_columns: list[str] = []

    candidate_date_columns = detect_candidate_date_columns(dataframe)

    potential_identifier_columns = detect_potential_identifier_columns(dataframe)

    outlier_summary: dict[str, dict[str, float | int]] = {}

    for column in dataframe.columns:
        series = dataframe[column]

        if series.nunique(dropna=True) <= 1:
            constant_columns.append(column)

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()

            if not non_null.empty:
                q1 = float(non_null.quantile(0.25))
                q3 = float(non_null.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                outlier_count = int(
                    ((non_null < lower) | (non_null > upper)).sum()
                )

                outlier_summary[column] = {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "outlier_count": outlier_count,
                }

        if pd.api.types.is_string_dtype(series):
            non_null = series.dropna().astype(str)

            if non_null.empty:
                continue

            numeric_conversion = pd.to_numeric(
                non_null,
                errors="coerce",
            )

            numeric_ratio = float(numeric_conversion.notna().mean())

            if numeric_ratio >= 0.8:
                numeric_like_columns[column] = {
                    "non_null_count": len(non_null),
                    "numeric_like_count": int(
                        numeric_conversion.notna().sum()
                    ),
                    "numeric_like_ratio": numeric_ratio,
                }

            stripped = non_null.str.strip()

            whitespace_count = int(
                (non_null != stripped).sum()
            )

            case_variants: dict[str, list[str]] = {}

            for value in non_null.unique():
                normalized = value.strip().casefold()
                case_variants.setdefault(normalized, []).append(value)

            variants = {
                key: sorted(set(values))
                for key, values in case_variants.items()
                if len(set(values)) > 1
            }

            if variants:
                categorical_inconsistencies[column] = variants

            if whitespace_count:
                formatting_issues[column] = {
                    "leading_or_trailing_whitespace": whitespace_count,
                }



    preparation_questions: list[str] = []

    if missing_value_summary:
        preparation_questions.append(
            "How should missing values be treated in affected columns?"
        )

    if numeric_like_columns:
        preparation_questions.append(
            "Should numeric-like text columns be converted to numeric types?"
        )

    if categorical_inconsistencies:
        preparation_questions.append(
            "Which categorical spelling, case, or whitespace variants are "
            "business-equivalent?"
        )

    if formatting_issues:
        preparation_questions.append(
            "Should observed formatting inconsistencies be standardized?"
        )

    if candidate_date_columns:
        preparation_questions.append(
            "Which candidate date columns should be converted to date types?"
        )

    if potential_identifier_columns:
        preparation_questions.append(
            "Which high-cardinality columns are genuine business identifiers?"
        )

    if outlier_summary:
        preparation_questions.append(
            "Which observed outliers, if any, require business investigation?"
        )

    duplicate_row_count = int(dataframe.duplicated().sum())

    if duplicate_row_count:
        preparation_questions.append(
            "Do exact duplicate rows represent duplicate records or valid "
            "repeated observations?"
        )

    issues = [
        *detect_missing_value_issues(dataframe),
        *detect_fake_null_issues(dataframe),
        *detect_categorical_inconsistency_issues(dataframe),
        *detect_date_conversion_issues(dataframe),
        *detect_numeric_conversion_issues(dataframe),
        *detect_outlier_issues(dataframe),
        *detect_formatting_issues(dataframe),
        *detect_feature_variability_issues(dataframe),
        *detect_exact_duplicate_issues(dataframe),
        *detect_identifier_issues(dataframe),
        *detect_validation_rule_issues(
            dataframe,
            approved_validation_rules or [],
        ),
    ]

    seen_issue_ids: set[str] = set()

    for issue in issues:
        if issue.issue_id is None:
            scope = issue.column if issue.column is not None else "dataset"
            issue.issue_id = f"{issue.issue_type}:{scope}"

        if issue.issue_id in seen_issue_ids:
            raise ValueError(
                f"Duplicate data preparation issue ID: '{issue.issue_id}'."
            )

        seen_issue_ids.add(issue.issue_id)

    return DataPreparationArtifact(
        file_name=file_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=list(dataframe.columns),
        datatype_summary={
            column: str(dataframe[column].dtype)
            for column in dataframe.columns
        },
        missing_value_summary=missing_value_summary,
        numeric_like_columns=numeric_like_columns,
        categorical_inconsistencies=categorical_inconsistencies,
        formatting_issues=formatting_issues,
        outlier_summary=outlier_summary,
        constant_columns=constant_columns,
        candidate_date_columns=candidate_date_columns,
        potential_identifier_columns=potential_identifier_columns,
        duplicate_row_count=duplicate_row_count,
        preparation_questions=preparation_questions,
        issues=issues,
    )
