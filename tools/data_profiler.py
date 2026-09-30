from pathlib import Path

from domain.data_understanding import DataUnderstandingArtifact
from tools.data_loader import load_dataset
from tools.data_quality import (
    count_blank_and_null_like_values,
    count_duplicate_rows,
    detect_categorical_inconsistencies,
    detect_iqr_outliers,
    detect_numeric_like_columns,
    detect_string_formatting_issues,
    find_duplicate_values,
)


def profile_dataset(file_path: str) -> DataUnderstandingArtifact:
    """Build a deterministic data-understanding artifact."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset '{file_path}' was not found.")

    dataframe = load_dataset(file_path)

    missing_value_summary = count_blank_and_null_like_values(dataframe)
    duplicate_row_count = count_duplicate_rows(dataframe)

    datatype_summary = {
        column: str(dtype)
        for column, dtype in dataframe.dtypes.items()
    }

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_like_columns = detect_numeric_like_columns(dataframe)

    duplicate_value_summary: dict[str, dict[str, int]] = {}

    for column in dataframe.columns:
        duplicates = find_duplicate_values(dataframe, column)

        if duplicates:
            duplicate_value_summary[column] = duplicates

    categorical_inconsistencies: dict[str, dict[str, list[str]]] = {}

    for column in categorical_columns:
        inconsistencies = detect_categorical_inconsistencies(
            dataframe,
            column,
        )

        if inconsistencies:
            categorical_inconsistencies[column] = inconsistencies

    formatting_issues: dict[str, dict[str, int]] = {}

    for column in categorical_columns:
        issues = detect_string_formatting_issues(
            dataframe,
            column,
        )

        if any(issues.values()):
            formatting_issues[column] = issues

    outlier_summary: dict[str, dict[str, float | int]] = {}

    for column in numeric_columns:
        outliers = detect_iqr_outliers(
            dataframe,
            column,
        )

        if outliers["outlier_count"] > 0:
            outlier_summary[column] = outliers

    potential_issues: list[str] = []
    analyst_questions: list[str] = []

    if duplicate_row_count > 0:
        potential_issues.append(
            f"{duplicate_row_count} exact duplicate row(s) detected."
        )
        analyst_questions.append(
            "Should exact duplicate rows be removed or retained?"
        )

    if missing_value_summary:
        potential_issues.append(
            f"Missing or null-like values detected in "
            f"{len(missing_value_summary)} column(s)."
        )
        analyst_questions.append(
            "How should missing or null-like values be treated?"
        )

    if numeric_like_columns:
        potential_issues.append(
            f"{len(numeric_like_columns)} text column(s) appear to contain "
            "predominantly numeric values."
        )
        analyst_questions.append(
            "Should numeric-like text columns be converted to numeric types?"
        )

    if categorical_inconsistencies:
        potential_issues.append(
            f"Categorical inconsistencies detected in "
            f"{len(categorical_inconsistencies)} column(s)."
        )
        analyst_questions.append(
            "Which categorical normalization rules should be approved?"
        )

    if formatting_issues:
        potential_issues.append(
            f"String formatting issues detected in "
            f"{len(formatting_issues)} column(s)."
        )

    if outlier_summary:
        potential_issues.append(
            f"IQR-based outliers detected in "
            f"{len(outlier_summary)} numeric column(s)."
        )
        analyst_questions.append(
            "Which statistical outliers represent valid business observations?"
        )

    return DataUnderstandingArtifact(
        file_name=path.name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=dataframe.columns.tolist(),
        duplicate_row_count=duplicate_row_count,
        duplicate_value_summary=duplicate_value_summary,
        missing_value_summary=missing_value_summary,
        datatype_summary=datatype_summary,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_like_columns=numeric_like_columns,
        categorical_inconsistencies=categorical_inconsistencies,
        formatting_issues=formatting_issues,
        outlier_summary=outlier_summary,
        potential_issues=potential_issues,
        analyst_questions=analyst_questions,
    )
