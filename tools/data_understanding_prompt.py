import json

from domain.data_understanding import DataUnderstandingArtifact


def build_data_understanding_prompt(
    artifact: DataUnderstandingArtifact,
) -> str:
    """Build a compact evidence payload for the Data Understanding Agent."""

    evidence = {
        "file_name": artifact.file_name,
        "row_count": artifact.row_count,
        "column_count": artifact.column_count,
        "columns": artifact.columns,
        "duplicate_row_count": artifact.duplicate_row_count,
        "missing_value_summary": artifact.missing_value_summary,
        "datatype_summary": artifact.datatype_summary,
        "numeric_columns": artifact.numeric_columns,
        "categorical_columns": artifact.categorical_columns,
        "numeric_like_columns": artifact.numeric_like_columns,
        "categorical_inconsistencies": artifact.categorical_inconsistencies,
        "formatting_issues": artifact.formatting_issues,
        "outlier_summary": artifact.outlier_summary,
        "potential_issues": artifact.potential_issues,
        "analyst_questions": artifact.analyst_questions,
        "numeric_summary": artifact.numeric_summary,
        "numeric_correlations": artifact.numeric_correlations,
        "numeric_target_relationships": artifact.numeric_target_relationships,
    }

    return (
        "Review the following validated data-understanding evidence. "
        "Treat it as observed evidence and do not invent statistics. "
        "Interpret the supplied evidence only; do not calculate new statistics.\n\n"
        "DATA_UNDERSTANDING_ARTIFACT:\n"
        f"{json.dumps(evidence, indent=2, sort_keys=True)}"
    )
