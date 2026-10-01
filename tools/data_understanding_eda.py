import pandas as pd

from domain.data_understanding import DataUnderstandingArtifact
from tools.eda_analysis import (
    calculate_numeric_correlations,
    summarize_categorical_columns,
    summarize_numeric_columns,
)
from tools.eda_target_analysis import (
    summarize_categorical_target_relationships,
    summarize_numeric_target_relationships,
)


def add_eda_evidence(
    artifact: DataUnderstandingArtifact,
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> DataUnderstandingArtifact:
    """Add deterministic EDA evidence to an existing data artifact.

    The original artifact is not mutated. A new validated artifact containing
    the calculated EDA evidence is returned.
    """

    updates: dict[str, object] = {
        "numeric_summary": summarize_numeric_columns(dataframe),
        "categorical_summary": summarize_categorical_columns(dataframe),
        "numeric_correlations": calculate_numeric_correlations(dataframe),
    }

    if target_column is not None:
        updates["numeric_target_relationships"] = (
            summarize_numeric_target_relationships(
                dataframe,
                target_column,
            )
        )
        updates["categorical_target_relationships"] = (
            summarize_categorical_target_relationships(
                dataframe,
                target_column,
            )
        )

    return artifact.model_copy(update=updates)
