import pandas as pd

from domain.data_understanding import DataUnderstandingArtifact
from tools.data_understanding_eda import add_eda_evidence


def build_artifact() -> DataUnderstandingArtifact:
    return DataUnderstandingArtifact(
        file_name="customers.csv",
        row_count=5,
        column_count=4,
        columns=[
            "tenure",
            "revenue",
            "segment",
            "churn",
        ],
        duplicate_row_count=0,
    )


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5],
            "revenue": [10, 20, 30, 40, 50],
            "segment": [
                "Consumer",
                "Consumer",
                "Business",
                "Business",
                "Consumer",
            ],
            "churn": [1, 1, 0, 0, 0],
        }
    )


def test_adds_general_eda_evidence():
    artifact = build_artifact()
    dataframe = build_dataframe()

    result = add_eda_evidence(
        artifact,
        dataframe,
    )

    assert result.numeric_summary["revenue"]["mean"] == 30.0
    assert result.categorical_summary["segment"]["unique_count"] == 2
    assert result.numeric_correlations["tenure"]["revenue"] == 1.0


def test_adds_target_eda_evidence_when_target_is_supplied():
    artifact = build_artifact()
    dataframe = build_dataframe()

    result = add_eda_evidence(
        artifact,
        dataframe,
        target_column="churn",
    )

    assert "tenure" in result.numeric_target_relationships
    assert "revenue" in result.numeric_target_relationships
    assert "segment" in result.categorical_target_relationships


def test_does_not_add_target_evidence_without_target():
    artifact = build_artifact()
    dataframe = build_dataframe()

    result = add_eda_evidence(
        artifact,
        dataframe,
    )

    assert result.numeric_target_relationships == {}
    assert result.categorical_target_relationships == {}


def test_original_artifact_is_not_mutated():
    artifact = build_artifact()
    dataframe = build_dataframe()

    result = add_eda_evidence(
        artifact,
        dataframe,
    )

    assert artifact.numeric_summary == {}
    assert result.numeric_summary != {}
