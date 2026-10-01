import pandas as pd

from domain.data_understanding import DataUnderstandingArtifact
from workflow.data_understanding_stage import (
    build_data_understanding_evidence,
    prepare_data_understanding_stage,
)


def build_artifact() -> DataUnderstandingArtifact:
    return DataUnderstandingArtifact(
        file_name="customers.csv",
        row_count=4,
        column_count=3,
        columns=[
            "customer_id",
            "revenue",
            "churn",
        ],
        duplicate_row_count=0,
        datatype_summary={
            "customer_id": "object",
            "revenue": "float64",
            "churn": "int64",
        },
        numeric_columns=[
            "revenue",
            "churn",
        ],
        categorical_columns=[
            "customer_id",
        ],
    )


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "revenue": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "churn": [
                0,
                1,
                0,
                1,
            ],
        }
    )


def test_stage_accepts_validated_artifact():
    artifact = build_artifact()
    dataframe = build_dataframe()

    result = prepare_data_understanding_stage(
        artifact,
        dataframe,
    )

    assert isinstance(result, str)
    assert "revenue" in result


def test_stage_includes_target_relationships_when_target_is_supplied():
    artifact = build_artifact()
    dataframe = build_dataframe()

    result = prepare_data_understanding_stage(
        artifact,
        dataframe,
        target_column="churn",
    )

    assert isinstance(result, str)
    assert "churn" in result


def test_stage_does_not_modify_artifact():
    artifact = build_artifact()
    dataframe = build_dataframe()

    before = artifact.model_dump()

    prepare_data_understanding_stage(
        artifact,
        dataframe,
    )

    assert artifact.model_dump() == before

def test_build_data_understanding_evidence_returns_enriched_artifact():
    artifact = build_artifact()
    dataframe = build_dataframe()

    result = build_data_understanding_evidence(
        artifact,
        dataframe,
        target_column="churn",
    )

    assert isinstance(result, DataUnderstandingArtifact)
    assert result.file_name == artifact.file_name
    assert result.row_count == artifact.row_count
    assert result.numeric_summary
    assert result.numeric_correlations
    assert result.numeric_target_relationships