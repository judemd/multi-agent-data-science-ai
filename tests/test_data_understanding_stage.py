from domain.data_understanding import DataUnderstandingArtifact
from workflow.data_understanding_stage import (
    prepare_data_understanding_stage,
)


def test_stage_accepts_validated_artifact():
    artifact = DataUnderstandingArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=2,
        columns=["customer_id", "revenue"],
        duplicate_row_count=0,
        datatype_summary={
            "customer_id": "object",
            "revenue": "float64",
        },
        numeric_columns=["revenue"],
        categorical_columns=["customer_id"],
    )

    result = prepare_data_understanding_stage(artifact)

    assert "DATA_UNDERSTANDING_ARTIFACT" in result
    assert "customers.csv" in result
    assert '"row_count": 100' in result


def test_stage_does_not_modify_artifact():
    artifact = DataUnderstandingArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=2,
        columns=["customer_id", "revenue"],
        duplicate_row_count=0,
    )

    before = artifact.model_dump()

    prepare_data_understanding_stage(artifact)

    after = artifact.model_dump()

    assert after == before
