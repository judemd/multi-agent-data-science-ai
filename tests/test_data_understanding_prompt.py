from domain.data_understanding import DataUnderstandingArtifact
from tools.data_understanding_prompt import (
    build_data_understanding_prompt,
)


def test_builds_prompt_from_validated_artifact():
    artifact = DataUnderstandingArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
        columns=[
            "customer_id",
            "revenue",
            "segment",
        ],
        duplicate_row_count=2,
        missing_value_summary={
            "revenue": 4,
        },
        datatype_summary={
            "customer_id": "object",
            "revenue": "float64",
            "segment": "object",
        },
        numeric_columns=[
            "revenue",
        ],
        categorical_columns=[
            "customer_id",
            "segment",
        ],
        potential_issues=[
            "Missing revenue values detected.",
        ],
        analyst_questions=[
            "How should missing revenue values be treated?",
        ],
    )

    prompt = build_data_understanding_prompt(artifact)

    assert "DATA_UNDERSTANDING_ARTIFACT" in prompt
    assert "customers.csv" in prompt
    assert '"row_count": 100' in prompt
    assert "Missing revenue values detected." in prompt


def test_prompt_contains_no_instruction_to_modify_data():
    artifact = DataUnderstandingArtifact(
        file_name="customers.csv",
        row_count=10,
        column_count=1,
        columns=["customer_id"],
        duplicate_row_count=0,
    )

    prompt = build_data_understanding_prompt(artifact)

    assert "DATA_UNDERSTANDING_ARTIFACT" in prompt
    assert "customers.csv" in prompt
