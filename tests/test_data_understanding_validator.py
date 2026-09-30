import pytest
from pydantic import ValidationError

from tools.data_understanding_validator import validate_data_understanding


def test_valid_data_understanding_artifact_is_accepted():
    artifact = validate_data_understanding(
        {
            "file_name": "customer_churn_poc.csv",
            "row_count": 100,
            "column_count": 10,
            "columns": [
                "customer_id",
                "tenure_months",
            ],
            "duplicate_row_count": 2,
            "duplicate_value_summary": {},
            "missing_value_summary": {
                "tenure_months": 3,
            },
            "datatype_summary": {
                "customer_id": "object",
                "tenure_months": "int64",
            },
            "numeric_columns": [
                "tenure_months",
            ],
            "categorical_columns": [
                "customer_id",
            ],
            "numeric_like_columns": {},
            "categorical_inconsistencies": {},
            "formatting_issues": {},
            "outlier_summary": {},
            "potential_issues": [
                "Missing values detected.",
            ],
            "analyst_questions": [
                "How should missing tenure values be treated?",
            ],
        }
    )

    assert artifact.file_name == "customer_churn_poc.csv"
    assert artifact.row_count == 100
    assert artifact.duplicate_row_count == 2


def test_invalid_data_understanding_artifact_is_rejected():
    with pytest.raises(ValidationError):
        validate_data_understanding(
            {
                "file_name": "customer_churn_poc.csv",
                "row_count": -1,
                "column_count": 10,
            }
        )


def test_missing_required_metadata_is_rejected():
    with pytest.raises(ValidationError):
        validate_data_understanding(
            {
                "file_name": "customer_churn_poc.csv",
                "row_count": 100,
            }
        )
