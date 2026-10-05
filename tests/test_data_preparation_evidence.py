import pandas as pd

from domain.data_preparation import DataPreparationArtifact
from tools.data_preparation_evidence import (
    build_data_preparation_evidence,
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
                None,
                300.0,
                400.0,
            ],
            "segment": [
                "Enterprise",
                "enterprise ",
                "SMB",
                "SMB",
            ],
            "signup_date": [
                "2025-01-01",
                "2025-02-01",
                "2025-03-01",
                "2025-04-01",
            ],
        }
    )


def test_build_data_preparation_evidence_returns_artifact():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert isinstance(result, DataPreparationArtifact)
    assert result.file_name == "customers.csv"
    assert result.row_count == 4
    assert result.column_count == 4


def test_evidence_identifies_missing_values():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert result.missing_value_summary == {
        "revenue": 1,
    }


def test_evidence_identifies_categorical_inconsistency():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert "segment" in result.categorical_inconsistencies


def test_evidence_identifies_candidate_date_column():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert "signup_date" in result.candidate_date_columns


def test_evidence_does_not_modify_dataframe():
    dataframe = build_dataframe()
    before = dataframe.copy(deep=True)

    build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    pd.testing.assert_frame_equal(dataframe, before)
