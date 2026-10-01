from pathlib import Path

import pandas as pd

from tools.data_profiler import profile_dataframe, profile_dataset


def test_profile_csv_dataset():
    dataset = Path("datasets/customer_churn_poc.csv")

    artifact = profile_dataset(str(dataset))

    assert artifact.file_name == "customer_churn_poc.csv"
    assert artifact.row_count > 0
    assert artifact.column_count > 0
    assert len(artifact.columns) == artifact.column_count


def test_profile_csv_identifies_column_types():
    dataset = Path("datasets/customer_churn_poc.csv")

    artifact = profile_dataset(str(dataset))

    assert len(artifact.numeric_columns) > 0
    assert len(artifact.categorical_columns) > 0
    assert artifact.datatype_summary


def test_profile_csv_contains_quality_sections():
    dataset = Path("datasets/customer_churn_poc.csv")

    artifact = profile_dataset(str(dataset))

    assert artifact.missing_value_summary is not None
    assert artifact.duplicate_value_summary is not None
    assert artifact.numeric_like_columns is not None
    assert artifact.categorical_inconsistencies is not None
    assert artifact.formatting_issues is not None
    assert artifact.outlier_summary is not None
    assert artifact.potential_issues is not None
    assert artifact.analyst_questions is not None


def test_profile_xlsx_dataset(tmp_path):
    dataset = tmp_path / "sample.xlsx"

    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "tenure_months": [12, 24, 36],
            "segment": ["Consumer", "Business", "Consumer"],
        }
    )

    dataframe.to_excel(dataset, index=False)

    artifact = profile_dataset(str(dataset))

    assert artifact.file_name == "sample.xlsx"
    assert artifact.row_count == 3
    assert artifact.column_count == 3
    assert "tenure_months" in artifact.numeric_columns
    assert "segment" in artifact.categorical_columns

def test_profile_dataframe_accepts_in_memory_dataframe():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    result = profile_dataframe(
        dataframe,
        file_name="customers.csv",
    )

    assert result.file_name == "customers.csv"
    assert result.row_count == 3
    assert result.column_count == 2
    assert result.columns == [
        "customer_id",
        "revenue",
    ]
    assert result.numeric_columns == ["revenue"]
