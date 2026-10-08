import pandas as pd
import pytest

from domain.data_preparation_action import DataPreparationAction
from tools.data_preparation_executor import execute_data_preparation_actions


def test_normalize_categories_uses_most_frequent_spelling():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "consumer",
                "Consumer",
                " Consumer ",
                "Business",
                "business",
                "business",
            ],
            "other": [" untouched "] * 7,
        }
    )

    action = DataPreparationAction(
        operation="normalize_categories",
        column="segment",
        reason="Human approved categorical normalization.",
    )

    prepared = execute_data_preparation_actions(dataframe, [action])

    assert prepared["segment"].tolist() == [
        "Consumer",
        "Consumer",
        "Consumer",
        "Consumer",
        "business",
        "business",
        "business",
    ]
    assert prepared["other"].tolist() == [" untouched "] * 7
    assert dataframe.loc[1, "segment"] == "consumer"
    assert dataframe.loc[3, "segment"] == " Consumer "
    assert prepared is not dataframe


def test_normalize_categories_uses_first_observed_spelling_on_ties():
    dataframe = pd.DataFrame(
        {
            "status": ["YES", "yes", "Pending", "pending"],
        }
    )

    action = DataPreparationAction(
        operation="normalize_categories",
        column="status",
        reason="Human approved deterministic canonical labels.",
    )

    prepared = execute_data_preparation_actions(dataframe, [action])

    assert prepared["status"].tolist() == [
        "YES",
        "YES",
        "Pending",
        "Pending",
    ]


def test_normalize_categories_preserves_nulls_markers_and_non_strings():
    dataframe = pd.DataFrame(
        {
            "status": [
                "Active",
                "active",
                "N/A",
                " n/a ",
                None,
                "unknown",
                42,
            ],
        }
    )

    action = DataPreparationAction(
        operation="normalize_categories",
        column="status",
        reason="Human approved category normalization.",
    )

    prepared = execute_data_preparation_actions(dataframe, [action])

    assert prepared["status"].tolist()[:4] == [
        "Active",
        "Active",
        "N/A",
        " n/a ",
    ]
    assert pd.isna(prepared.loc[4, "status"])
    assert prepared.loc[5, "status"] == "unknown"
    assert prepared.loc[6, "status"] == 42
    assert dataframe.loc[1, "status"] == "active"


@pytest.mark.parametrize("column", [None, "unknown_column"])
def test_normalize_categories_rejects_missing_or_unknown_column(column):
    dataframe = pd.DataFrame({"status": ["Active", "active"]})

    action = DataPreparationAction(
        operation="normalize_categories",
        column=column,
        reason="Approved category normalization.",
    )

    with pytest.raises(ValueError):
        execute_data_preparation_actions(dataframe, [action])
