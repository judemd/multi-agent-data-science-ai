import pandas as pd
import pytest

from domain.data_preparation_action import DataPreparationAction
from tools.data_preparation_executor import execute_data_preparation_actions


def test_normalize_nulls_converts_supported_textual_markers_only():
    dataframe = pd.DataFrame(
        {
            "status": [
                "Active",
                " N/A ",
                "NULL",
                "none",
                "NaN",
                " na ",
                "",
                "   ",
                "unknown",
                None,
                42,
            ],
            "other": [" N/A "] * 11,
        }
    )

    action = DataPreparationAction(
        operation="normalize_nulls",
        column="status",
        reason="Human approved converting textual null placeholders.",
    )

    prepared = execute_data_preparation_actions(dataframe, [action])

    assert prepared["status"].isna().sum() == 8
    assert prepared.loc[0, "status"] == "Active"
    assert prepared.loc[8, "status"] == "unknown"
    assert pd.isna(prepared.loc[9, "status"])
    assert prepared.loc[10, "status"] == 42

    assert prepared["other"].tolist() == [" N/A "] * 11
    assert dataframe.loc[1, "status"] == " N/A "
    assert dataframe.loc[6, "status"] == ""
    assert prepared is not dataframe


@pytest.mark.parametrize("column", [None, "unknown_column"])
def test_normalize_nulls_rejects_missing_or_unknown_column(column):
    dataframe = pd.DataFrame({"status": [" N/A "]})

    action = DataPreparationAction(
        operation="normalize_nulls",
        column=column,
        reason="Approved null normalization.",
    )

    with pytest.raises(ValueError):
        execute_data_preparation_actions(dataframe, [action])

def test_normalize_nulls_before_mode_imputation():
    dataframe = pd.DataFrame(
        {
            "status": ["Active", "Active", " N/A ", None],
        }
    )

    actions = [
        DataPreparationAction(
            operation="normalize_nulls",
            column="status",
            reason="Human approved normalization.",
        ),
        DataPreparationAction(
            operation="impute_missing",
            column="status",
            strategy="mode",
            reason="Human approved mode imputation.",
        ),
    ]

    prepared = execute_data_preparation_actions(dataframe, actions)

    assert prepared["status"].tolist() == [
        "Active",
        "Active",
        "Active",
        "Active",
    ]
    assert dataframe.loc[2, "status"] == " N/A "
    assert pd.isna(dataframe.loc[3, "status"])
