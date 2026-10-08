import pandas as pd
import pytest

from domain.data_preparation_action import DataPreparationAction
from tools.data_preparation_executor import execute_data_preparation_actions


def test_trim_whitespace_preserves_non_strings_and_original_dataframe():
    dataframe = pd.DataFrame(
        {
            "value": ["  Active  ", None, 42, "  ", "Pending  "],
            "other": ["  unchanged  "] * 5,
        }
    )

    action = DataPreparationAction(
        operation="trim_whitespace",
        column="value",
        reason="Human approved trimming leading and trailing whitespace.",
    )

    prepared = execute_data_preparation_actions(dataframe, [action])

    assert prepared["value"].tolist() == [
        "Active",
        None,
        42,
        "",
        "Pending",
    ]
    assert prepared["other"].tolist() == ["  unchanged  "] * 5

    assert dataframe.loc[0, "value"] == "  Active  "
    assert dataframe.loc[3, "value"] == "  "
    assert prepared is not dataframe


@pytest.mark.parametrize("column", [None, "unknown"])
def test_trim_whitespace_rejects_missing_or_unknown_column(column):
    dataframe = pd.DataFrame({"status": [" Active "]})

    action = DataPreparationAction(
        operation="trim_whitespace",
        column=column,
        reason="Approved whitespace treatment.",
    )

    with pytest.raises(ValueError):
        execute_data_preparation_actions(dataframe, [action])
