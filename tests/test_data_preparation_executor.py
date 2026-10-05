import pandas as pd
import pytest

from domain.data_preparation_action import DataPreparationAction
from tools.data_preparation_executor import (
    execute_data_preparation_actions,
)


def test_remove_duplicates():
    dataframe = pd.DataFrame(
        {
            "customer": [1, 1, 2],
            "value": [10, 10, 20],
        }
    )

    actions = [
        DataPreparationAction(
            operation="remove_duplicates",
            reason="Duplicate rows were identified.",
        )
    ]

    result = execute_data_preparation_actions(
        dataframe,
        actions,
    )

    assert len(result) == 2
    assert result["customer"].tolist() == [1, 2]


def test_exclude_feature():
    dataframe = pd.DataFrame(
        {
            "customer": [1, 2],
            "country": ["UK", "US"],
            "value": [10, 20],
        }
    )

    actions = [
        DataPreparationAction(
            operation="exclude_feature",
            column="country",
            reason="Country is not required.",
        )
    ]

    result = execute_data_preparation_actions(
        dataframe,
        actions,
    )

    assert "country" not in result.columns
    assert list(result.columns) == ["customer", "value"]


def test_median_imputation():
    dataframe = pd.DataFrame(
        {
            "score": [10.0, 20.0, None, 30.0],
        }
    )

    actions = [
        DataPreparationAction(
            operation="impute_missing",
            column="score",
            strategy="median",
            reason="Missing scores require treatment.",
        )
    ]

    result = execute_data_preparation_actions(
        dataframe,
        actions,
    )

    assert result["score"].tolist() == [
        10.0,
        20.0,
        20.0,
        30.0,
    ]


def test_missing_exclude_column_raises():
    dataframe = pd.DataFrame(
        {"value": [1, 2, 3]}
    )

    actions = [
        DataPreparationAction(
            operation="exclude_feature",
            column="missing_column",
            reason="Test invalid column.",
        )
    ]

    with pytest.raises(
        ValueError,
        match="Cannot exclude missing column",
    ):
        execute_data_preparation_actions(
            dataframe,
            actions,
        )


def test_missing_imputation_column_raises():
    dataframe = pd.DataFrame(
        {"value": [1, 2, 3]}
    )

    actions = [
        DataPreparationAction(
            operation="impute_missing",
            column="missing_column",
            strategy="median",
            reason="Test invalid column.",
        )
    ]

    with pytest.raises(ValueError, match="unknown column"):
        execute_data_preparation_actions(
            dataframe,
            actions,
        )


def test_median_imputation_requires_numeric_column():
    dataframe = pd.DataFrame(
        {"score": ["low", None, "high"]}
    )

    actions = [
        DataPreparationAction(
            operation="impute_missing",
            column="score",
            strategy="median",
            reason="Test invalid strategy.",
        )
    ]

    with pytest.raises(
        ValueError,
        match="requires a numeric column",
    ):
        execute_data_preparation_actions(
            dataframe,
            actions,
        )


def test_unsupported_imputation_strategy_raises():
    dataframe = pd.DataFrame(
        {"score": [10.0, None, 30.0]}
    )

    actions = [
        DataPreparationAction(
            operation="impute_missing",
            column="score",
            strategy="mean",
            reason="Test unsupported strategy.",
        )
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported imputation strategy",
    ):
        execute_data_preparation_actions(
            dataframe,
            actions,
        )