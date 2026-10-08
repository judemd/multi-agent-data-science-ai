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
            strategy="unsupported_strategy",
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

def test_mean_imputation():
    dataframe = pd.DataFrame(
        {
            "score": [10.0, 20.0, None, 60.0],
        }
    )

    actions = [
        DataPreparationAction(
            operation="impute_missing",
            column="score",
            strategy="mean",
            reason="Human approved mean imputation.",
        )
    ]

    result = execute_data_preparation_actions(
        dataframe,
        actions,
    )

    assert result["score"].tolist() == [
        10.0,
        20.0,
        30.0,
        60.0,
    ]
    assert pd.isna(dataframe.loc[2, "score"])


def test_mode_imputation_for_categorical_column():
    dataframe = pd.DataFrame(
        {"segment": ["retail", None, "business", "retail"]}
    )

    action = DataPreparationAction(
        operation="impute_missing",
        column="segment",
        strategy="mode",
        reason="Human approved categorical mode imputation.",
    )

    result = execute_data_preparation_actions(dataframe, [action])

    assert result["segment"].tolist() == [
        "retail",
        "retail",
        "business",
        "retail",
    ]
    assert pd.isna(dataframe.loc[1, "segment"])


def test_mode_imputation_uses_first_observed_value_for_ties():
    dataframe = pd.DataFrame(
        {"segment": ["business", "retail", None, "retail", "business"]}
    )

    action = DataPreparationAction(
        operation="impute_missing",
        column="segment",
        strategy="mode",
        reason="Use deterministic tie-breaking.",
    )

    result = execute_data_preparation_actions(dataframe, [action])

    assert result.loc[2, "segment"] == "business"


def test_mode_imputation_rejects_all_missing_column():
    dataframe = pd.DataFrame(
        {"segment": [None, None, None]}
    )

    action = DataPreparationAction(
        operation="impute_missing",
        column="segment",
        strategy="mode",
        reason="Test empty categorical column.",
    )

    with pytest.raises(
        ValueError,
        match="Cannot calculate mode",
    ):
        execute_data_preparation_actions(dataframe, [action])


def test_drop_rows_only_for_approved_column():
    dataframe = pd.DataFrame(
        {
            "revenue": [10.0, None, 30.0, 40.0],
            "segment": ["retail", "business", None, "retail"],
        }
    )

    action = DataPreparationAction(
        operation="drop_rows",
        column="revenue",
        reason="Human approved removing rows with missing revenue.",
    )

    result = execute_data_preparation_actions(dataframe, [action])

    assert result["revenue"].tolist() == [10.0, 30.0, 40.0]
    assert result["segment"].tolist()[:1] == ["retail"]
    assert pd.isna(result.loc[1, "segment"])
    assert result.index.tolist() == [0, 1, 2]

    # Execution must not mutate the source dataset.
    assert len(dataframe) == 4
    assert pd.isna(dataframe.loc[1, "revenue"])


def test_drop_rows_requires_column():
    dataframe = pd.DataFrame({"revenue": [10.0, None]})

    action = DataPreparationAction(
        operation="drop_rows",
        reason="Test missing column.",
    )

    with pytest.raises(ValueError, match="requires a column"):
        execute_data_preparation_actions(dataframe, [action])


def test_drop_rows_rejects_unknown_column():
    dataframe = pd.DataFrame({"revenue": [10.0, None]})

    action = DataPreparationAction(
        operation="drop_rows",
        column="unknown",
        reason="Test unknown column.",
    )

    with pytest.raises(ValueError, match="unknown column"):
        execute_data_preparation_actions(dataframe, [action])
