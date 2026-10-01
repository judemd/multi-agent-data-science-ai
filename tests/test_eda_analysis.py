import pandas as pd

from tools.eda_analysis import (
    calculate_numeric_correlations,
    summarize_categorical_columns,
    summarize_numeric_columns,
)


def test_summarizes_numeric_columns():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300, 400],
            "tenure": [10, 20, 30, 40],
            "segment": ["A", "B", "A", "B"],
        }
    )

    result = summarize_numeric_columns(dataframe)

    assert set(result) == {"revenue", "tenure"}

    assert result["revenue"]["count"] == 4
    assert result["revenue"]["mean"] == 250.0
    assert result["revenue"]["min"] == 100.0
    assert result["revenue"]["max"] == 400.0


def test_returns_empty_numeric_summary_when_no_numeric_columns_exist():
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B", "C"],
        }
    )

    assert summarize_numeric_columns(dataframe) == {}


def test_summarizes_categorical_columns():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Business",
                "Consumer",
                "Consumer",
            ],
            "revenue": [100, 200, 300, 400],
        }
    )

    result = summarize_categorical_columns(dataframe)

    assert set(result) == {"segment"}
    assert result["segment"]["unique_count"] == 2
    assert result["segment"]["top_values"]["Consumer"] == 3
    assert result["segment"]["top_values"]["Business"] == 1


def test_categorical_summary_includes_missing_values():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                None,
                "Business",
                None,
            ],
        }
    )

    result = summarize_categorical_columns(dataframe)

    assert result["segment"]["unique_count"] == 2
    assert result["segment"]["top_values"]["nan"] == 2


def test_returns_empty_categorical_summary_when_no_categorical_columns_exist():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    assert summarize_categorical_columns(dataframe) == {}


def test_calculates_numeric_correlations():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300, 400],
            "tenure": [10, 20, 30, 40],
            "age": [20, 30, 40, 50],
        }
    )

    result = calculate_numeric_correlations(dataframe)

    assert set(result) == {
        "revenue",
        "tenure",
        "age",
    }

    assert result["revenue"]["tenure"] == 1.0
    assert result["revenue"]["age"] == 1.0


def test_correlation_summary_excludes_self_correlation():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
            "tenure": [10, 20, 30],
        }
    )

    result = calculate_numeric_correlations(dataframe)

    assert "revenue" not in result["revenue"]
    assert "tenure" not in result["tenure"]


def test_returns_empty_correlations_with_one_numeric_column():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    assert calculate_numeric_correlations(dataframe) == {}


def test_returns_empty_correlations_without_numeric_columns():
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B", "C"],
        }
    )

    assert calculate_numeric_correlations(dataframe) == {}
