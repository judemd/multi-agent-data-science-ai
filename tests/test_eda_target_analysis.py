import pandas as pd
import pytest

from tools.eda_target_analysis import (
    summarize_categorical_target_relationships,
    summarize_numeric_target_relationships,
)


def test_summarizes_numeric_target_relationships():
    dataframe = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5],
            "revenue": [10, 20, 30, 40, 50],
            "churn": [1, 1, 0, 0, 0],
        }
    )

    result = summarize_numeric_target_relationships(
        dataframe,
        "churn",
    )

    assert "tenure" in result
    assert "revenue" in result
    assert result["tenure"] < 0
    assert result["revenue"] < 0


def test_numeric_target_relationship_excludes_target_itself():
    dataframe = pd.DataFrame(
        {
            "revenue": [10, 20, 30],
            "churn": [0, 1, 0],
        }
    )

    result = summarize_numeric_target_relationships(
        dataframe,
        "churn",
    )

    assert "churn" not in result


def test_numeric_target_relationship_requires_existing_target():
    dataframe = pd.DataFrame(
        {
            "revenue": [10, 20, 30],
        }
    )

    with pytest.raises(KeyError):
        summarize_numeric_target_relationships(
            dataframe,
            "churn",
        )


def test_numeric_target_relationship_requires_numeric_target():
    dataframe = pd.DataFrame(
        {
            "revenue": [10, 20, 30],
            "churn": ["yes", "no", "yes"],
        }
    )

    with pytest.raises(TypeError):
        summarize_numeric_target_relationships(
            dataframe,
            "churn",
        )


def test_summarizes_categorical_target_relationships():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Consumer",
                "Business",
                "Business",
            ],
            "churn": [1, 0, 0, 0],
        }
    )

    result = summarize_categorical_target_relationships(
        dataframe,
        "churn",
    )

    assert result["segment"]["Consumer"]["count"] == 2
    assert result["segment"]["Consumer"]["target_rate"] == 0.5
    assert result["segment"]["Business"]["count"] == 2
    assert result["segment"]["Business"]["target_rate"] == 0.0


def test_categorical_target_relationship_requires_binary_target():
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B", "A"],
            "churn": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError):
        summarize_categorical_target_relationships(
            dataframe,
            "churn",
        )


def test_categorical_target_relationship_requires_numeric_target():
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B", "A"],
            "churn": ["yes", "no", "yes"],
        }
    )

    with pytest.raises(TypeError):
        summarize_categorical_target_relationships(
            dataframe,
            "churn",
        )


def test_target_relationship_requires_existing_target():
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B", "A"],
        }
    )

    with pytest.raises(KeyError):
        summarize_categorical_target_relationships(
            dataframe,
            "churn",
        )
