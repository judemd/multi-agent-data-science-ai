"""Tests for deterministic model training safeguards."""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tools.model_training import (
    evaluate_training_result,
    train_selected_model,
    validate_training_features,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [21, 35, 48, 52],
        "region": ["North", "South", "North", "South"],
        "churned": [0, 1, 0, 1],
    })


def test_valid_features_preserve_source_data(sample_data):
    original = sample_data.copy(deep=True)

    selected = validate_training_features(
        sample_data,
        "churned",
        "Logistic Regression",
        ["age", "region"],
    )

    assert selected.columns.tolist() == [
        "age",
        "region",
        "churned",
    ]
    pd.testing.assert_frame_equal(sample_data, original)


def test_random_forest_is_supported(sample_data):
    selected = validate_training_features(
        sample_data,
        "churned",
        "Random Forest",
        ["age"],
    )

    assert selected.columns.tolist() == ["age", "churned"]


def test_unsupported_model_is_rejected(sample_data):
    with pytest.raises(ValueError, match="Unsupported training model"):
        validate_training_features(
            sample_data,
            "churned",
            "Unapproved Model",
            ["age"],
        )


def test_empty_feature_selection_is_rejected(sample_data):
    with pytest.raises(ValueError, match="At least one approved"):
        validate_training_features(
            sample_data,
            "churned",
            "Logistic Regression",
            [],
        )


def test_duplicate_feature_selection_is_rejected(sample_data):
    with pytest.raises(ValueError, match="contain duplicates"):
        validate_training_features(
            sample_data,
            "churned",
            "Logistic Regression",
            ["age", "age"],
        )


def test_target_as_feature_is_rejected(sample_data):
    with pytest.raises(ValueError, match="cannot be included"):
        validate_training_features(
            sample_data,
            "churned",
            "Logistic Regression",
            ["age", "churned"],
        )


def test_unknown_feature_is_rejected(sample_data):
    with pytest.raises(ValueError, match="Unknown approved"):
        validate_training_features(
            sample_data,
            "churned",
            "Logistic Regression",
            ["unknown_feature"],
        )


def test_missing_target_column_is_rejected(sample_data):
    with pytest.raises(ValueError, match="Target column"):
        validate_training_features(
            sample_data,
            "missing_target",
            "Logistic Regression",
            ["age"],
        )


def test_blank_target_values_are_rejected(sample_data):
    sample_data.loc[0, "churned"] = None

    with pytest.raises(ValueError, match="sanity check failed"):
        validate_training_features(
            sample_data,
            "churned",
            "Logistic Regression",
            ["age"],
        )


@pytest.fixture
def training_data():
    return pd.DataFrame({
        "age": [21, 35, 48, 52, 28, 42, 31, 60, 24, 55,
                39, 44, 29, 63, 33, 57, 26, 49, 37, 54],
        "region": ["North", "South"] * 10,
        "churned": [0, 1] * 10,
    })


def test_logistic_regression_training(training_data):
    result = train_selected_model(
        training_data,
        "churned",
        "Logistic Regression",
        ["age", "region"],
    )

    assert isinstance(
        result.model_pipeline.named_steps["classifier"],
        LogisticRegression,
    )
    assert isinstance(
        result.baseline_pipeline.named_steps["classifier"],
        DummyClassifier,
    )
    assert result.train_row_count == 16
    assert result.test_row_count == 4
    assert len(result.model_pipeline.predict(result.x_test)) == 4


def test_random_forest_training(training_data):
    result = train_selected_model(
        training_data,
        "churned",
        "Random Forest",
        ["age", "region"],
    )

    assert isinstance(
        result.model_pipeline.named_steps["classifier"],
        RandomForestClassifier,
    )
    assert result.train_row_count == 16
    assert result.test_row_count == 4
    assert len(result.baseline_pipeline.predict(result.x_test)) == 4


def test_training_is_reproducible(training_data):
    first = train_selected_model(
        training_data,
        "churned",
        "Logistic Regression",
        ["age", "region"],
    )
    second = train_selected_model(
        training_data,
        "churned",
        "Logistic Regression",
        ["age", "region"],
    )

    assert (
        first.model_pipeline.predict(first.x_test).tolist()
        == second.model_pipeline.predict(second.x_test).tolist()
    )
    assert first.x_test.index.tolist() == second.x_test.index.tolist()


def test_training_handles_blank_categorical_values(training_data):
    training_data.loc[0, "region"] = " "
    training_data.loc[1, "region"] = "N/A"

    result = train_selected_model(
        training_data,
        "churned",
        "Logistic Regression",
        ["age", "region"],
    )

    assert len(result.model_pipeline.predict(result.x_test)) == 4


def test_training_handles_missing_numeric_values(training_data):
    training_data.loc[0, "age"] = None
    training_data.loc[1, "age"] = None

    result = train_selected_model(
        training_data,
        "churned",
        "Logistic Regression",
        ["age", "region"],
    )

    assert len(result.model_pipeline.predict(result.x_test)) == 4


def test_training_preserves_original_dataframe(training_data):
    original = training_data.copy(deep=True)

    train_selected_model(
        training_data,
        "churned",
        "Random Forest",
        ["age", "region"],
    )

    pd.testing.assert_frame_equal(training_data, original)


@pytest.mark.parametrize(
    "model_name",
    ["Logistic Regression", "Random Forest"],
)
def test_evaluation_returns_metrics_for_both_models(
    training_data,
    model_name,
):
    trained = train_selected_model(
        training_data,
        "churned",
        model_name,
        ["age", "region"],
    )

    result = evaluate_training_result(trained)

    assert result.selected_model == model_name
    assert result.train_row_count == 16
    assert result.test_row_count == 4
    assert result.feature_columns == ["age", "region"]

    for metrics in [result.model_metrics, result.baseline_metrics]:
        for metric in [
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.roc_auc,
        ]:
            assert 0.0 <= metric <= 1.0


def test_evaluation_baseline_uses_majority_class(training_data):
    trained = train_selected_model(
        training_data,
        "churned",
        "Logistic Regression",
        ["age", "region"],
    )

    result = evaluate_training_result(trained)

    assert result.baseline_metrics.roc_auc == pytest.approx(0.5)
    assert result.baseline_metrics.recall == pytest.approx(0.0)


def test_training_rejects_non_binary_encoded_labels(training_data):
    training_data["churned"] = training_data["churned"].map({
        0: "no",
        1: "yes",
    })

    with pytest.raises(ValueError, match="labels 0 and 1"):
        train_selected_model(
            training_data,
            "churned",
            "Logistic Regression",
            ["age", "region"],
        )