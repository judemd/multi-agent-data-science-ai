import numpy as np
import pandas as pd

from tools.modeling_sanity_check import check_modeling_dataset


def test_clean_dataset_passes():
    dataframe = pd.DataFrame({
        "feature": [10, 20, 30, 40],
        "target": [0, 1, 0, 1],
    })

    result = check_modeling_dataset(dataframe, "target")

    assert result.status == "PASS"
    assert result.errors == []
    assert result.warnings == []


def test_blank_and_textual_null_features_produce_warning():
    dataframe = pd.DataFrame({
        "feature": ["A", " ", "N/A", "B"],
        "target": [0, 1, 0, 1],
    })

    result = check_modeling_dataset(dataframe, "target")

    assert result.status == "WARNING"
    assert len(result.warnings) == 1
    assert "2 blank or textual-null values" in result.warnings[0]


def test_missing_feature_values_produce_warning():
    dataframe = pd.DataFrame({
        "feature": [1.0, np.nan, 3.0, 4.0],
        "target": [0, 1, 0, 1],
    })

    result = check_modeling_dataset(dataframe, "target")

    assert result.status == "WARNING"
    assert "1 missing values" in result.warnings[0]


def test_blank_target_blocks_training():
    dataframe = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": ["yes", " ", "no", "yes"],
    })

    result = check_modeling_dataset(dataframe, "target")

    assert result.status == "BLOCK"
    assert any("Target contains" in error for error in result.errors)


def test_infinite_numeric_feature_blocks_training():
    dataframe = pd.DataFrame({
        "feature": [1.0, np.inf, 3.0, 4.0],
        "target": [0, 1, 0, 1],
    })

    result = check_modeling_dataset(dataframe, "target")

    assert result.status == "BLOCK"
    assert any("infinite numeric values" in error for error in result.errors)


def test_single_class_target_blocks_training():
    dataframe = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [1, 1, 1, 1],
    })

    result = check_modeling_dataset(dataframe, "target")

    assert result.status == "BLOCK"
    assert any("exactly two target classes" in error for error in result.errors)