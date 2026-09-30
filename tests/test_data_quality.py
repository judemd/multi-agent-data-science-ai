import pandas as pd
import pytest

from tools.data_quality import (
    count_blank_and_null_like_values,
    count_duplicate_rows,
    detect_categorical_inconsistencies,
    detect_iqr_outliers,
    detect_numeric_like_columns,
    detect_string_formatting_issues,
    find_duplicate_values,
)


def test_detects_blank_and_null_like_values():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004"],
            "segment": ["Consumer", "", "  ", "N/A"],
            "status": ["Active", "NA", "null", "Inactive"],
            "value": [100.0, None, 200.0, 300.0],
        }
    )

    counts = count_blank_and_null_like_values(dataframe)

    assert counts["segment"] == 3
    assert counts["status"] == 2
    assert counts["value"] == 1


def test_clean_columns_are_not_reported():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "status": ["Active", "Inactive"],
            "value": [100.0, 200.0],
        }
    )

    counts = count_blank_and_null_like_values(dataframe)

    assert counts == {}


def test_counts_exact_duplicate_rows():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C002", "C003"],
            "value": [100, 200, 200, 300],
        }
    )

    assert count_duplicate_rows(dataframe) == 1


def test_returns_zero_when_no_duplicate_rows_exist():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "value": [100, 200, 300],
        }
    )

    assert count_duplicate_rows(dataframe) == 0


def test_finds_duplicate_values_in_a_column():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C002", "C003", "C003", "C003"],
        }
    )

    duplicates = find_duplicate_values(dataframe, "customer_id")

    assert duplicates == {
        "C002": 2,
        "C003": 3,
    }


def test_ignores_null_values_when_finding_duplicates():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", None, None, "C001"],
        }
    )

    duplicates = find_duplicate_values(dataframe, "customer_id")

    assert duplicates == {"C001": 2}


def test_raises_for_unknown_column():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
        }
    )

    with pytest.raises(KeyError):
        find_duplicate_values(dataframe, "account_id")


def test_detects_case_and_whitespace_inconsistencies():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "consumer",
                " Consumer ",
                "Business",
                "business",
            ]
        }
    )

    inconsistencies = detect_categorical_inconsistencies(
        dataframe,
        "segment",
    )

    assert inconsistencies == {
        "business": ["Business", "business"],
        "consumer": [" Consumer ", "Consumer", "consumer"],
    }


def test_does_not_report_already_consistent_categories():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Consumer",
                "Business",
                "Business",
            ]
        }
    )

    inconsistencies = detect_categorical_inconsistencies(
        dataframe,
        "segment",
    )

    assert inconsistencies == {}


def test_raises_for_unknown_categorical_column():
    dataframe = pd.DataFrame(
        {
            "segment": ["Consumer", "Business"],
        }
    )

    with pytest.raises(KeyError):
        detect_categorical_inconsistencies(
            dataframe,
            "unknown_column",
        )


def test_detects_iqr_outliers():
    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                105,
                110,
                115,
                120,
                125,
                130,
                1000,
            ]
        }
    )

    result = detect_iqr_outliers(dataframe, "revenue")

    assert result["outlier_count"] == 1
    assert result["upper_bound"] < 1000


def test_iqr_detector_returns_zero_when_no_outliers_exist():
    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                105,
                110,
                115,
                120,
            ]
        }
    )

    result = detect_iqr_outliers(dataframe, "revenue")

    assert result["outlier_count"] == 0


def test_iqr_detector_ignores_missing_values():
    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                105,
                None,
                110,
                115,
                120,
                1000,
            ]
        }
    )

    result = detect_iqr_outliers(dataframe, "revenue")

    assert result["outlier_count"] == 1


def test_iqr_detector_rejects_non_numeric_column():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Business",
                "Consumer",
            ]
        }
    )

    with pytest.raises(TypeError):
        detect_iqr_outliers(dataframe, "segment")


def test_iqr_detector_raises_for_unknown_column():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    with pytest.raises(KeyError):
        detect_iqr_outliers(dataframe, "unknown_column")


def test_detects_numeric_like_text_column():
    dataframe = pd.DataFrame(
        {
            "tenure_months": [
                "12",
                "18",
                "24",
                "36",
                "unknown",
            ]
        }
    )

    result = detect_numeric_like_columns(dataframe)

    assert "tenure_months" in result
    assert result["tenure_months"]["numeric_like_count"] == 4
    assert result["tenure_months"]["non_null_count"] == 5
    assert result["tenure_months"]["numeric_like_proportion"] == 0.8


def test_does_not_report_normal_categorical_column_as_numeric_like():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Business",
                "Consumer",
                "Business",
            ]
        }
    )

    result = detect_numeric_like_columns(dataframe)

    assert result == {}


def test_numeric_like_detection_respects_threshold():
    dataframe = pd.DataFrame(
        {
            "value": [
                "10",
                "20",
                "30",
                "unknown",
                "unknown",
            ]
        }
    )

    result = detect_numeric_like_columns(
        dataframe,
        threshold=0.8,
    )

    assert result == {}


def test_numeric_like_detection_rejects_invalid_threshold():
    dataframe = pd.DataFrame(
        {
            "value": ["10", "20"],
        }
    )

    with pytest.raises(ValueError):
        detect_numeric_like_columns(dataframe, threshold=0)

    with pytest.raises(ValueError):
        detect_numeric_like_columns(dataframe, threshold=1.1)


def test_detects_string_whitespace_and_case_formatting():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                " consumer",
                "Consumer ",
                "BUSINESS",
                "Business",
            ]
        }
    )

    result = detect_string_formatting_issues(
        dataframe,
        "segment",
    )

    assert result["leading_or_trailing_whitespace_count"] == 2
    assert result["case_inconsistency_count"] == 5


def test_clean_string_column_has_no_formatting_issues():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Consumer",
                "Business",
                "Business",
            ]
        }
    )

    result = detect_string_formatting_issues(
        dataframe,
        "segment",
    )

    assert result == {
        "leading_or_trailing_whitespace_count": 0,
        "case_inconsistency_count": 0,
    }


def test_formatting_detector_rejects_non_string_column():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    with pytest.raises(TypeError):
        detect_string_formatting_issues(dataframe, "revenue")


def test_formatting_detector_raises_for_unknown_column():
    dataframe = pd.DataFrame(
        {
            "segment": ["Consumer", "Business"],
        }
    )

    with pytest.raises(KeyError):
        detect_string_formatting_issues(
            dataframe,
            "unknown_column",
        )
