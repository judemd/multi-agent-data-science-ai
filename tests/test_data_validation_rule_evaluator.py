import pandas as pd
import pytest

from domain.data_validation_rule import DataValidationRule
from tools.data_validation_rule_evaluator import evaluate_data_validation_rule


def make_rule(rule_type, column="value", **kwargs):
    return DataValidationRule(
        rule_id="test-rule",
        rule_type=rule_type,
        issue_type="invalid_value",
        column=column,
        description="Test a declared validation constraint.",
        source="Test specification",
        **kwargs,
    )


def test_numeric_range_counts_violations_and_includes_boundaries():
    dataframe = pd.DataFrame({"value": [-1, 0, 50, 100, 101, None]})
    rule = make_rule("numeric_range", min_value=0, max_value=100)

    assert evaluate_data_validation_rule(dataframe, rule) == {
        "checked_count": 5,
        "violation_count": 2,
    }


def test_numeric_range_supports_one_sided_bounds():
    dataframe = pd.DataFrame({"value": [-2, 0, 3]})
    rule = make_rule("numeric_range", min_value=0)

    assert evaluate_data_validation_rule(dataframe, rule) == {
        "checked_count": 3,
        "violation_count": 1,
    }


def test_allowed_values_excludes_canonical_missing_markers():
    dataframe = pd.DataFrame({
        "value": ["active", "inactive", "pending", " NA ", "null", "", None]
    })
    rule = make_rule(
        "allowed_values",
        allowed_values=["active", "inactive"],
    )

    assert evaluate_data_validation_rule(dataframe, rule) == {
        "checked_count": 3,
        "violation_count": 1,
    }


def test_allowed_values_preserves_case_sensitive_matching():
    dataframe = pd.DataFrame({"value": ["Active", "active"]})
    rule = make_rule("allowed_values", allowed_values=["active"])

    assert evaluate_data_validation_rule(dataframe, rule) == {
        "checked_count": 2,
        "violation_count": 1,
    }


def test_numeric_column_comparison_counts_violations():
    dataframe = pd.DataFrame({
        "start": [1, 5, 8, None],
        "end": [2, 5, 3, 10],
    })
    rule = make_rule(
        "column_comparison",
        column="start",
        comparison_column="end",
        comparison_operator="<=",
    )

    assert evaluate_data_validation_rule(dataframe, rule) == {
        "checked_count": 3,
        "violation_count": 1,
    }


def test_datetime_column_comparison_counts_violations():
    dataframe = pd.DataFrame({
        "start": pd.to_datetime(["2026-01-01", "2026-03-01", None]),
        "end": pd.to_datetime(["2026-02-01", "2026-02-01", "2026-04-01"]),
    })
    rule = make_rule(
        "column_comparison",
        column="start",
        comparison_column="end",
        comparison_operator="<=",
    )

    assert evaluate_data_validation_rule(dataframe, rule) == {
        "checked_count": 2,
        "violation_count": 1,
    }


def test_missing_column_raises_key_error():
    dataframe = pd.DataFrame({"other": [1, 2]})
    rule = make_rule("numeric_range", min_value=0)

    with pytest.raises(KeyError, match="does not exist"):
        evaluate_data_validation_rule(dataframe, rule)


def test_missing_comparison_column_raises_key_error():
    dataframe = pd.DataFrame({"value": [1, 2]})
    rule = make_rule(
        "column_comparison",
        comparison_column="missing",
        comparison_operator="<=",
    )

    with pytest.raises(KeyError, match="does not exist"):
        evaluate_data_validation_rule(dataframe, rule)


@pytest.mark.parametrize(
    "values",
    [
        ["1", "2", "3"],
        [True, False, True],
    ],
)
def test_numeric_range_rejects_non_numeric_columns(values):
    dataframe = pd.DataFrame({"value": values})
    rule = make_rule("numeric_range", min_value=0)

    with pytest.raises(TypeError, match="requires a numeric column"):
        evaluate_data_validation_rule(dataframe, rule)


def test_allowed_values_rejects_numeric_column():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    rule = make_rule("allowed_values", allowed_values=["1", "2"])

    with pytest.raises(TypeError, match="categorical/text"):
        evaluate_data_validation_rule(dataframe, rule)


def test_column_comparison_rejects_unconverted_date_strings():
    dataframe = pd.DataFrame({
        "start": ["2026-01-01"],
        "end": ["2026-02-01"],
    })
    rule = make_rule(
        "column_comparison",
        column="start",
        comparison_column="end",
        comparison_operator="<=",
    )

    with pytest.raises(TypeError, match="compatible numeric or datetime"):
        evaluate_data_validation_rule(dataframe, rule)


def test_evaluator_does_not_modify_dataframe():
    dataframe = pd.DataFrame({
        "value": ["active", "pending", " NA ", None],
    })
    original = dataframe.copy(deep=True)
    rule = make_rule("allowed_values", allowed_values=["active"])

    evaluate_data_validation_rule(dataframe, rule)

    pd.testing.assert_frame_equal(dataframe, original)


def test_rule_with_no_eligible_rows_returns_zero_counts():
    dataframe = pd.DataFrame({"value": [None, "NA", " null "]})
    rule = make_rule("allowed_values", allowed_values=["active"])

    assert evaluate_data_validation_rule(dataframe, rule) == {
        "checked_count": 0,
        "violation_count": 0,
    }
