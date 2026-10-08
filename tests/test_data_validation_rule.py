import pytest
from pydantic import ValidationError

from domain.data_validation_rule import DataValidationRule


BASE = {
    "rule_id": "rule-001",
    "issue_type": "invalid_value",
    "column": "age",
    "description": "Check the declared validity constraint.",
    "source": "Human-approved data specification",
}


def make_rule(**overrides):
    return DataValidationRule(**(BASE | overrides))


def test_numeric_range_accepts_valid_bounds():
    rule = make_rule(
        rule_type="numeric_range",
        min_value=0,
        max_value=120,
    )

    assert rule.min_value == 0
    assert rule.max_value == 120


def test_numeric_range_accepts_one_sided_bound():
    rule = make_rule(
        rule_type="numeric_range",
        min_value=0,
    )

    assert rule.max_value is None


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"min_value": 100, "max_value": 10},
        {"min_value": 0, "allowed_values": ["0", "1"]},
    ],
)
def test_numeric_range_rejects_invalid_parameters(parameters):
    with pytest.raises(ValidationError):
        make_rule(rule_type="numeric_range", **parameters)


def test_allowed_values_accepts_declared_categories():
    rule = make_rule(
        rule_type="allowed_values",
        allowed_values=["active", "inactive"],
    )

    assert rule.allowed_values == ["active", "inactive"]


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"allowed_values": []},
        {"allowed_values": ["active", ""]},
        {"allowed_values": ["active"], "min_value": 0},
    ],
)
def test_allowed_values_rejects_invalid_parameters(parameters):
    with pytest.raises(ValidationError):
        make_rule(rule_type="allowed_values", **parameters)


def test_column_comparison_accepts_valid_relationship():
    rule = make_rule(
        rule_type="column_comparison",
        issue_type="business_rule_violation",
        column="start_value",
        comparison_column="end_value",
        comparison_operator="<=",
    )

    assert rule.comparison_operator == "<="


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"comparison_column": "end_value"},
        {"comparison_operator": "<="},
        {"comparison_column": "age", "comparison_operator": "<="},
        {
            "comparison_column": "end_value",
            "comparison_operator": "<=",
            "max_value": 100,
        },
    ],
)
def test_column_comparison_rejects_invalid_parameters(parameters):
    with pytest.raises(ValidationError):
        make_rule(
            rule_type="column_comparison",
            **parameters,
        )


@pytest.mark.parametrize(
    "field",
    ["rule_id", "column", "description", "source"],
)
def test_rule_rejects_empty_required_metadata(field):
    with pytest.raises(ValidationError):
        make_rule(
            rule_type="numeric_range",
            min_value=0,
            **{field: ""},
        )


def test_rule_rejects_unsupported_issue_type():
    with pytest.raises(ValidationError):
        make_rule(
            rule_type="numeric_range",
            min_value=0,
            issue_type="outlier",
        )
