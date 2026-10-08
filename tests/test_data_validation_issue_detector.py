import pandas as pd
import pytest

from domain.data_validation_rule import DataValidationRule
from tools.data_validation_issue_detector import detect_validation_rule_issues


def make_rule(rule_id="rule-001", **overrides):
    parameters = {
        "rule_id": rule_id,
        "rule_type": "numeric_range",
        "issue_type": "invalid_value",
        "column": "age",
        "description": "Age must be between 0 and 120.",
        "source": "Approved test specification",
        "min_value": 0,
        "max_value": 120,
    }
    return DataValidationRule(**(parameters | overrides))


def test_invalid_values_produce_typed_issue():
    dataframe = pd.DataFrame({"age": [-1, 25, 150, None]})

    issues = detect_validation_rule_issues(dataframe, [make_rule()])

    assert len(issues) == 1
    issue = issues[0]

    assert issue.issue_type == "invalid_value"
    assert issue.column == "age"
    assert "rule-001" in issue.evidence
    assert "Age must be between 0 and 120." in issue.evidence
    assert "Approved test specification" in issue.evidence
    assert "2 of 3 checked row(s)" in issue.evidence
    assert issue.allowed_treatments == [
        "set_missing",
        "drop_rows",
        "retain",
        "investigate",
    ]
    assert issue.requires_explicit_human_decision is True


def test_business_rule_violation_produces_typed_issue():
    dataframe = pd.DataFrame({
        "start": [1, 5, 10],
        "end": [2, 4, 10],
    })

    rule = DataValidationRule(
        rule_id="business-001",
        rule_type="column_comparison",
        issue_type="business_rule_violation",
        column="start",
        comparison_column="end",
        comparison_operator="<=",
        description="Start must not exceed end.",
        source="Approved business requirement",
    )

    issues = detect_validation_rule_issues(dataframe, [rule])

    assert len(issues) == 1
    assert issues[0].issue_type == "business_rule_violation"
    assert issues[0].column == "start"
    assert "business-001" in issues[0].evidence
    assert "1 of 3 checked row(s)" in issues[0].evidence
    assert issues[0].requires_explicit_human_decision is True


def test_rule_without_violations_produces_no_issue():
    dataframe = pd.DataFrame({"age": [0, 25, 120, None]})

    assert detect_validation_rule_issues(dataframe, [make_rule()]) == []


def test_empty_rule_collection_produces_no_issues():
    dataframe = pd.DataFrame({"age": [-1, 150]})

    assert detect_validation_rule_issues(dataframe, []) == []


def test_multiple_rules_produce_separate_issues():
    dataframe = pd.DataFrame({
        "age": [-1, 25],
        "status": ["active", "unknown"],
    })

    status_rule = DataValidationRule(
        rule_id="status-001",
        rule_type="allowed_values",
        issue_type="invalid_value",
        column="status",
        allowed_values=["active", "inactive"],
        description="Status must use an approved category.",
        source="Approved test specification",
    )

    issues = detect_validation_rule_issues(
        dataframe,
        [make_rule(), status_rule],
    )

    assert len(issues) == 2
    assert [issue.column for issue in issues] == ["age", "status"]


def test_same_column_rules_receive_distinct_stable_issue_ids():
    dataframe = pd.DataFrame({"age": [-5, 25, 150]})

    minimum_rule = make_rule(
        rule_id="age-minimum",
        min_value=0,
        max_value=None,
    )
    maximum_rule = make_rule(
        rule_id="age-maximum",
        min_value=None,
        max_value=120,
    )

    issues = detect_validation_rule_issues(
        dataframe,
        [minimum_rule, maximum_rule],
    )

    assert len(issues) == 2
    assert [issue.column for issue in issues] == ["age", "age"]
    assert [issue.issue_id for issue in issues] == [
        "rule:age-minimum",
        "rule:age-maximum",
    ]

    repeated = detect_validation_rule_issues(
        dataframe,
        [maximum_rule, minimum_rule],
    )

    assert {issue.issue_id for issue in repeated} == {
        "rule:age-minimum",
        "rule:age-maximum",
    }


def test_duplicate_rule_ids_are_rejected():
    dataframe = pd.DataFrame({"age": [-1, 25]})
    duplicate = make_rule(rule_id="rule-001", max_value=100)

    with pytest.raises(ValueError, match="Duplicate validation rule ID"):
        detect_validation_rule_issues(
            dataframe,
            [make_rule(), duplicate],
        )


def test_missing_column_error_is_propagated():
    dataframe = pd.DataFrame({"other": [1, 2]})

    with pytest.raises(KeyError, match="does not exist"):
        detect_validation_rule_issues(dataframe, [make_rule()])


def test_incompatible_type_error_is_propagated():
    dataframe = pd.DataFrame({"age": ["25", "150"]})

    with pytest.raises(TypeError, match="requires a numeric column"):
        detect_validation_rule_issues(dataframe, [make_rule()])


def test_adapter_does_not_modify_dataframe():
    dataframe = pd.DataFrame({"age": [-1, 25, 150, None]})
    original = dataframe.copy(deep=True)

    detect_validation_rule_issues(dataframe, [make_rule()])

    pd.testing.assert_frame_equal(dataframe, original)
