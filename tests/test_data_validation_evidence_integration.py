import pandas as pd
import pytest

from domain.data_preparation_issue import DataPreparationIssue

from domain.data_validation_rule import DataValidationRule
from tools.data_preparation_evidence import build_data_preparation_evidence


def make_rule(**overrides):
    parameters = {
        "rule_id": "age-range-001",
        "rule_type": "numeric_range",
        "issue_type": "invalid_value",
        "column": "age",
        "description": "Age must be between 0 and 120.",
        "source": "Approved test specification",
        "min_value": 0,
        "max_value": 120,
    }
    return DataValidationRule(**(parameters | overrides))


def test_ordinary_issue_ids_are_stable_when_evidence_counts_change():
    first = pd.DataFrame({"revenue": [10.0, None, 20.0]})
    second = pd.DataFrame({"revenue": [10.0, None, None, 20.0]})

    first_artifact = build_data_preparation_evidence(
        first,
        "customers.csv",
    )
    second_artifact = build_data_preparation_evidence(
        second,
        "customers.csv",
    )

    first_issue = next(
        issue for issue in first_artifact.issues
        if issue.issue_type == "missing_values"
        and issue.column == "revenue"
    )
    second_issue = next(
        issue for issue in second_artifact.issues
        if issue.issue_type == "missing_values"
        and issue.column == "revenue"
    )

    assert first_issue.issue_id == "missing_values:revenue"
    assert second_issue.issue_id == first_issue.issue_id


def test_aggregation_preserves_validation_rule_issue_id():
    dataframe = pd.DataFrame({"age": [-1, 25, 150]})

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
        approved_validation_rules=[make_rule()],
    )

    rule_issue = next(
        issue for issue in artifact.issues
        if issue.issue_type == "invalid_value"
    )

    assert rule_issue.issue_id == "rule:age-range-001"


def test_mixed_issue_inventory_has_unique_nonempty_ids():
    dataframe = pd.DataFrame({
        "age": [-1, 25, 150],
        "revenue": [10.0, None, 20.0],
    })

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
        approved_validation_rules=[make_rule()],
    )

    issue_ids = [issue.issue_id for issue in artifact.issues]

    assert issue_ids
    assert all(issue_ids)
    assert len(issue_ids) == len(set(issue_ids))


def test_duplicate_issue_ids_are_rejected(monkeypatch):
    def duplicate_detector(dataframe):
        return [
            DataPreparationIssue(
                issue_type="missing_values",
                column="revenue",
                evidence="First finding.",
                allowed_treatments=["retain"],
            ),
            DataPreparationIssue(
                issue_type="missing_values",
                column="revenue",
                evidence="Second finding.",
                allowed_treatments=["retain"],
            ),
        ]

    monkeypatch.setattr(
        "tools.data_preparation_evidence.detect_missing_value_issues",
        duplicate_detector,
    )

    dataframe = pd.DataFrame({"revenue": [10.0, None, 20.0]})

    with pytest.raises(
        ValueError,
        match="Duplicate data preparation issue ID",
    ):
        build_data_preparation_evidence(
            dataframe,
            "customers.csv",
        )


def test_existing_call_without_rules_remains_compatible():
    dataframe = pd.DataFrame({"age": [-1, 25, 150]})

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    assert not any(
        issue.issue_type in {"invalid_value", "business_rule_violation"}
        for issue in artifact.issues
    )


def test_empty_rule_collection_matches_default_behavior():
    dataframe = pd.DataFrame({"age": [-1, 25, 150]})

    default = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )
    explicit_empty = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
        approved_validation_rules=[],
    )

    assert default.model_dump() == explicit_empty.model_dump()


def test_approved_rule_violations_are_added_to_issues():
    dataframe = pd.DataFrame({"age": [-1, 25, 150, None]})

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
        approved_validation_rules=[make_rule()],
    )

    issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "invalid_value"
    ]

    assert len(issues) == 1
    assert issues[0].column == "age"
    assert "age-range-001" in issues[0].evidence
    assert "2 of 3 checked row(s)" in issues[0].evidence
    assert issues[0].requires_explicit_human_decision is True


def test_business_rule_violations_are_added_to_issues():
    dataframe = pd.DataFrame({
        "start": [1, 5, 10],
        "end": [2, 4, 10],
    })

    rule = DataValidationRule(
        rule_id="start-before-end-001",
        rule_type="column_comparison",
        issue_type="business_rule_violation",
        column="start",
        comparison_column="end",
        comparison_operator="<=",
        description="Start must not exceed end.",
        source="Approved business requirement",
    )

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
        approved_validation_rules=[rule],
    )

    issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "business_rule_violation"
    ]

    assert len(issues) == 1
    assert "start-before-end-001" in issues[0].evidence
    assert "1 of 3 checked row(s)" in issues[0].evidence
    assert issues[0].requires_explicit_human_decision is True


def test_rule_without_violations_adds_no_validation_issue():
    dataframe = pd.DataFrame({"age": [0, 25, 120]})

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
        approved_validation_rules=[make_rule()],
    )

    assert not any(
        issue.issue_type == "invalid_value"
        for issue in artifact.issues
    )


def test_invalid_rule_evaluation_error_is_propagated():
    dataframe = pd.DataFrame({"age": ["25", "150"]})

    with pytest.raises(TypeError, match="requires a numeric column"):
        build_data_preparation_evidence(
            dataframe,
            "customers.csv",
            approved_validation_rules=[make_rule()],
        )


def test_evidence_integration_preserves_dataframe():
    dataframe = pd.DataFrame({"age": [-1, 25, 150, None]})
    original = dataframe.copy(deep=True)

    build_data_preparation_evidence(
        dataframe,
        "customers.csv",
        approved_validation_rules=[make_rule()],
    )

    pd.testing.assert_frame_equal(dataframe, original)
