import pytest
from pydantic import ValidationError

from domain.data_preparation_issue import DataPreparationIssue


def test_data_preparation_issue_accepts_controlled_treatments():
    issue = DataPreparationIssue(
        issue_type="missing_values",
        column="revenue",
        evidence="Revenue contains 12 missing values.",
        allowed_treatments=[
            "median",
            "mean",
            "mode",
            "drop_rows",
            "retain",
        ],
    )

    assert issue.issue_type == "missing_values"
    assert issue.column == "revenue"
    assert "median" in issue.allowed_treatments
    assert issue.requires_explicit_human_decision is False


def test_governance_issue_can_require_explicit_human_decision():
    issue = DataPreparationIssue(
        issue_type="leakage_risk",
        column="account_closed_date",
        evidence=(
            "Column may contain information unavailable at prediction time."
        ),
        allowed_treatments=[
            "exclude_feature",
            "retain",
            "investigate",
        ],
        requires_explicit_human_decision=True,
    )

    assert issue.requires_explicit_human_decision is True


def test_data_preparation_issue_rejects_unknown_treatment():
    with pytest.raises(ValidationError):
        DataPreparationIssue(
            issue_type="missing_values",
            column="revenue",
            evidence="Revenue contains missing values.",
            allowed_treatments=[
                "make_it_better",
            ],
        )


def test_data_preparation_issue_requires_at_least_one_treatment():
    with pytest.raises(ValidationError):
        DataPreparationIssue(
            issue_type="missing_values",
            column="revenue",
            evidence="Revenue contains missing values.",
            allowed_treatments=[],
        )
