import pandas as pd

from domain.data_preparation_issue import DataPreparationIssue
from domain.data_validation_rule import DataValidationRule
from tools.data_validation_rule_evaluator import evaluate_data_validation_rule


def detect_validation_rule_issues(
    dataframe: pd.DataFrame,
    rules: list[DataValidationRule],
) -> list[DataPreparationIssue]:
    """Convert violations of supplied rules into human-reviewed issues."""

    issues: list[DataPreparationIssue] = []
    seen_rule_ids: set[str] = set()

    for rule in rules:
        if rule.rule_id in seen_rule_ids:
            raise ValueError(
                f"Duplicate validation rule ID: '{rule.rule_id}'."
            )

        seen_rule_ids.add(rule.rule_id)

        result = evaluate_data_validation_rule(dataframe, rule)

        if result["violation_count"] == 0:
            continue

        issues.append(
            DataPreparationIssue(
                issue_id=f"rule:{rule.rule_id}",
                issue_type=rule.issue_type,
                column=rule.column,
                evidence=(
                    f"Validation rule '{rule.rule_id}' "
                    f"({rule.description}) from source '{rule.source}' "
                    f"was violated by {result['violation_count']} "
                    f"of {result['checked_count']} checked row(s)."
                ),
                allowed_treatments=[
                    "set_missing",
                    "drop_rows",
                    "retain",
                    "investigate",
                ],
                requires_explicit_human_decision=True,
            )
        )

    return issues
