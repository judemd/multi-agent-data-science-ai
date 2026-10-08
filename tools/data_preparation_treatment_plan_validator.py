"""Validate human treatment decisions against reviewed preparation evidence."""

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_treatment_plan import DataPreparationTreatmentPlan
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)


def validate_data_preparation_treatment_plan(
    plan: DataPreparationTreatmentPlan,
    artifact: DataPreparationArtifact,
    dataset_fingerprint: str,
) -> None:
    """Reject stale, incomplete, or impermissible human treatment plans."""

    if plan.dataset_fingerprint != dataset_fingerprint:
        raise ValueError(
            "Treatment plan dataset fingerprint does not match "
            "the reviewed dataset."
        )

    current_evidence_fingerprint = fingerprint_preparation_evidence(
        artifact
    )

    if plan.evidence_fingerprint != current_evidence_fingerprint:
        raise ValueError(
            "Treatment plan evidence fingerprint does not match "
            "the reviewed preparation evidence."
        )

    issues_by_id = {}

    for issue in artifact.issues:
        if not issue.issue_id:
            raise ValueError(
                "Every preparation issue must have a stable issue ID."
            )

        if issue.issue_id in issues_by_id:
            raise ValueError(
                f"Duplicate preparation issue ID: '{issue.issue_id}'."
            )

        issues_by_id[issue.issue_id] = issue

    decisions_by_id = {
        decision.issue_id: decision
        for decision in plan.decisions
    }

    unknown_ids = set(decisions_by_id) - set(issues_by_id)

    if unknown_ids:
        raise ValueError(
            "Treatment plan references unknown issue IDs: "
            + ", ".join(sorted(unknown_ids))
        )

    missing_ids = set(issues_by_id) - set(decisions_by_id)

    if missing_ids:
        raise ValueError(
            "Treatment plan is missing decisions for issue IDs: "
            + ", ".join(sorted(missing_ids))
        )

    for issue_id, decision in decisions_by_id.items():
        issue = issues_by_id[issue_id]

        if decision.treatment not in issue.allowed_treatments:
            raise ValueError(
                f"Treatment '{decision.treatment}' is not permitted "
                f"for issue '{issue_id}'."
            )
