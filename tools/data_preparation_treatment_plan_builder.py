"""Build a validated treatment plan from explicit human selections."""

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_treatment_decision import (
    DataPreparationTreatmentDecision,
)
from domain.data_preparation_treatment_plan import (
    DataPreparationTreatmentPlan,
)
from tools.data_preparation_treatment_plan_validator import (
    validate_data_preparation_treatment_plan,
)


def build_data_preparation_treatment_plan(
    artifact: DataPreparationArtifact,
    dataset_fingerprint: str,
    evidence_fingerprint: str,
    reviewer: str,
    selections: dict[str, tuple[str | None, str]],
) -> DataPreparationTreatmentPlan:
    """Require and validate one explicit human selection per issue."""

    if not reviewer.strip():
        raise ValueError("A treatment plan requires a reviewer.")

    decisions = []

    for issue in artifact.issues:
        issue_id = issue.issue_id

        if not issue_id:
            raise ValueError(
                "Every preparation issue must have a stable issue ID."
            )

        if issue_id not in selections:
            raise ValueError(
                f"Missing human treatment selection for issue '{issue_id}'."
            )

        treatment, rationale = selections[issue_id]

        if treatment is None:
            raise ValueError(
                f"Select a treatment for issue '{issue_id}'."
            )

        if not rationale.strip():
            raise ValueError(
                f"Enter a treatment rationale for issue '{issue_id}'."
            )

        decisions.append(
            DataPreparationTreatmentDecision(
                issue_id=issue_id,
                treatment=treatment,
                rationale=rationale.strip(),
            )
        )

    plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer=reviewer.strip(),
        decisions=decisions,
    )

    validate_data_preparation_treatment_plan(
        plan,
        artifact,
        dataset_fingerprint,
    )

    return plan
