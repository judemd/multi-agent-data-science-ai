"""Human-controlled approval of the Finalization handoff."""

from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.dataset_fingerprint import fingerprint_dataset_file
from tools.finalization_evidence_fingerprint import (
    fingerprint_finalization_evidence,
)
from workflow.states import WorkflowState


def apply_handoff_decision(
    project: ProjectState,
    decision: HITLDecision,
) -> ProjectState:
    """Apply a human handoff decision without authorizing deployment."""

    if project.current_state != WorkflowState.AWAITING_HANDOFF_APPROVAL:
        raise ValueError(
            "Handoff decisions require AWAITING_HANDOFF_APPROVAL."
        )

    if project.finalization is None:
        raise ValueError(
            "Handoff approval requires Finalization evidence."
        )

    if project.finalization.deployment_approved:
        raise ValueError(
            "Handoff cannot authorize deployment."
        )

    if not project.finalization.requires_human_handoff_approval:
        raise ValueError(
            "Finalization must require explicit human handoff approval."
        )

    if not project.finalization_evidence_fingerprint:
        raise ValueError(
            "Handoff requires a recorded Finalization evidence fingerprint."
        )

    current_evidence_fingerprint = fingerprint_finalization_evidence(
        project.finalization
    )

    if (
        current_evidence_fingerprint
        != project.finalization_evidence_fingerprint
    ):
        raise ValueError(
            "Handoff Finalization evidence fingerprint mismatch."
        )

    if not decision.reviewer.strip():
        raise ValueError("Handoff reviewer cannot be blank.")

    if not decision.rationale.strip():
        raise ValueError("Handoff rationale cannot be blank.")

    if decision.decision == "request_revision":
        if not any(item.strip() for item in decision.feedback):
            raise ValueError(
                "Handoff revision requires specific feedback."
            )

    if not project.dataset_path or not project.data_preparation_dataset_fingerprint:
        raise ValueError(
            "Handoff requires approved source dataset evidence."
        )

    if not project.prepared_dataset_path or not project.prepared_dataset_fingerprint:
        raise ValueError(
            "Handoff requires approved prepared dataset evidence."
        )

    try:
        source_fingerprint = fingerprint_dataset_file(project.dataset_path)
        prepared_fingerprint = fingerprint_dataset_file(
            project.prepared_dataset_path
        )
    except OSError as exc:
        raise ValueError(
            "Handoff cannot read the approved datasets."
        ) from exc

    if source_fingerprint != project.data_preparation_dataset_fingerprint:
        raise ValueError(
            "Handoff source dataset fingerprint mismatch."
        )

    if prepared_fingerprint != project.prepared_dataset_fingerprint:
        raise ValueError(
            "Handoff prepared dataset fingerprint mismatch."
        )

    if (
        project.finalization.prepared_dataset_fingerprint
        != project.prepared_dataset_fingerprint
    ):
        raise ValueError(
            "Handoff artifact prepared dataset fingerprint mismatch."
        )

    # All validation must succeed before changing persisted state.
    project.handoff_decision = decision.model_copy(deep=True)

    if decision.decision == "approve":
        project.current_state = WorkflowState.COMPLETE

    elif decision.decision == "request_revision":
        project.current_state = WorkflowState.FINALIZATION
        project.finalization = None
        project.finalization_evidence_fingerprint = None
        project.finalization_revision_resolution = None

    else:
        project.current_state = WorkflowState.BLOCKED

    return project