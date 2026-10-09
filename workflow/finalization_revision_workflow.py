"""Human-controlled resolution of Finalization revision feedback."""

from domain.finalization_revision import FinalizationRevisionResolution
from domain.project_state import ProjectState
from workflow.states import WorkflowState


def resolve_finalization_revision(
    project: ProjectState,
    resolution: FinalizationRevisionResolution,
) -> ProjectState:
    """Record complete human-reviewed resolutions before rebuilding."""

    if project.current_state != WorkflowState.FINALIZATION:
        raise ValueError(
            "Finalization revisions can only be resolved in FINALIZATION."
        )

    decision = project.handoff_decision

    if decision is None or decision.decision != "request_revision":
        raise ValueError(
            "Finalization requires an outstanding human revision request."
        )

    if project.finalization is not None:
        raise ValueError(
            "Previous Finalization evidence must be invalidated."
        )

    if project.finalization_evidence_fingerprint is not None:
        raise ValueError(
            "Previous Finalization fingerprint must be invalidated."
        )

    if not resolution.reviewer.strip():
        raise ValueError("Revision resolution reviewer cannot be blank.")

    if not resolution.rationale.strip():
        raise ValueError("Revision resolution rationale cannot be blank.")

    original_feedback = [item.strip() for item in decision.feedback]

    if not original_feedback or any(not item for item in original_feedback):
        raise ValueError(
            "Outstanding Finalization revision feedback is invalid."
        )

    if len(original_feedback) != len(set(original_feedback)):
        raise ValueError(
            "Outstanding Finalization feedback contains duplicates."
        )

    resolved_feedback = [item.feedback for item in resolution.items]

    if set(resolved_feedback) != set(original_feedback):
        raise ValueError(
            "Every original Finalization feedback item must have "
            "exactly one matching resolution."
        )

    if any(item.status != "addressed" for item in resolution.items):
        raise ValueError(
            "All Finalization revision items must be addressed "
            "before rebuilding."
        )

    if any(not item.resolution.strip() for item in resolution.items):
        raise ValueError(
            "Finalization revision resolutions cannot be blank."
        )

    # All validation succeeds before persisted state is modified.
    project.handoff_decision_history.append(
        decision.model_copy(deep=True)
    )
    project.finalization_revision_resolution = resolution.model_copy(
        deep=True
    )
    project.finalization_revision_history.append(
        resolution.model_copy(deep=True)
    )
    project.handoff_decision = None

    return project