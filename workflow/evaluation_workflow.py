"""Human-controlled transitions after deterministic Evaluation."""

from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from workflow.states import WorkflowState


def apply_evaluation_decision(
    project: ProjectState,
    decision: HITLDecision,
) -> ProjectState:
    """Apply a human GO, ITERATE, or NO-GO decision."""

    if project.current_state != WorkflowState.AWAITING_GO_NO_GO:
        raise ValueError(
            "Evaluation decisions can only be applied while "
            "awaiting GO / NO-GO review."
        )

    if project.evaluation is None:
        raise ValueError(
            "Evaluation evidence is required before a human decision."
        )

    if not decision.reviewer.strip():
        raise ValueError("Evaluation reviewer cannot be blank.")

    if not decision.rationale.strip():
        raise ValueError("Evaluation decision rationale cannot be blank.")

    if decision.decision == "request_revision":
        if not any(item.strip() for item in decision.feedback):
            raise ValueError(
                "ITERATE requires at least one specific revision request."
            )

    # Validate before mutating project state.
    project.evaluation_history.append(
        project.evaluation.model_copy(deep=True)
    )
    project.evaluation_decision_history.append(
        decision.model_copy(deep=True)
    )
    project.evaluation_decision = decision.model_copy(deep=True)

    if decision.decision == "approve":
        project.current_state = WorkflowState.FINALIZATION

    elif decision.decision == "request_revision":
        project.current_state = WorkflowState.MODELING
        project.revision += 1

        # Preserve prior evidence in history, but prevent stale approvals
        # and artifacts from being reused in the next Modeling cycle.
        project.evaluation = None
        project.modeling = None
        project.modeling_review = None
        project.modeling_decision = None
        project.selected_model = None
        project.selected_feature_columns = None

    else:
        project.current_state = WorkflowState.BLOCKED

    return project