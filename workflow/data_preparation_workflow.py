from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from workflow.data_preparation_transitions import (
    evaluate_and_transition_data_preparation,
)
from workflow.states import WorkflowState


def apply_data_preparation_decision(
    project: ProjectState,
    decision: HITLDecision,
) -> ProjectState:
    """Apply a Data Preparation human decision to persisted project state."""

    if project.current_state != WorkflowState.AWAITING_PREPARATION_APPROVAL:
        raise ValueError(
            "Data Preparation decisions can only be applied while "
            "awaiting preparation approval."
        )

    if project.data_preparation_review is None:
        raise ValueError(
            "Cannot apply a Data Preparation decision without a review."
        )

    next_state = evaluate_and_transition_data_preparation(
        project.data_preparation_review,
        decision,
    )

    project.data_preparation_decision = decision

    if next_state == "modeling":
        project.current_state = WorkflowState.MODELING

    elif next_state == "revision":
        project.current_state = WorkflowState.DATA_PREPARATION
        project.data_preparation = None
        project.data_preparation_review = None

    else:
        project.current_state = WorkflowState.BLOCKED

    return project
