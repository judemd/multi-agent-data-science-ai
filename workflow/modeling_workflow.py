from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from workflow.modeling_transitions import (
    evaluate_and_transition_modeling,
)
from workflow.states import WorkflowState


def apply_modeling_decision(
    project: ProjectState,
    decision: HITLDecision,
) -> ProjectState:
    """Apply a Modeling human decision to persisted project state."""

    if project.current_state != WorkflowState.AWAITING_MODEL_SELECTION:
        raise ValueError(
            "Modeling decisions can only be applied while "
            "awaiting model selection."
        )

    if project.modeling_review is None:
        raise ValueError(
            "Cannot apply a Modeling decision without a review."
        )

    next_state = evaluate_and_transition_modeling(
        project.modeling_review,
        decision,
    )

    project.modeling_decision = decision

    if next_state == "evaluation":
        project.current_state = WorkflowState.EVALUATION

    elif next_state == "revision":
        project.current_state = WorkflowState.MODELING
        project.modeling = None
        project.modeling_review = None

    else:
        project.current_state = WorkflowState.BLOCKED

    return project
