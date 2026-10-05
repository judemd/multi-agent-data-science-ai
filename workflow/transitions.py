from workflow.decisions import HumanDecision
from workflow.states import WorkflowState


class InvalidTransitionError(ValueError):
    """Raised when a decision is not valid for the current workflow state."""


TRANSITIONS: dict[
    tuple[WorkflowState, HumanDecision],
    WorkflowState,
] = {
    (
        WorkflowState.AWAITING_PROBLEM_APPROVAL,
        HumanDecision.APPROVE,
    ): WorkflowState.DATA_UNDERSTANDING,
    (
        WorkflowState.AWAITING_PROBLEM_APPROVAL,
        HumanDecision.REQUEST_CHANGES,
    ): WorkflowState.PROBLEM_FRAMING,
    (
        WorkflowState.AWAITING_PREPARATION_APPROVAL,
        HumanDecision.APPROVE,
    ): WorkflowState.MODELING,
    (
        WorkflowState.AWAITING_PREPARATION_APPROVAL,
        HumanDecision.REQUEST_CHANGES,
    ): WorkflowState.DATA_PREPARATION,
}


def get_next_state(
    current_state: WorkflowState,
    decision: HumanDecision,
) -> WorkflowState:
    """Return the next workflow state for an approved state/decision pair."""

    transition = (current_state, decision)

    if transition not in TRANSITIONS:
        raise InvalidTransitionError(
            f"Decision {decision.value} is not valid from state {current_state.value}."
        )

    return TRANSITIONS[transition]
