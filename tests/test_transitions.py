import pytest

from workflow.decisions import HumanDecision
from workflow.states import WorkflowState
from workflow.transitions import InvalidTransitionError, get_next_state


def test_problem_approval_moves_to_data_understanding():
    next_state = get_next_state(
        WorkflowState.AWAITING_PROBLEM_APPROVAL,
        HumanDecision.APPROVE,
    )

    assert next_state == WorkflowState.DATA_UNDERSTANDING


def test_problem_request_changes_returns_to_problem_framing():
    next_state = get_next_state(
        WorkflowState.AWAITING_PROBLEM_APPROVAL,
        HumanDecision.REQUEST_CHANGES,
    )

    assert next_state == WorkflowState.PROBLEM_FRAMING


def test_invalid_decision_cannot_bypass_problem_approval():
    with pytest.raises(InvalidTransitionError):
        get_next_state(
            WorkflowState.AWAITING_PROBLEM_APPROVAL,
            HumanDecision.GO,
        )
