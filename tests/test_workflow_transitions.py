import pytest

from workflow.decisions import HumanDecision
from workflow.states import WorkflowState
from workflow.transitions import (
    InvalidTransitionError,
    get_next_state,
)


def test_problem_approval_moves_to_data_understanding():
    assert (
        get_next_state(
            WorkflowState.AWAITING_PROBLEM_APPROVAL,
            HumanDecision.APPROVE,
        )
        == WorkflowState.DATA_UNDERSTANDING
    )


def test_problem_changes_return_to_problem_framing():
    assert (
        get_next_state(
            WorkflowState.AWAITING_PROBLEM_APPROVAL,
            HumanDecision.REQUEST_CHANGES,
        )
        == WorkflowState.PROBLEM_FRAMING
    )


def test_preparation_approval_moves_to_modeling():
    assert (
        get_next_state(
            WorkflowState.AWAITING_PREPARATION_APPROVAL,
            HumanDecision.APPROVE,
        )
        == WorkflowState.MODELING
    )


def test_preparation_changes_return_to_data_preparation():
    assert (
        get_next_state(
            WorkflowState.AWAITING_PREPARATION_APPROVAL,
            HumanDecision.REQUEST_CHANGES,
        )
        == WorkflowState.DATA_PREPARATION
    )


def test_invalid_transition_raises_error():
    with pytest.raises(InvalidTransitionError):
        get_next_state(
            WorkflowState.DATA_PREPARATION,
            HumanDecision.APPROVE,
        )
