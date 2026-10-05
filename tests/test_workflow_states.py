from workflow.states import WorkflowState


def test_workflow_contains_expected_states():
    expected_states = {
        "PROBLEM_FRAMING",
        "AWAITING_PROBLEM_APPROVAL",
        "DATA_UNDERSTANDING",
        "AWAITING_DATA_APPROVAL",
        "DATA_PREPARATION",
        "AWAITING_PREPARATION_APPROVAL",
        "MODELING",
        "AWAITING_MODEL_SELECTION",
        "EVALUATION",
        "AWAITING_GO_NO_GO",
        "FINALIZATION",
        "AWAITING_HANDOFF_APPROVAL",
        "BLOCKED",
        "COMPLETE",
    }

    actual_states = {state.value for state in WorkflowState}

    assert actual_states == expected_states


def test_initial_workflow_state_is_problem_framing():
    assert WorkflowState.PROBLEM_FRAMING.value == "PROBLEM_FRAMING"


def test_complete_is_a_valid_workflow_state():
    assert WorkflowState.COMPLETE in WorkflowState
