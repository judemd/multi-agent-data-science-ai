from workflow.states import WorkflowState
from workflow.display import get_current_stage


def test_awaiting_data_approval_displays_data_understanding():
    assert get_current_stage(
        WorkflowState.AWAITING_DATA_APPROVAL
    ) == "Data Understanding"


def test_blocked_state_has_no_current_stage():
    assert get_current_stage(WorkflowState.BLOCKED) is None


def test_complete_state_has_no_current_stage():
    assert get_current_stage(WorkflowState.COMPLETE) is None



def test_every_workflow_state_has_a_defined_display_result():
    valid_stages = {
        "Problem Framing",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Final Recommendation",
        None,
    }

    for state in WorkflowState:
        assert get_current_stage(state) in valid_stages
