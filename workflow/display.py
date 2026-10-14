from workflow.states import WorkflowState


def get_current_stage(state: WorkflowState) -> str | None:
    """Return the workflow stage corresponding to a project state."""

    stage_by_state = {
        WorkflowState.PROBLEM_FRAMING: "Problem Framing",
        WorkflowState.AWAITING_PROBLEM_APPROVAL: "Problem Framing",
        WorkflowState.DATA_UNDERSTANDING: "Data Understanding",
        WorkflowState.AWAITING_DATA_APPROVAL: "Data Understanding",
        WorkflowState.DATA_PREPARATION: "Data Preparation",
        WorkflowState.AWAITING_PREPARATION_APPROVAL: "Data Preparation",
        WorkflowState.MODELING: "Modeling",
        WorkflowState.AWAITING_MODEL_SELECTION: "Modeling",
        WorkflowState.EVALUATION: "Evaluation",
        WorkflowState.AWAITING_GO_NO_GO: "Evaluation",
        WorkflowState.FINALIZATION: "Final Recommendation",
        WorkflowState.AWAITING_HANDOFF_APPROVAL: "Final Recommendation",
    }

    return stage_by_state.get(state)
