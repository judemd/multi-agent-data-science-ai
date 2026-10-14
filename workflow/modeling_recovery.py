"""Controlled recovery of interrupted Evaluation before results exist."""

from domain.project_state import ProjectState
from workflow.states import WorkflowState


def recover_interrupted_evaluation(
    project: ProjectState,
    reason: str,
) -> ProjectState:
    """Supersede an approval and reopen explicit human Modeling review."""

    if project.current_state != WorkflowState.EVALUATION:
        raise ValueError("Recovery requires the EVALUATION state.")

    if project.evaluation is not None:
        raise ValueError("Cannot recover after Evaluation evidence exists.")

    if not reason.strip():
        raise ValueError("A recovery reason is required.")

    if (
        project.modeling is None
        or project.modeling_review is None
        or project.modeling_decision is None
        or project.modeling_decision.decision != "approve"
        or project.selected_model is None
        or not project.selected_feature_columns
    ):
        raise ValueError("A complete prior Modeling approval is required.")

    record = {
        "decision": project.modeling_decision.model_dump(mode="json"),
        "selected_model": project.selected_model,
        "selected_feature_columns": list(project.selected_feature_columns),
        "recovery_reason": reason.strip(),
    }

    project.modeling_approval_history.append(record)
    project.modeling_decision = None
    project.selected_model = None
    project.selected_feature_columns = None
    project.current_state = WorkflowState.AWAITING_MODEL_SELECTION

    return project
