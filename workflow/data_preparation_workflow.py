from pathlib import Path

import pandas as pd

from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.data_preparation_executor import (
    execute_data_preparation_actions,
)
from workflow.data_preparation_transitions import (
    evaluate_and_transition_data_preparation,
)
from workflow.states import WorkflowState


def apply_data_preparation_decision(
    project: ProjectState,
    decision: HITLDecision,
    dataframe: pd.DataFrame | None = None,
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
        if dataframe is None:
            raise ValueError(
                "A dataframe is required before approved Data Preparation "
                "actions can be executed."
            )

        prepared_dataframe = execute_data_preparation_actions(
            dataframe,
            project.data_preparation_review.proposed_actions,
        )

        prepared_path = (
            Path("data/projects")
            / f"{project.project_id}_prepared.csv"
        )
        prepared_path.parent.mkdir(parents=True, exist_ok=True)

        prepared_dataframe.to_csv(
            prepared_path,
            index=False,
        )

        project.prepared_dataset_path = str(prepared_path)
        project.current_state = WorkflowState.MODELING

    elif next_state == "revision":
        project.current_state = WorkflowState.DATA_PREPARATION
        project.data_preparation = None
        project.data_preparation_review = None

    else:
        project.current_state = WorkflowState.BLOCKED

    return project