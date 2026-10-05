import pandas as pd

from agents.data_preparation_runner import run_data_preparation_agent
from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_review import DataPreparationReview
from domain.project_state import ProjectState
from tools.data_preparation_evidence import build_data_preparation_evidence
from tools.data_preparation_prompt import build_data_preparation_prompt
from workflow.states import WorkflowState


def prepare_data_preparation_stage(
    dataframe: pd.DataFrame,
    file_name: str,
) -> str:
    """Prepare deterministic evidence for the Data Preparation Agent."""

    artifact = build_data_preparation_evidence(
        dataframe,
        file_name,
    )

    return build_data_preparation_prompt(artifact)


def build_data_preparation_stage_artifact(
    dataframe: pd.DataFrame,
    file_name: str,
) -> DataPreparationArtifact:
    """Build deterministic preparation evidence without modifying the input."""

    return build_data_preparation_evidence(
        dataframe,
        file_name,
    )


def run_data_preparation_stage(
    dataframe: pd.DataFrame,
    file_name: str,
) -> DataPreparationReview:
    """Build deterministic evidence and obtain the agent's preparation review."""

    artifact = build_data_preparation_stage_artifact(
        dataframe,
        file_name,
    )

    prompt = build_data_preparation_prompt(artifact)

    return run_data_preparation_agent(prompt)


def run_data_preparation_stage_for_project(
    project: ProjectState,
    dataframe: pd.DataFrame,
    file_name: str,
) -> ProjectState:
    """Run Data Preparation and return the updated project state."""

    if project.current_state != WorkflowState.DATA_PREPARATION:
        raise ValueError(
            "Data Preparation can only run from the DATA_PREPARATION state."
        )

    artifact = build_data_preparation_stage_artifact(
        dataframe,
        file_name,
    )

    prompt = build_data_preparation_prompt(artifact)

    review = run_data_preparation_agent(prompt)

    project.data_preparation = artifact
    project.data_preparation_review = review
    project.current_state = WorkflowState.AWAITING_PREPARATION_APPROVAL

    return project
