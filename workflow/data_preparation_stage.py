import pandas as pd

from agents.data_preparation_runner import run_data_preparation_agent
from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_review import DataPreparationReview
from domain.project_state import ProjectState
from tools.data_preparation_evidence import build_data_preparation_evidence
from tools.data_preparation_prompt import build_data_preparation_prompt
from tools.data_loader import load_dataset
from tools.dataset_fingerprint import fingerprint_dataset_file
from tools.preparation_evidence_fingerprint import fingerprint_preparation_evidence
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

    if not project.dataset_path:
        raise ValueError(
            "Data Preparation requires a persisted project dataset."
        )

    dataset_fingerprint = fingerprint_dataset_file(project.dataset_path)
    persisted_dataframe = load_dataset(project.dataset_path)

    try:
        pd.testing.assert_frame_equal(
            dataframe,
            persisted_dataframe,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError(
            "The supplied DataFrame does not match the persisted project dataset."
        ) from exc

    artifact = build_data_preparation_stage_artifact(
        persisted_dataframe,
        file_name,
    )

    evidence_fingerprint = fingerprint_preparation_evidence(artifact)
    prompt = build_data_preparation_prompt(artifact)
    review = run_data_preparation_agent(prompt)

    if fingerprint_dataset_file(project.dataset_path) != dataset_fingerprint:
        raise ValueError(
            "The project dataset changed during Data Preparation review."
        )

    project.data_preparation = artifact
    project.data_preparation_review = review
    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint
    project.current_state = WorkflowState.AWAITING_PREPARATION_APPROVAL

    return project
