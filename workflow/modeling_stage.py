from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

from agents.modeling_runner import run_modeling_agent
from domain.modeling import ModelingArtifact
from domain.modeling_review import ModelingReview
from domain.project_state import ProjectState

from tools.modeling_evidence import build_modeling_evidence
from tools.modeling_prompt import build_modeling_prompt
from tools.modeling_sanity_check import check_modeling_dataset
from workflow.states import WorkflowState


def prepare_modeling_stage(
    dataframe: pd.DataFrame,
    target_column: str,
    file_name: str,
) -> str:
    """Prepare deterministic evidence for the Modeling Agent."""

    artifact = build_modeling_evidence(
        dataframe,
        target_column,
        file_name,
    )

    return build_modeling_prompt(artifact)


def build_modeling_stage_artifact(
    dataframe: pd.DataFrame,
    target_column: str,
    file_name: str,
) -> ModelingArtifact:
    """Build deterministic modeling evidence without modifying the input."""

    return build_modeling_evidence(
        dataframe,
        target_column,
        file_name,
    )


def run_modeling_stage(
    dataframe: pd.DataFrame,
    target_column: str,
    file_name: str,
) -> ModelingReview:
    """Build deterministic evidence and obtain the agent's modeling review."""

    artifact = build_modeling_stage_artifact(
        dataframe,
        target_column,
        file_name,
    )

    prompt = build_modeling_prompt(artifact)

    return run_modeling_agent(prompt)


def run_modeling_stage_for_project(
    project: ProjectState,
    dataframe: pd.DataFrame,
    target_column: str,
    file_name: str,
) -> ProjectState:
    """Run Modeling and return the updated project state."""

    if project.current_state != WorkflowState.MODELING:
        raise ValueError(
            "Modeling can only run from the MODELING state."
        )

    if not project.prepared_dataset_path:
        raise ValueError(
            "Modeling requires an approved prepared dataset path."
        )

    if not project.prepared_dataset_fingerprint:
        raise ValueError(
            "Modeling requires a recorded prepared dataset fingerprint."
        )

    try:
        dataset_bytes = Path(project.prepared_dataset_path).read_bytes()
    except OSError as exc:
        raise ValueError(
            "The approved prepared dataset cannot be read for Modeling."
        ) from exc

    current_fingerprint = sha256(dataset_bytes).hexdigest()

    if current_fingerprint != project.prepared_dataset_fingerprint:
        raise ValueError(
            "Prepared dataset fingerprint mismatch: the approved "
            "dataset has changed since Data Preparation."
        )

    try:
        dataframe = pd.read_csv(BytesIO(dataset_bytes))
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError(
            "The approved prepared dataset cannot be parsed for Modeling."
        ) from exc

    sanity = check_modeling_dataset(dataframe, target_column)

    if sanity.status == "BLOCK":
        raise ValueError(
            "Modeling sanity check failed: " + "; ".join(sanity.errors)
        )

    for warning in sanity.warnings:
        print(f"Modeling sanity warning: {warning}")

    artifact = build_modeling_stage_artifact(
        dataframe,
        target_column,
        file_name,
    )

    prompt = build_modeling_prompt(artifact)

    review = run_modeling_agent(prompt)

    project.modeling = artifact
    project.modeling_review = review
    project.current_state = WorkflowState.AWAITING_MODEL_SELECTION

    return project
