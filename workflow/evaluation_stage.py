"""Deterministic evaluation stage for human-approved modeling choices."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

from domain.evaluation import EvaluationArtifact, EvaluationMetrics
from domain.project_state import ProjectState

from tools.model_training import (
    evaluate_training_result,
    train_selected_model,
)
from workflow.states import WorkflowState


def run_evaluation_stage(project: ProjectState) -> ProjectState:
    """Evaluate the approved model using the approved prepared dataset."""

    if project.current_state != WorkflowState.EVALUATION:
        raise ValueError(
            "Evaluation can only run while the project is in EVALUATION."
        )

    if not project.prepared_dataset_path:
        raise ValueError(
            "Evaluation requires an approved prepared dataset path."
        )

    if project.modeling_decision is None or (
        project.modeling_decision.decision != "approve"
    ):
        raise ValueError(
            "Evaluation requires an approved Modeling decision."
        )

    if not project.selected_model:
        raise ValueError(
            "Evaluation requires a human-selected model."
        )

    if not project.selected_feature_columns:
        raise ValueError(
            "Evaluation requires explicitly approved feature columns."
        )

    if project.modeling is None:
        raise ValueError(
            "Evaluation requires Modeling evidence identifying the target."
        )

    dataset_path = Path(project.prepared_dataset_path)

    if not dataset_path.is_file():
        raise ValueError(
            f"Approved prepared dataset does not exist: {dataset_path}"
        )

    if not project.prepared_dataset_fingerprint:
        raise ValueError(
            "Evaluation requires a recorded prepared dataset fingerprint."
        )

    # Fingerprint and parse the same bytes, avoiding a second file read.
    dataset_bytes = dataset_path.read_bytes()
    current_fingerprint = sha256(dataset_bytes).hexdigest()

    if current_fingerprint != project.prepared_dataset_fingerprint:
        raise ValueError(
            "Prepared dataset fingerprint mismatch: the approved "
            "dataset has changed since Data Preparation."
        )

    dataframe = pd.read_csv(BytesIO(dataset_bytes))
    target_column = project.modeling.target_column

    trained = train_selected_model(
        dataframe=dataframe,
        target_column=target_column,
        selected_model=project.selected_model,
        feature_columns=project.selected_feature_columns,
    )

    result = evaluate_training_result(trained)

    excluded_columns = [
        column
        for column in dataframe.columns
        if column != target_column
        and column not in project.selected_feature_columns
    ]

    limitations = [
        "Evaluation uses one stratified 80/20 holdout split.",
        "No hyperparameter tuning or cross-validation was performed.",
        "The classification threshold was selected by maximizing F1 "
        "on one internal training-only validation split.",
        "Results do not establish suitability for deployment.",
    ]

    if result.model_metrics.recall < 0.5:
        limitations.append(
            "The selected model identifies fewer than half of actual "
            "positive cases at the default classification threshold."
        )

    if (
        result.model_metrics.accuracy
        <= result.baseline_metrics.accuracy
    ):
        limitations.append(
            "The selected model does not exceed baseline accuracy."
        )

    artifact = EvaluationArtifact(
        selected_model=result.selected_model,
        target_column=target_column,
        feature_columns=result.feature_columns,
        excluded_columns=excluded_columns,
        train_row_count=result.train_row_count,
        test_row_count=result.test_row_count,
        model_metrics=EvaluationMetrics(
            **vars(result.model_metrics)
        ),
        baseline_metrics=EvaluationMetrics(
            **vars(result.baseline_metrics)
        ),
        selected_threshold=result.selected_threshold,
        validation_f1=result.validation_f1,
        threshold_metrics=EvaluationMetrics(
            **vars(result.threshold_metrics)
        ),
        limitations=limitations,
    )

    # Persist evidence and advance only after all operations succeed.
    project.evaluation = artifact
    project.current_state = WorkflowState.AWAITING_GO_NO_GO

    return project