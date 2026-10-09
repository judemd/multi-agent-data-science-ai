"""Tests for deterministic Evaluation stage execution."""

import pandas as pd
import pytest

from domain.hitl_decision import HITLDecision
from domain.modeling import ModelingArtifact
from domain.project_state import ProjectState
from tools.dataset_fingerprint import fingerprint_dataset_file
from workflow.evaluation_stage import run_evaluation_stage
from workflow.states import WorkflowState


def build_project(tmp_path) -> ProjectState:
    dataframe = pd.DataFrame(
        {
            "age": [20 + i for i in range(40)],
            "region": ["north", "south"] * 20,
            "churn": [0, 1] * 20,
        }
    )

    dataset_path = tmp_path / "prepared.csv"
    dataframe.to_csv(dataset_path, index=False)

    return ProjectState(
        project_id="evaluation-test",
        project_name="Evaluation Test",
        current_state=WorkflowState.EVALUATION,
        prepared_dataset_path=str(dataset_path),
        prepared_dataset_fingerprint=fingerprint_dataset_file(dataset_path),
        modeling=ModelingArtifact(
            file_name="prepared.csv",
            row_count=40,
            column_count=3,
            target_column="churn",
            target_dtype="int64",
            target_missing_count=0,
            target_unique_count=2,
            target_distribution={"0": 20, "1": 20},
            feature_columns=["age", "region"],
        ),
        modeling_decision=HITLDecision(
            decision="approve",
            reviewer="test_reviewer",
            rationale="Approved for testing.",
        ),
        selected_model="Logistic Regression",
        selected_feature_columns=["age", "region"],
    )


def test_evaluation_produces_evidence_and_advances_state(tmp_path):
    project = run_evaluation_stage(build_project(tmp_path))

    assert project.current_state == WorkflowState.AWAITING_GO_NO_GO
    assert project.evaluation is not None
    assert project.evaluation.selected_model == "Logistic Regression"
    assert project.evaluation.target_column == "churn"
    assert project.evaluation.feature_columns == ["age", "region"]
    assert project.evaluation.train_row_count == 32
    assert project.evaluation.test_row_count == 8
    assert project.evaluation.limitations

    assert 0 <= project.evaluation.model_metrics.roc_auc <= 1
    assert 0 <= project.evaluation.baseline_metrics.accuracy <= 1

    # Threshold selection uses internal training-only validation.
    assert project.evaluation.selected_threshold is not None
    assert 0.0 <= project.evaluation.selected_threshold <= 1.0
    assert project.evaluation.validation_f1 is not None
    assert 0.0 <= project.evaluation.validation_f1 <= 1.0
    assert project.evaluation.threshold_metrics is not None

    # Both threshold choices use the same held-out probability scores.
    assert project.evaluation.threshold_metrics.roc_auc == pytest.approx(
        project.evaluation.model_metrics.roc_auc
    )

    assert any(
        "internal training-only validation split" in limitation
        for limitation in project.evaluation.limitations
    )


def test_evaluation_uses_only_approved_features(tmp_path):
    project = build_project(tmp_path)
    project.selected_feature_columns = ["age"]

    result = run_evaluation_stage(project)

    assert result.evaluation.feature_columns == ["age"]
    assert result.evaluation.excluded_columns == ["region"]


@pytest.mark.parametrize(
    "state",
    [
        WorkflowState.MODELING,
        WorkflowState.AWAITING_MODEL_SELECTION,
        WorkflowState.AWAITING_GO_NO_GO,
    ],
)
def test_evaluation_rejects_incorrect_state(tmp_path, state):
    project = build_project(tmp_path)
    project.current_state = state

    with pytest.raises(ValueError, match="only run"):
        run_evaluation_stage(project)

    assert project.evaluation is None
    assert project.current_state == state


def test_evaluation_requires_prepared_dataset(tmp_path):
    project = build_project(tmp_path)
    project.prepared_dataset_path = None

    with pytest.raises(ValueError, match="prepared dataset path"):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None


def test_evaluation_rejects_missing_prepared_file(tmp_path):
    project = build_project(tmp_path)
    project.prepared_dataset_path = str(tmp_path / "missing.csv")

    with pytest.raises(ValueError, match="does not exist"):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None


def test_evaluation_requires_approved_modeling_decision(tmp_path):
    project = build_project(tmp_path)
    project.modeling_decision = HITLDecision(
        decision="reject",
        reviewer="test_reviewer",
        rationale="Not approved.",
    )

    with pytest.raises(ValueError, match="approved Modeling decision"):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None


@pytest.mark.parametrize(
    "model, features, error",
    [
        (None, ["age"], "human-selected model"),
        ("Logistic Regression", None, "approved feature columns"),
        ("Logistic Regression", [], "approved feature columns"),
    ],
)
def test_evaluation_requires_model_and_features(
    tmp_path, model, features, error
):
    project = build_project(tmp_path)
    project.selected_model = model
    project.selected_feature_columns = features

    with pytest.raises(ValueError, match=error):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None


def test_evaluation_requires_modeling_evidence(tmp_path):
    project = build_project(tmp_path)
    project.modeling = None

    with pytest.raises(ValueError, match="Modeling evidence"):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None


def test_evaluation_rejects_unknown_approved_feature(tmp_path):
    project = build_project(tmp_path)
    project.selected_feature_columns = ["unknown_column"]

    with pytest.raises(ValueError, match="Unknown approved feature"):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None


def test_evaluation_artifact_survives_json_roundtrip(tmp_path):
    project = run_evaluation_stage(build_project(tmp_path))

    restored = ProjectState.model_validate_json(project.model_dump_json())

    assert restored.current_state == WorkflowState.AWAITING_GO_NO_GO
    assert restored.evaluation == project.evaluation


def test_evaluation_rejects_missing_prepared_fingerprint(tmp_path):
    project = build_project(tmp_path)
    project.prepared_dataset_fingerprint = None

    with pytest.raises(ValueError, match="recorded prepared dataset fingerprint"):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None


def test_evaluation_rejects_tampered_prepared_dataset(tmp_path):
    project = build_project(tmp_path)

    dataset_path = tmp_path / "prepared.csv"
    dataframe = pd.read_csv(dataset_path)
    dataframe.loc[0, "age"] = 999
    dataframe.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Prepared dataset fingerprint mismatch"):
        run_evaluation_stage(project)

    assert project.current_state == WorkflowState.EVALUATION
    assert project.evaluation is None