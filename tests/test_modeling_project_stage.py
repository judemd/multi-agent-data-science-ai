"""Tests for approved-dataset integrity at the project Modeling boundary."""

from hashlib import sha256

import pandas as pd
import pytest

from domain.modeling_review import ModelingReview, ProposedModel
from domain.project_state import ProjectState
from workflow.modeling_stage import run_modeling_stage_for_project
from workflow.states import WorkflowState


def build_project(tmp_path):
    dataset_path = tmp_path / "prepared.csv"
    dataset_bytes = (
        b"age,region,churn\n"
        b"21,North,0\n"
        b"35,South,1\n"
        b"42,North,0\n"
        b"53,South,1\n"
        b"29,East,0\n"
        b"61,West,1\n"
    )
    dataset_path.write_bytes(dataset_bytes)

    project = ProjectState(
        project_id="modeling-integrity-test",
        project_name="Modeling Integrity Test",
        current_state=WorkflowState.MODELING,
        prepared_dataset_path=str(dataset_path),
        prepared_dataset_fingerprint=sha256(dataset_bytes).hexdigest(),
    )

    return project, dataset_path


def fake_review():
    return ModelingReview(
        proposed_models=[
            ProposedModel(
                name="Logistic Regression",
                model_family="linear",
                rationale="Suitable binary classification candidate.",
            )
        ],
        recommended_model="Logistic Regression",
    )


def run_stage(project, dataframe=None):
    if dataframe is None:
        dataframe = pd.DataFrame(
            {"age": [99, 98], "churn": [0, 1]}
        )

    return run_modeling_stage_for_project(
        project=project,
        dataframe=dataframe,
        target_column="churn",
        file_name="prepared.csv",
    )


def test_uses_verified_csv_instead_of_caller_dataframe(
    tmp_path, monkeypatch
):
    project, _ = build_project(tmp_path)

    calls = []

    def mocked_agent(prompt):
        calls.append(prompt)
        return fake_review()

    monkeypatch.setattr(
        "workflow.modeling_stage.run_modeling_agent",
        mocked_agent,
    )

    result = run_stage(project)

    assert result.current_state == WorkflowState.AWAITING_MODEL_SELECTION
    assert result.modeling is not None
    assert result.modeling_review is not None
    assert result.modeling.feature_columns == ["age", "region"]
    assert len(calls) == 1


def test_missing_fingerprint_blocks_before_agent(
    tmp_path, monkeypatch
):
    project, _ = build_project(tmp_path)
    project.prepared_dataset_fingerprint = None

    def forbidden_agent(prompt):
        pytest.fail("Agent must not run without approved fingerprint.")

    monkeypatch.setattr(
        "workflow.modeling_stage.run_modeling_agent",
        forbidden_agent,
    )

    with pytest.raises(ValueError, match="recorded prepared dataset fingerprint"):
        run_stage(project)

    assert project.current_state == WorkflowState.MODELING
    assert project.modeling is None


def test_modified_prepared_csv_blocks_before_agent(
    tmp_path, monkeypatch
):
    project, dataset_path = build_project(tmp_path)
    dataset_path.write_bytes(
        dataset_path.read_bytes() + b"70,North,0\n"
    )

    def forbidden_agent(prompt):
        pytest.fail("Agent must not run on modified prepared dataset.")

    monkeypatch.setattr(
        "workflow.modeling_stage.run_modeling_agent",
        forbidden_agent,
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_stage(project)

    assert project.current_state == WorkflowState.MODELING
    assert project.modeling is None


def test_missing_prepared_file_blocks_before_agent(
    tmp_path, monkeypatch
):
    project, dataset_path = build_project(tmp_path)
    dataset_path.unlink()

    def forbidden_agent(prompt):
        pytest.fail("Agent must not run without prepared dataset.")

    monkeypatch.setattr(
        "workflow.modeling_stage.run_modeling_agent",
        forbidden_agent,
    )

    with pytest.raises(ValueError, match="cannot be read for Modeling"):
        run_stage(project)

    assert project.current_state == WorkflowState.MODELING


def test_incorrect_state_blocks_before_agent(
    tmp_path, monkeypatch
):
    project, _ = build_project(tmp_path)
    project.current_state = WorkflowState.AWAITING_MODEL_SELECTION

    def forbidden_agent(prompt):
        pytest.fail("Agent must not run in incorrect workflow state.")

    monkeypatch.setattr(
        "workflow.modeling_stage.run_modeling_agent",
        forbidden_agent,
    )

    with pytest.raises(ValueError, match="MODELING state"):
        run_stage(project)

    assert project.current_state == WorkflowState.AWAITING_MODEL_SELECTION


def test_missing_prepared_path_blocks_before_agent(
    tmp_path, monkeypatch
):
    project, _ = build_project(tmp_path)
    project.prepared_dataset_path = None

    def forbidden_agent(prompt):
        pytest.fail("Agent must not run without prepared dataset path.")

    monkeypatch.setattr(
        "workflow.modeling_stage.run_modeling_agent",
        forbidden_agent,
    )

    with pytest.raises(ValueError, match="approved prepared dataset path"):
        run_stage(project)

    assert project.current_state == WorkflowState.MODELING