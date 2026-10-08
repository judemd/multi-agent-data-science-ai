import pandas as pd
import pytest

from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.project_state import ProjectState
from tools.dataset_fingerprint import fingerprint_dataset_file
from tools.preparation_evidence_fingerprint import fingerprint_preparation_evidence
from workflow.data_preparation_stage import (
    run_data_preparation_stage_for_project,
)
from workflow.states import WorkflowState


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "revenue": [100.0, None, 300.0],
            "segment": ["Enterprise", "enterprise ", "SMB"],
        }
    )


def build_project(tmp_path) -> ProjectState:
    dataset_path = tmp_path / "customers.csv"
    build_dataframe().to_csv(dataset_path, index=False)

    return ProjectState(
        project_id="project-001",
        project_name="Customer Churn",
        dataset_path=str(dataset_path),
        current_state=WorkflowState.DATA_PREPARATION,
    )


def test_project_stage_stores_artifact_review_and_fingerprints(
    monkeypatch,
    tmp_path,
):
    expected_review = DataPreparationReview(
        observed_evidence=[
            "Revenue contains one missing value.",
        ],
        proposed_actions=[
            DataPreparationAction(
                operation="impute_missing",
                column="revenue",
                strategy="median",
                reason="Investigate the appropriate missing-value treatment.",
            ),
        ],
        requires_human_review=True,
    )

    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        lambda prompt: expected_review,
    )

    project = build_project(tmp_path)

    result = run_data_preparation_stage_for_project(
        project,
        build_dataframe(),
        "customers.csv",
    )

    assert result.current_state == WorkflowState.AWAITING_PREPARATION_APPROVAL
    assert result.data_preparation is not None
    assert result.data_preparation.file_name == "customers.csv"
    assert result.data_preparation_review == expected_review
    assert result.data_preparation_dataset_fingerprint == (
        fingerprint_dataset_file(project.dataset_path)
    )
    assert result.data_preparation_evidence_fingerprint == (
        fingerprint_preparation_evidence(result.data_preparation)
    )


def test_project_stage_rejects_invalid_state(monkeypatch, tmp_path):
    def fail_if_called(prompt: str) -> DataPreparationReview:
        raise AssertionError("Agent should not run from an invalid state.")

    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        fail_if_called,
    )

    project = build_project(tmp_path)
    project.current_state = WorkflowState.MODELING

    with pytest.raises(ValueError, match="DATA_PREPARATION"):
        run_data_preparation_stage_for_project(
            project,
            build_dataframe(),
            "customers.csv",
        )


def test_project_stage_does_not_modify_dataframe(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        lambda prompt: DataPreparationReview(requires_human_review=True),
    )

    dataframe = build_dataframe()
    before = dataframe.copy(deep=True)

    run_data_preparation_stage_for_project(
        build_project(tmp_path),
        dataframe,
        "customers.csv",
    )

    pd.testing.assert_frame_equal(dataframe, before)


def test_project_stage_rejects_missing_dataset_path(monkeypatch):
    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        lambda prompt: pytest.fail("Agent must not run."),
    )

    project = ProjectState(
        project_id="missing-dataset",
        project_name="Missing Dataset",
        current_state=WorkflowState.DATA_PREPARATION,
    )

    with pytest.raises(ValueError, match="persisted project dataset"):
        run_data_preparation_stage_for_project(
            project,
            build_dataframe(),
            "customers.csv",
        )

    assert project.current_state == WorkflowState.DATA_PREPARATION
    assert project.data_preparation is None
    assert project.data_preparation_dataset_fingerprint is None


def test_project_stage_rejects_mismatched_dataframe(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        lambda prompt: pytest.fail("Agent must not run."),
    )

    project = build_project(tmp_path)
    different_dataframe = build_dataframe()
    different_dataframe.loc[0, "revenue"] = 999.0

    with pytest.raises(ValueError, match="does not match"):
        run_data_preparation_stage_for_project(
            project,
            different_dataframe,
            "customers.csv",
        )

    assert project.current_state == WorkflowState.DATA_PREPARATION
    assert project.data_preparation is None
    assert project.data_preparation_review is None
    assert project.data_preparation_dataset_fingerprint is None
    assert project.data_preparation_evidence_fingerprint is None


def test_project_stage_rejects_dataset_changed_during_review(
    monkeypatch,
    tmp_path,
):
    project = build_project(tmp_path)

    def change_dataset_during_agent_review(prompt: str):
        with open(project.dataset_path, "ab") as dataset_file:
            dataset_file.write(b"\n")
        return DataPreparationReview(requires_human_review=True)

    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        change_dataset_during_agent_review,
    )

    with pytest.raises(ValueError, match="changed during Data Preparation"):
        run_data_preparation_stage_for_project(
            project,
            build_dataframe(),
            "customers.csv",
        )

    assert project.current_state == WorkflowState.DATA_PREPARATION
    assert project.data_preparation is None
    assert project.data_preparation_review is None
    assert project.data_preparation_dataset_fingerprint is None
    assert project.data_preparation_evidence_fingerprint is None
