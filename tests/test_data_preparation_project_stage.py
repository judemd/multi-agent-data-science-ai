import pandas as pd
import pytest

from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.project_state import ProjectState
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


def build_project() -> ProjectState:
    return ProjectState(
        project_id="project-001",
        project_name="Customer Churn",
        current_state=WorkflowState.DATA_PREPARATION,
    )


def test_project_stage_stores_artifact_and_review(monkeypatch):
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

    def fake_run_agent(prompt: str) -> DataPreparationReview:
        assert "DATA_PREPARATION_ARTIFACT" in prompt
        assert "revenue" in prompt
        return expected_review

    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        fake_run_agent,
    )

    project = run_data_preparation_stage_for_project(
        build_project(),
        build_dataframe(),
        "customers.csv",
    )

    assert project.current_state == WorkflowState.AWAITING_PREPARATION_APPROVAL
    assert project.data_preparation is not None
    assert project.data_preparation.file_name == "customers.csv"
    assert project.data_preparation_review == expected_review


def test_project_stage_rejects_invalid_state(monkeypatch):
    def fail_if_called(prompt: str) -> DataPreparationReview:
        raise AssertionError("Agent should not run from an invalid state.")

    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        fail_if_called,
    )

    project = build_project()
    project.current_state = WorkflowState.MODELING

    with pytest.raises(ValueError, match="DATA_PREPARATION"):
        run_data_preparation_stage_for_project(
            project,
            build_dataframe(),
            "customers.csv",
        )


def test_project_stage_does_not_modify_dataframe(monkeypatch):
    expected_review = DataPreparationReview(
        requires_human_review=True,
    )

    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        lambda prompt: expected_review,
    )

    dataframe = build_dataframe()
    before = dataframe.copy(deep=True)

    run_data_preparation_stage_for_project(
        build_project(),
        dataframe,
        "customers.csv",
    )

    pd.testing.assert_frame_equal(dataframe, before)


