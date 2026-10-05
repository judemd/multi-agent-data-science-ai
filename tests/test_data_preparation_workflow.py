import pytest

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from workflow.data_preparation_workflow import (
    apply_data_preparation_decision,
)
from workflow.states import WorkflowState


def build_project() -> ProjectState:
    return ProjectState(
        project_id="project-004",
        project_name="Customer Churn",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        data_preparation_review=DataPreparationReview(
            observed_evidence=[
                "Revenue contains missing values.",
            ],
            proposed_actions=[
            DataPreparationAction(
                operation="impute_missing",
                column="revenue",
                strategy="median",
                reason="Investigate the appropriate missing-value treatment.",
            ),
        ],
            requires_human_review=False,
        ),
    )


def build_decision(decision: str) -> HITLDecision:
    return HITLDecision(
        decision=decision,
        reviewer="data_scientist",
        rationale="Decision based on the preparation review.",
    )


def test_approval_moves_project_to_modeling():
    project = apply_data_preparation_decision(
        build_project(),
        build_decision("approve"),
    )

    assert project.current_state == WorkflowState.MODELING


def test_revision_returns_project_to_data_preparation():
    project = apply_data_preparation_decision(
        build_project(),
        build_decision("request_revision"),
    )

    assert project.current_state == WorkflowState.DATA_PREPARATION


def test_rejection_moves_project_to_blocked():
    project = apply_data_preparation_decision(
        build_project(),
        build_decision("reject"),
    )

    assert project.current_state == WorkflowState.BLOCKED


def test_decision_requires_existing_review():
    project = build_project()
    project.data_preparation_review = None

    with pytest.raises(
        ValueError,
        match="without a review",
    ):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
        )


def test_decision_requires_approval_state():
    project = build_project()
    project.current_state = WorkflowState.DATA_PREPARATION

    with pytest.raises(
        ValueError,
        match="awaiting preparation approval",
    ):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
        )


def test_review_requiring_human_review_can_progress_after_approval():
    project = build_project()
    project.data_preparation_review.requires_human_review = True

    project = apply_data_preparation_decision(
        project,
        build_decision("approve"),
    )

    assert project.current_state == WorkflowState.MODELING
    
def test_revision_clears_previous_data_preparation_review():
    project = build_project()

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
    )

    project.data_preparation_review = DataPreparationReview(
        observed_evidence=["Previous evidence"],
        requires_human_review=True,
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("request_revision"),
    )

    assert updated.current_state == WorkflowState.DATA_PREPARATION
    assert updated.data_preparation is None
    assert updated.data_preparation_review is None


