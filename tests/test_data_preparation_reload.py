import pytest

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_review import DataPreparationReview
from domain.project_state import ProjectState
from tools.project_store import ProjectStore
from workflow.data_preparation_reload import (
    reload_data_preparation_project,
)
from workflow.states import WorkflowState


def build_project() -> ProjectState:
    return ProjectState(
        project_id="project-003",
        project_name="Customer Churn",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        data_preparation=DataPreparationArtifact(
            file_name="customers.csv",
            row_count=100,
            column_count=3,
        ),
        data_preparation_review=DataPreparationReview(
            observed_evidence=[
                "Revenue contains missing values.",
            ],
            requires_human_review=True,
        ),
    )


def test_reload_returns_existing_preparation_state(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))
    original = build_project()

    store.save(original)

    loaded = reload_data_preparation_project(
        "project-003",
        store=store,
    )

    assert loaded == original
    assert loaded.data_preparation is not None
    assert loaded.data_preparation_review is not None


def test_reload_rejects_project_not_awaiting_approval(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))

    project = build_project()
    project.current_state = WorkflowState.DATA_PREPARATION
    store.save(project)

    with pytest.raises(
        ValueError,
        match="not awaiting Data Preparation approval",
    ):
        reload_data_preparation_project(
            "project-003",
            store=store,
        )


def test_reload_rejects_missing_review(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))

    project = build_project()
    project.data_preparation_review = None
    store.save(project)

    with pytest.raises(
        ValueError,
        match="no Data Preparation review",
    ):
        reload_data_preparation_project(
            "project-003",
            store=store,
        )
