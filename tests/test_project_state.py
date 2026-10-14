import pytest
from pydantic import ValidationError

from domain.project_state import ProjectState
from workflow.states import WorkflowState


def test_new_project_starts_in_problem_framing():
    project = ProjectState(
        project_id="project-001",
        project_name="Customer Churn POC",
        dataset_path="datasets/customer_churn_poc.csv",
    )

    assert project.current_state == WorkflowState.PROBLEM_FRAMING
    assert project.revision == 1
    assert project.problem_framing is None


def test_project_requires_a_name():
    with pytest.raises(ValidationError):
        ProjectState(
            project_id="project-001",
            project_name="",
        )


def test_legacy_project_has_no_preparation_fingerprints():
    project = ProjectState(
        project_id="legacy-001",
        project_name="Legacy Project",
    )

    assert project.data_preparation_dataset_fingerprint is None
    assert project.data_preparation_evidence_fingerprint is None


def test_project_rejects_invalid_preparation_fingerprint():
    with pytest.raises(ValidationError):
        ProjectState(
            project_id="project-invalid",
            project_name="Invalid Fingerprint",
            data_preparation_dataset_fingerprint="not-a-sha256-hash",
        )


def test_uuid_project_ids_are_unique_for_same_dataset():
    from uuid import uuid4

    from domain.project_state import ProjectState

    dataset_name = "customers.csv"

    first = ProjectState(
        project_id=f"project-{uuid4().hex}",
        project_name="First analysis",
        dataset_path=dataset_name,
    )
    second = ProjectState(
        project_id=f"project-{uuid4().hex}",
        project_name="Second analysis",
        dataset_path=dataset_name,
    )

    assert first.project_id != second.project_id
    assert first.dataset_path == second.dataset_path
