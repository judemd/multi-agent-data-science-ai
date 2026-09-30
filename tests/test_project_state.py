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
