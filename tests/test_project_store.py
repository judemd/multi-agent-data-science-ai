from domain.project_state import ProjectState
from tools.project_store import ProjectStore
from workflow.states import WorkflowState


def test_project_can_be_saved_and_loaded(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))

    original = ProjectState(
        project_id="project-001",
        project_name="Customer Churn POC",
        dataset_path="datasets/customer_churn_poc.csv",
    )

    saved_path = store.save(original)
    loaded = store.load("project-001")

    assert saved_path.exists()
    assert loaded.project_id == original.project_id
    assert loaded.project_name == original.project_name
    assert loaded.dataset_path == original.dataset_path
    assert loaded.current_state == WorkflowState.PROBLEM_FRAMING
    assert loaded.revision == 1
