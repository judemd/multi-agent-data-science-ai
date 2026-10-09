from pathlib import Path
from unittest.mock import Mock

import app
from domain.project_state import ProjectState
from tools.project_store import ProjectStore


def test_start_over_preserves_saved_project_and_clears_active_session(
    tmp_path, monkeypatch
):
    store = ProjectStore(storage_dir=str(tmp_path))

    project = ProjectState(
        project_id="project-preserved",
        project_name="Preserved Customer Churn Project",
        dataset_path="datasets/customer_churn_poc.csv",
    )

    saved_path = store.save(project)

    active_state = tmp_path / "active_state.json"
    active_state.write_text('{"project_id": "project-preserved"}', encoding="utf-8")

    active_dataset = tmp_path / "active_dataset.csv"
    active_dataset.write_text("customer_id,churned\n1,0\n", encoding="utf-8")

    session = {
        "project_state": project,
        "persisted_dataset_path": str(active_dataset),
    }
    rerun = Mock()

    monkeypatch.setattr(app, "PROJECT_STORE", store)
    monkeypatch.setattr(app, "PERSISTENCE_DIR", tmp_path)
    monkeypatch.setattr(app, "ACTIVE_STATE_PATH", active_state)
    monkeypatch.setattr(app.st, "session_state", session)
    monkeypatch.setattr(app.st, "rerun", rerun)

    app._start_over()

    assert saved_path.exists()
    assert store.load(project.project_id).project_name == project.project_name
    assert not active_state.exists()
    assert not active_dataset.exists()
    assert session == {}
    rerun.assert_called_once_with()
