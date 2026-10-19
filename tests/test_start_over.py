from pathlib import Path
from unittest.mock import MagicMock, Mock

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


def test_start_over_preserves_source_dataset_referenced_by_saved_project(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    dataset = tmp_path / "active_dataset.csv"
    dataset.write_text("feature,target\n1,0\n", encoding="utf-8")
    project = ProjectState(
        project_id="project-retained", project_name="Retained", dataset_path=str(dataset)
    )
    store = ProjectStore(storage_dir=str(tmp_path))
    store.save(project)
    monkeypatch.setattr(app, "PERSISTENCE_DIR", tmp_path)
    monkeypatch.setattr(app, "ACTIVE_STATE_PATH", tmp_path / "active_state.json")
    monkeypatch.setattr(app.st, "session_state", {"project_state": project})
    monkeypatch.setattr(app.st, "rerun", Mock())

    app._start_over()

    assert Path(store.load(project.project_id).dataset_path).read_bytes() == dataset.read_bytes()


def test_start_over_requires_confirmation_during_setup(monkeypatch):
    session = {"business_problem": "Churn"}
    monkeypatch.setattr(app.st, "session_state", session)
    monkeypatch.setattr(app.st, "button", Mock(return_value=True))
    monkeypatch.setattr(app.st, "rerun", Mock())
    reset = Mock()
    monkeypatch.setattr(app, "_start_over", reset)

    app.render_global_start_over()

    assert session["start_over_confirmation"] is True
    reset.assert_not_called()


def test_start_over_cancel_keeps_active_workflow(monkeypatch):
    session = {"start_over_confirmation": True, "business_problem": "Churn"}
    monkeypatch.setattr(app.st, "session_state", session)
    monkeypatch.setattr(app.st, "button", Mock(side_effect=[True, False]))
    monkeypatch.setattr(app.st, "columns", Mock(return_value=[MagicMock(), MagicMock()]))
    monkeypatch.setattr(app.st, "warning", Mock())
    monkeypatch.setattr(app.st, "rerun", Mock())
    reset = Mock()
    monkeypatch.setattr(app, "_start_over", reset)

    app.render_global_start_over()

    assert session["start_over_confirmation"] is False
    assert session["business_problem"] == "Churn"
    reset.assert_not_called()


def test_start_over_confirm_resets_workflow(monkeypatch):
    monkeypatch.setattr(app.st, "session_state", {"start_over_confirmation": True})
    monkeypatch.setattr(app.st, "button", Mock(side_effect=[False, True]))
    monkeypatch.setattr(app.st, "columns", Mock(return_value=[MagicMock(), MagicMock()]))
    monkeypatch.setattr(app.st, "warning", Mock())
    reset = Mock()
    monkeypatch.setattr(app, "_start_over", reset)

    app.render_global_start_over()

    reset.assert_called_once_with()


def test_upload_does_not_overwrite_saved_source_dataset(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.setattr(app, "PERSISTENCE_DIR", tmp_path)
    monkeypatch.setattr(app.st, "session_state", {})
    upload = Mock(name="upload")
    upload.name = "data.csv"
    upload.getvalue.return_value = b"feature,target\n1,0\n"
    first = app._persist_uploaded_dataset(upload)
    assert app._persist_uploaded_dataset(upload) == first
    upload.getvalue.return_value = b"feature,target\n2,1\n"

    second = app._persist_uploaded_dataset(upload)

    assert first != second
    assert first.read_bytes() == b"feature,target\n1,0\n"
    assert second.read_bytes() == b"feature,target\n2,1\n"
