from tools.project_store import ProjectStore


def test_project_store_defaults_to_local_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    store = ProjectStore(storage_dir=str(tmp_path))

    assert store.storage_dir == tmp_path
    assert list(tmp_path.iterdir()) == []


def test_project_store_reads_gcs_bucket_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    store = ProjectStore(storage_dir=str(tmp_path))

    assert store.bucket_name == "test-private-bucket"


def test_project_store_reads_gcs_bucket_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    store = ProjectStore(storage_dir=str(tmp_path))

    assert store.bucket_name == "test-private-bucket"

def test_project_save_uploads_json_to_gcs(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from domain.project_state import ProjectState

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-001",
        project_name="Customer Churn POC",
        dataset_path="datasets/customer_churn_poc.csv",
    )

    store.save(project)

    client.bucket.assert_called_once_with("test-private-bucket")
    blob = client.bucket.return_value.blob
    blob.assert_called_once_with("project-001.json")
    blob.return_value.upload_from_filename.assert_called_once_with(
        str(tmp_path / "project-001.json"),
        if_generation_match=1,
    )


def test_project_load_downloads_missing_json_from_gcs(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from domain.project_state import ProjectState

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    project = ProjectState(
        project_id="project-001",
        project_name="Customer Churn POC",
        dataset_path="datasets/customer_churn_poc.csv",
    )

    client = MagicMock()

    def download_file(filename):
        from pathlib import Path
        Path(filename).write_text(project.model_dump_json(), encoding="utf-8")

    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        download_file
    )

    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)

    store = ProjectStore(storage_dir=str(tmp_path))
    restored = store.load("project-001")

    assert restored.project_id == "project-001"
    assert restored.project_name == "Customer Churn POC"
    client.bucket.assert_called_once_with("test-private-bucket")
    client.bucket.return_value.blob.assert_called_once_with("project-001.json")
    client.bucket.return_value.blob.return_value.download_to_filename.assert_called_once_with(
        str(tmp_path / "project-001.json")
    )


def test_project_load_handles_missing_gcs_object(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import pytest
    from google.api_core.exceptions import NotFound

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        NotFound("Object not found")
    )
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)

    store = ProjectStore(storage_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="was not found"):
        store.load("missing-project")


def test_project_save_uploads_prepared_dataset_before_json(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from domain.project_state import ProjectState

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    prepared_path = tmp_path / "project-001_prepared.csv"
    prepared_path.write_text("customer_id,revenue\n1,100\n", encoding="utf-8")

    project = ProjectState(
        project_id="project-001",
        project_name="Customer Churn POC",
        dataset_path="datasets/customer_churn_poc.csv",
        prepared_dataset_path=str(prepared_path),
    )

    client = MagicMock()
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)

    store = ProjectStore(storage_dir=str(tmp_path))
    store.save(project)

    calls = client.bucket.return_value.blob.call_args_list
    assert [call.args[0] for call in calls] == [
        "project-001_prepared.csv",
        "project-001.json",
    ]

    prepared_blob = client.bucket.return_value.blob.return_value
    assert prepared_blob.upload_from_filename.call_args_list[0].kwargs.get(
        "if_generation_match"
    ) == 0


def test_project_load_restores_prepared_dataset(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from pathlib import Path
    from domain.project_state import ProjectState

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    prepared_path = tmp_path / "project-001_prepared.csv"
    project = ProjectState(
        project_id="project-001",
        project_name="Customer Churn POC",
        dataset_path="datasets/customer_churn_poc.csv",
        prepared_dataset_path=str(prepared_path),
    )

    client = MagicMock()

    def download_file(filename):
        destination = Path(filename)
        if destination.suffix == ".json":
            destination.write_text(project.model_dump_json(), encoding="utf-8")
        else:
            destination.write_text(
                "customer_id,revenue\n1,100\n",
                encoding="utf-8",
            )

    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        download_file
    )
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)

    store = ProjectStore(storage_dir=str(tmp_path))
    restored = store.load("project-001")

    assert restored.project_id == "project-001"
    assert prepared_path.read_text(encoding="utf-8") == (
        "customer_id,revenue\n1,100\n"
    )


def test_active_state_is_uploaded_to_gcs(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock

    import app

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    persistence_dir = tmp_path / "projects"
    persistence_dir.mkdir()
    active_state = persistence_dir / "active_state.json"

    client = MagicMock()
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)
    monkeypatch.setattr(app, "PERSISTENCE_DIR", persistence_dir)
    monkeypatch.setattr(app, "ACTIVE_STATE_PATH", active_state)
    monkeypatch.setattr(app, "PROJECT_STORE", MagicMock())

    monkeypatch.setattr(
        app.st,
        "session_state",
        {
            "project_id": "project-001",
            "business_problem": "Customer churn",
            "business_objective": "Reduce churn",
            "desired_outcome": "Improve retention",
        },
    )

    app._persist_active_state()

    assert active_state.exists()
    payload = json.loads(active_state.read_text(encoding="utf-8"))
    assert payload["project_id"] == "project-001"

    client.bucket.assert_called_once_with("test-private-bucket")
    client.bucket.return_value.blob.assert_called_once_with("active_state.json")
    client.bucket.return_value.blob.return_value.upload_from_filename.assert_called_once_with(
        str(active_state),
        if_generation_match=1,
    )


def test_active_state_is_restored_from_gcs(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock

    import app

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    persistence_dir = tmp_path / "projects"
    persistence_dir.mkdir()
    active_state = persistence_dir / "active_state.json"

    payload = {
        "project_id": "project-001",
        "business_context": {
            "business_problem": "Customer churn",
            "business_objective": "Reduce churn",
            "desired_outcome": "Improve retention",
        },
        "target_column": "churned",
        "persisted_dataset_path": None,
        "persisted_dataset_name": None,
    }

    client = MagicMock()

    def download_file(filename):
        from pathlib import Path
        Path(filename).write_text(json.dumps(payload), encoding="utf-8")

    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        download_file
    )

    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)
    monkeypatch.setattr(app, "PERSISTENCE_DIR", persistence_dir)
    monkeypatch.setattr(app, "ACTIVE_STATE_PATH", active_state)
    monkeypatch.setattr(app, "PROJECT_STORE", MagicMock())

    session = {}
    monkeypatch.setattr(app.st, "session_state", session)

    app._restore_persisted_state()

    assert session["business_problem"] == "Customer churn"
    assert session["business_objective"] == "Reduce churn"
    assert session["desired_outcome"] == "Improve retention"
    assert session["target_column"] == "churned"
    assert session["project_state"] is not None
    app.PROJECT_STORE.load.assert_called_once_with("project-001")


def test_uploaded_dataset_is_uploaded_to_gcs(tmp_path, monkeypatch):
    from io import BytesIO
    from unittest.mock import MagicMock

    import app

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")
    monkeypatch.setattr(app, "PERSISTENCE_DIR", tmp_path)

    client = MagicMock()
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)

    uploaded_file = MagicMock()
    uploaded_file.name = "customers.csv"
    uploaded_file.getvalue.return_value = b"customer_id,churned\n1,0\n"

    dataset_path = app._persist_uploaded_dataset(uploaded_file)

    assert dataset_path.read_bytes() == b"customer_id,churned\n1,0\n"
    assert app.st.session_state["persisted_dataset_name"] == "customers.csv"

    client.bucket.assert_called_once_with("test-private-bucket")
    client.bucket.return_value.blob.assert_called_once_with("active_dataset.csv")
    client.bucket.return_value.blob.return_value.upload_from_filename.assert_called_once_with(
        str(dataset_path)
    )


def test_start_over_deletes_active_gcs_objects_but_preserves_projects(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    import app
    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    # Create the saved project in local-only mode before enabling the
    # mocked Cloud Storage cleanup path.
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-preserved",
        project_name="Preserved Customer Churn Project",
        dataset_path="datasets/customer_churn_poc.csv",
    )
    saved_path = store.save(project)

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    active_state = tmp_path / "active_state.json"
    active_state.write_text('{"project_id": "project-preserved"}', encoding="utf-8")

    active_dataset = tmp_path / "active_dataset.csv"
    active_dataset.write_text("customer_id,churned\n1,0\n", encoding="utf-8")

    client = MagicMock()
    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)
    monkeypatch.setattr(app, "PROJECT_STORE", store)
    monkeypatch.setattr(app, "PERSISTENCE_DIR", tmp_path)
    monkeypatch.setattr(app, "ACTIVE_STATE_PATH", active_state)
    monkeypatch.setattr(
        app.st,
        "session_state",
        {
            "project_state": project,
            "persisted_dataset_path": str(active_dataset),
        },
    )
    rerun = MagicMock()
    monkeypatch.setattr(app.st, "rerun", rerun)

    app._start_over()

    deleted_names = [
        call.args[0]
        for call in client.bucket.return_value.blob.call_args_list
    ]
    assert deleted_names == ["active_state.json", "active_dataset.csv"]
    assert client.bucket.return_value.blob.return_value.delete.call_count == 2
    assert saved_path.exists()
    rerun.assert_called_once_with()


def test_project_load_prefers_cloud_version_over_stale_local_cache(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    local_project = ProjectState(
        project_id="project-cache-test",
        project_name="Stale Local Project",
        dataset_path="datasets/local.csv",
    )
    cloud_project = ProjectState(
        project_id="project-cache-test",
        project_name="Updated Cloud Project",
        dataset_path="datasets/cloud.csv",
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    local_path = tmp_path / "project-cache-test.json"
    local_path.write_text(
        local_project.model_dump_json(),
        encoding="utf-8",
    )

    client = MagicMock()
    cloud_payload = cloud_project.model_dump_json()

    def download_file(filename):
        Path(filename).write_text(
            cloud_payload,
            encoding="utf-8",
        )

    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        download_file
    )
    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    loaded = store.load("project-cache-test")

    assert loaded.project_name == "Updated Cloud Project"
    client.bucket.return_value.blob.return_value.download_to_filename.assert_called_once()


def test_project_load_prefers_cloud_version_over_stale_local_cache(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    local_project = ProjectState(
        project_id="project-cache-test",
        project_name="Stale Local Project",
        dataset_path="datasets/local.csv",
    )
    cloud_project = ProjectState(
        project_id="project-cache-test",
        project_name="Updated Cloud Project",
        dataset_path="datasets/cloud.csv",
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    local_path = tmp_path / "project-cache-test.json"
    local_path.write_text(
        local_project.model_dump_json(),
        encoding="utf-8",
    )

    client = MagicMock()
    cloud_payload = cloud_project.model_dump_json()

    def download_file(filename):
        Path(filename).write_text(
            cloud_payload,
            encoding="utf-8",
        )

    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        download_file
    )
    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    loaded = store.load("project-cache-test")

    assert loaded.project_name == "Updated Cloud Project"
    client.bucket.return_value.blob.return_value.download_to_filename.assert_called_once()


def test_project_load_prefers_cloud_version_over_stale_local_cache(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    local_project = ProjectState(
        project_id="project-cache-test",
        project_name="Stale Local Project",
        dataset_path="datasets/local.csv",
    )
    cloud_project = ProjectState(
        project_id="project-cache-test",
        project_name="Updated Cloud Project",
        dataset_path="datasets/cloud.csv",
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    local_path = tmp_path / "project-cache-test.json"
    local_path.write_text(
        local_project.model_dump_json(),
        encoding="utf-8",
    )

    client = MagicMock()
    cloud_payload = cloud_project.model_dump_json()

    def download_file(filename):
        Path(filename).write_text(
            cloud_payload,
            encoding="utf-8",
        )

    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        download_file
    )
    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    loaded = store.load("project-cache-test")

    assert loaded.project_name == "Updated Cloud Project"
    client.bucket.return_value.blob.return_value.download_to_filename.assert_called_once()


def test_project_save_uses__cloud_generation_precondition(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-generation-test",
        project_name="Generation Precondition Test",
        dataset_path="datasets/test.csv",
    )

    store.save(project)

    upload = (
        client.bucket.return_value
        .blob.return_value.upload_from_filename
    )
    assert upload.call_args.kwargs.get("if_generation_match") is not None


def test_project_save_uses_zero_generation_for_new_cloud_object(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generation = None

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-new-object",
        project_name="New Cloud Project",
        dataset_path="datasets/new.csv",
    )

    store.save(project)

    blob.upload_from_filename.assert_called_once_with(
        str(tmp_path / "project-new-object.json"),
        if_generation_match=0,
    )


def test_project_save_rejects_generation_conflict(
    tmp_path, monkeypatch
):
    import pytest
    from unittest.mock import MagicMock
    from google.api_core.exceptions import PreconditionFailed

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generation = 42
    blob.upload_from_filename.side_effect = PreconditionFailed(
        "Another instance updated this project."
    )

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-conflict",
        project_name="Concurrent Update Test",
        dataset_path="datasets/test.csv",
    )

    with pytest.raises(PreconditionFailed):
        store.save(project)

    blob.upload_from_filename.assert_called_once_with(
        str(tmp_path / "project-conflict.json"),
        if_generation_match=42,
    )


def test_project_load_tracks__cloud_generation(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generation = 42

    project = ProjectState(
        project_id="project-loaded-generation",
        project_name="Loaded Generation Test",
    )

    def download_file(filename):
        from pathlib import Path
        Path(filename).write_text(
            project.model_dump_json(),
            encoding="utf-8",
        )

    blob.download_to_filename.side_effect = download_file

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    loaded = store.load("project-loaded-generation")

    assert loaded._cloud_generation == 42
    assert "_cloud_generation" not in loaded.model_dump()


def test_project_save_uses_generation_from_loaded_state(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generation = 43

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-stale-update",
        project_name="Stale Update Test",
    )

    # This project was loaded when the cloud object was generation 42.
    # Another instance has since advanced the cloud object to 43.
    project._cloud_generation = 42

    store.save(project)

    blob.upload_from_filename.assert_called_once_with(
        str(tmp_path / "project-stale-update.json"),
        if_generation_match=42,
    )


def test_project_save_handles_missing_cloud_object(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    import pytest
    from google.api_core.exceptions import NotFound

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.reload.side_effect = NotFound("Object does not exist")

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-first-save",
        project_name="First Save Test",
    )

    store.save(project)

    blob.upload_from_filename.assert_called_once_with(
        str(tmp_path / "project-first-save.json"),
        if_generation_match=0,
    )


def test_project_delete_removes_cloud_object(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project_path = tmp_path / "project-to-delete.json"
    project_path.write_text("{}", encoding="utf-8")

    store.delete("project-to-delete")

    assert not project_path.exists()
    client.bucket.return_value.blob.assert_called_once_with(
        "project-to-delete.json"
    )
    client.bucket.return_value.blob.return_value.delete.assert_called_once()


def test_project_load_does_not_restore_deleted_cloud_project_from_local_cache(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    import pytest
    from google.api_core.exceptions import NotFound

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    project = ProjectState(
        project_id="deleted-cloud-project",
        project_name="Stale Local Copy",
    )
    local_path = tmp_path / "deleted-cloud-project.json"
    local_path.write_text(
        project.model_dump_json(),
        encoding="utf-8",
    )

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.download_to_filename.side_effect = NotFound(
        "Cloud project no longer exists"
    )

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        store.load("deleted-cloud-project")


def test_load_persisted_dataframe_restores_missing_file_from_gcs(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    import pandas as pd
    import app

    dataset_path = tmp_path / "active_dataset.csv"
    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")
    monkeypatch.setattr(
        app.st,
        "session_state",
        {"persisted_dataset_path": str(dataset_path)},
    )

    client = MagicMock()

    def download_file(filename):
        from pathlib import Path
        Path(filename).write_text(
            "feature,target\n1,0\n2,1\n",
            encoding="utf-8",
        )

    client.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
        download_file
    )

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    result = app._load_persisted_dataframe()

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    client.bucket.return_value.blob.assert_called_once_with(
        "active_dataset.csv"
    )


def test_project_save_updates_cloud_generation_after_upload(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generation = 42

    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: client,
    )

    store = ProjectStore(storage_dir=str(tmp_path))
    project = ProjectState(
        project_id="project-generation-update",
        project_name="Generation Update Test",
    )
    project._cloud_generation = 41

    store.save(project)

    assert project._cloud_generation == 42
    assert "_cloud_generation" not in project.model_dump()


def test_active_state_upload_uses_generation_precondition(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import app

    monkeypatch.setenv("GCS_BUCKET", "test-private-bucket")

    persistence_dir = tmp_path / "projects"
    active_state = persistence_dir / "active_state.json"

    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generation = 17

    monkeypatch.setattr("google.cloud.storage.Client", lambda: client)
    monkeypatch.setattr(app, "PERSISTENCE_DIR", persistence_dir)
    monkeypatch.setattr(app, "ACTIVE_STATE_PATH", active_state)
    monkeypatch.setattr(app, "PROJECT_STORE", MagicMock())
    monkeypatch.setattr(
        app.st,
        "session_state",
        {"project_id": "project-001"},
    )

    app._persist_active_state()

    blob.upload_from_filename.assert_called_once_with(
        str(active_state),
        if_generation_match=17,
    )
