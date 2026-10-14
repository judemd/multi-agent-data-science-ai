import os
from pathlib import Path

from domain.project_state import ProjectState


class ProjectStore:
    """Persist and retrieve project workflow state as JSON."""

    def __init__(self, storage_dir: str = "data/projects") -> None:
        self.bucket_name = os.getenv("GCS_BUCKET") or None
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, project: ProjectState) -> Path:
        """Persist a project and return its storage path."""
        path = self.storage_dir / f"{project.project_id}.json"
        path.write_text(
            project.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if self.bucket_name:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            if project.prepared_dataset_path:
                prepared_path = Path(project.prepared_dataset_path)
                if not prepared_path.is_file():
                    raise FileNotFoundError(
                        f"Prepared dataset '{prepared_path}' was not found."
                    )
                prepared_blob = bucket.blob(prepared_path.name)
                if not prepared_blob.exists():
                    prepared_blob.upload_from_filename(
                        str(prepared_path),
                        if_generation_match=0,
                    )

            blob = bucket.blob(path.name)
            if project._cloud_generation is not None:
                generation = project._cloud_generation
            else:
                from google.api_core.exceptions import NotFound

                try:
                    blob.reload()
                    generation = blob.generation
                except NotFound:
                    generation = 0

            blob.upload_from_filename(
                str(path),
                if_generation_match=(
                    int(generation) if generation is not None else 0
                ),
            )
            if blob.generation is not None:
                project._cloud_generation = int(blob.generation)

        return path

    def load(self, project_id: str) -> ProjectState:
        """Load a project, preferring Cloud Storage when configured."""
        path = self.storage_dir / f"{project_id}.json"

        if self.bucket_name:
            from google.api_core.exceptions import NotFound
            from google.cloud import storage

            client = storage.Client()
            blob = client.bucket(self.bucket_name).blob(path.name)

            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(path))
            except NotFound:
                raise FileNotFoundError(
                    f"Project '{project_id}' was not found."
                ) from None

        if not path.exists():
            raise FileNotFoundError(
                f"Project '{project_id}' was not found."
            )

        project = ProjectState.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if self.bucket_name:
            project._cloud_generation = (
                int(blob.generation)
                if blob.generation is not None
                else None
            )

        if self.bucket_name and project.prepared_dataset_path:
            prepared_path = Path(project.prepared_dataset_path)

            if not prepared_path.exists():
                from google.cloud import storage

                prepared_path.parent.mkdir(parents=True, exist_ok=True)
                client = storage.Client()
                blob = client.bucket(self.bucket_name).blob(
                    prepared_path.name
                )
                blob.download_to_filename(str(prepared_path))

        return project

    def delete(self, project_id: str) -> None:
        """Delete a persisted project locally and from Cloud Storage."""
        path = self.storage_dir / f"{project_id}.json"

        if self.bucket_name:
            from google.api_core.exceptions import NotFound
            from google.cloud import storage

            client = storage.Client()
            blob = client.bucket(self.bucket_name).blob(path.name)

            try:
                blob.delete()
            except NotFound:
                pass

        if path.exists():
            path.unlink()
