from pathlib import Path

from domain.project_state import ProjectState


class ProjectStore:
    """Persist and retrieve project workflow state as JSON."""

    def __init__(self, storage_dir: str = "data/projects") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, project: ProjectState) -> Path:
        """Persist a project and return its storage path."""
        path = self.storage_dir / f"{project.project_id}.json"
        path.write_text(
            project.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return path

    def load(self, project_id: str) -> ProjectState:
        """Load a previously persisted project."""
        path = self.storage_dir / f"{project_id}.json"

        if not path.exists():
            raise FileNotFoundError(f"Project '{project_id}' was not found.")

        return ProjectState.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def delete(self, project_id: str) -> None:
        """Delete a persisted project if it exists."""
        path = self.storage_dir / f"{project_id}.json"

        if path.exists():
            path.unlink()