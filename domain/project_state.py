from datetime import UTC, datetime

from pydantic import BaseModel, Field

from domain.problem_framing import ProblemFramingArtifact
from workflow.states import WorkflowState


class ProjectState(BaseModel):
    """Persisted state for a single data science workflow."""

    project_id: str = Field(min_length=1)
    project_name: str = Field(min_length=1)

    current_state: WorkflowState = WorkflowState.PROBLEM_FRAMING

    dataset_path: str | None = None
    problem_framing: ProblemFramingArtifact | None = None

    revision: int = Field(default=1, ge=1)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
