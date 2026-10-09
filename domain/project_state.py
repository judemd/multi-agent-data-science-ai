from datetime import UTC, datetime

from pydantic import BaseModel, Field

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_review import DataPreparationReview
from domain.data_preparation_treatment_plan import DataPreparationTreatmentPlan
from domain.evaluation import EvaluationArtifact
from domain.finalization import FinalizationArtifact
from domain.finalization_revision import FinalizationRevisionResolution
from domain.hitl_decision import HITLDecision
from domain.modeling import ModelingArtifact
from domain.modeling_review import ModelingReview
from domain.problem_framing import ProblemFramingArtifact
from workflow.states import WorkflowState


class ProjectState(BaseModel):
    """Persisted state for a single data science workflow."""

    project_id: str = Field(min_length=1)
    project_name: str = Field(min_length=1)

    current_state: WorkflowState = WorkflowState.PROBLEM_FRAMING

    dataset_path: str | None = None
    prepared_dataset_path: str | None = None
    prepared_dataset_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    data_preparation_dataset_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    data_preparation_evidence_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    problem_framing: ProblemFramingArtifact | None = None

    data_preparation: DataPreparationArtifact | None = None
    data_preparation_review: DataPreparationReview | None = None
    data_preparation_treatment_plan: DataPreparationTreatmentPlan | None = None
    data_preparation_decision: HITLDecision | None = None

    modeling: ModelingArtifact | None = None
    modeling_review: ModelingReview | None = None
    modeling_decision: HITLDecision | None = None
    selected_model: str | None = None
    selected_feature_columns: list[str] | None = None

    evaluation: EvaluationArtifact | None = None
    evaluation_decision: HITLDecision | None = None
    evaluation_history: list[EvaluationArtifact] = Field(
        default_factory=list
    )
    evaluation_decision_history: list[HITLDecision] = Field(
        default_factory=list
    )

    finalization: FinalizationArtifact | None = None
    finalization_evidence_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    handoff_decision: HITLDecision | None = None
    handoff_decision_history: list[HITLDecision] = Field(
        default_factory=list
    )
    finalization_revision_resolution: FinalizationRevisionResolution | None = None
    finalization_revision_history: list[FinalizationRevisionResolution] = Field(
        default_factory=list
    )

    revision: int = Field(default=1, ge=1)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
