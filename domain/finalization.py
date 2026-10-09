"""Structured, auditable handoff evidence for Finalization."""

from pydantic import BaseModel, Field

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_treatment_plan import DataPreparationTreatmentPlan
from domain.evaluation import EvaluationArtifact
from domain.finalization_revision import FinalizationRevisionResolution
from domain.hitl_decision import HITLDecision
from domain.problem_framing import ProblemFramingArtifact


class FinalizationArtifact(BaseModel):
    """Evidence package awaiting a separate human handoff decision."""

    project_id: str = Field(min_length=1)
    project_name: str = Field(min_length=1)
    revision: int = Field(ge=1)

    problem_framing: ProblemFramingArtifact

    source_dataset_path: str = Field(min_length=1)
    prepared_dataset_path: str = Field(min_length=1)
    prepared_dataset_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    data_preparation: DataPreparationArtifact
    treatment_plan: DataPreparationTreatmentPlan
    preparation_decision: HITLDecision

    evaluation: EvaluationArtifact
    modeling_decision: HITLDecision
    evaluation_decision: HITLDecision

    unresolved_risks: list[str] = Field(default_factory=list)
    revision_resolution: FinalizationRevisionResolution | None = None

    deployment_approved: bool = Field(default=False)
    requires_human_handoff_approval: bool = Field(default=True)