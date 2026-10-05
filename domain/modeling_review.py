from pydantic import BaseModel, Field


class ProposedModel(BaseModel):
    """A candidate model proposed for human review."""

    name: str = Field(
        min_length=1,
        description="Name of the proposed model.",
    )

    model_family: str = Field(
        min_length=1,
        description="Model family or algorithm category.",
    )

    rationale: str = Field(
        min_length=1,
        description="Evidence-based reason this model is suitable.",
    )

    considerations: list[str] = Field(
        default_factory=list,
        description="Important assumptions, trade-offs, or considerations.",
    )


class ModelingReview(BaseModel):
    """Structured model-selection proposal produced from deterministic evidence."""

    observed_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence directly supported by the modeling artifact.",
    )

    proposed_models: list[ProposedModel] = Field(
        default_factory=list,
        description="Candidate models proposed for human review.",
    )

    recommended_model: str = Field(
        min_length=1,
        description="Recommended model from the proposed candidates.",
    )

    rationale: list[str] = Field(
        default_factory=list,
        description="Evidence-based rationale for the modeling recommendation.",
    )

    validation_strategy: list[str] = Field(
        default_factory=list,
        description="Proposed strategy for validating the selected model.",
    )

    risks_and_limitations: list[str] = Field(
        default_factory=list,
        description="Risks or limitations associated with the modeling proposal.",
    )

    human_review_questions: list[str] = Field(
        default_factory=list,
        description="Questions requiring human clarification or approval.",
    )

    requires_human_review: bool = Field(
        default=True,
        description="Whether human approval is required before progression.",
    )
