from pydantic import BaseModel, Field

from domain.data_preparation_action import DataPreparationAction


class DataPreparationReview(BaseModel):
    """Structured preparation proposal produced from deterministic evidence."""

    observed_evidence: list[str] = Field(
        default_factory=list,
        description="Preparation-relevant facts directly supported by the artifact.",
    )

    proposed_actions: list[DataPreparationAction] = Field(
        default_factory=list,
        description=(
            "Structured preparation actions proposed for human review. "
            "These actions are recommendations only and are not executed by the agent."
        ),
    )

    rationale: list[str] = Field(
        default_factory=list,
        description="Evidence-based rationale for proposed preparation actions.",
    )

    risks_and_limitations: list[str] = Field(
        default_factory=list,
        description="Risks or limitations associated with proposed preparation.",
    )

    human_review_questions: list[str] = Field(
        default_factory=list,
        description="Questions requiring human or business clarification.",
    )

    requires_human_review: bool = Field(
        default=True,
        description="Whether human approval is required before preparation.",
    )
