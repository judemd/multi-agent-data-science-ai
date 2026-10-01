from pydantic import BaseModel, Field


class DataUnderstandingReview(BaseModel):
    """Structured interpretation produced after data understanding."""

    observed_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence directly supported by the validated profiling artifact.",
    )

    interpretation: list[str] = Field(
        default_factory=list,
        description="Interpretations derived from the observed evidence.",
    )

    risks_and_limitations: list[str] = Field(
        default_factory=list,
        description="Material risks or limitations identified during review.",
    )

    human_review_questions: list[str] = Field(
        default_factory=list,
        description="Questions requiring human clarification or approval.",
    )

    recommended_next_investigation: list[str] = Field(
        default_factory=list,
        description="Evidence-based investigations to perform next.",
    )

    requires_human_review: bool = Field(
        default=True,
        description="Whether human review is required before progression.",
    )
