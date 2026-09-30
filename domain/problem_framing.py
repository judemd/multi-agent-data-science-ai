from pydantic import BaseModel, Field


class ProblemFramingArtifact(BaseModel):
    """Structured output produced during the problem-framing stage."""

    business_problem: str = Field(
        min_length=1,
        description="The business problem that the data science project should address.",
    )

    business_objective: str = Field(
        min_length=1,
        description="The business objective the project is intended to support.",
    )

    target_outcome: str = Field(
        min_length=1,
        description="The desired business outcome if the project succeeds.",
    )

    success_criteria: list[str] = Field(
        default_factory=list,
        description="Measurable criteria that would indicate success.",
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions that require validation.",
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Known business, operational, technical, or regulatory constraints.",
    )

    risks: list[str] = Field(
        default_factory=list,
        description="Material risks identified during problem framing.",
    )

    questions_for_human: list[str] = Field(
        default_factory=list,
        description="Questions that require human clarification before progression.",
    )
