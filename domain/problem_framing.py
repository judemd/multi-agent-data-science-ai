from pydantic import BaseModel, Field


class ProblemFramingArtifact(BaseModel):
    """Structured output produced during the problem-framing stage."""

    business_problem: str = Field(min_length=1)
    business_objective: str = Field(min_length=1)
    target_outcome: str = Field(min_length=1)
    success_criteria: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    questions_for_human: list[str] = Field(default_factory=list)
