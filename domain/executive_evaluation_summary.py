"""Structured executive interpretation of verified evaluation evidence."""

from pydantic import BaseModel, Field


class ExecutiveEvaluationSummary(BaseModel):
    """Plain-language assessment included in the audited handoff."""

    assessment: str = Field(min_length=1)
    performance_summary: str = Field(min_length=1)
    business_implications: list[str] = Field(min_length=1)
    recommended_actions: list[str] = Field(min_length=1)
    risks_and_limitations: list[str] = Field(default_factory=list)
    evaluation_human_decision: str = Field(min_length=1)
    deployment_authorized: bool = False
