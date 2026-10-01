from typing import Literal

from pydantic import BaseModel, Field


class HITLDecision(BaseModel):
    """Human decision controlling workflow progression."""

    decision: Literal[
        "approve",
        "reject",
        "request_revision",
    ]

    reviewer: str = Field(
        min_length=1,
        description="Human reviewer who made the decision.",
    )

    rationale: str = Field(
        min_length=1,
        description="Reason provided by the reviewer.",
    )

    feedback: list[str] = Field(
        default_factory=list,
        description="Requested changes or clarifications.",
    )
