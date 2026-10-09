"""Human-reviewed resolutions of Finalization revision requests."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FinalizationFeedbackResolution(BaseModel):
    """Document the response to one original handoff feedback item."""

    feedback: str = Field(min_length=1)
    resolution: str = Field(min_length=1)
    status: Literal["addressed", "unresolved"]


class FinalizationRevisionResolution(BaseModel):
    """Human confirmation of responses to a handoff revision request."""

    reviewer: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    items: list[FinalizationFeedbackResolution] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_feedback(self):
        feedback_items = [item.feedback for item in self.items]

        if len(feedback_items) != len(set(feedback_items)):
            raise ValueError(
                "Finalization revision resolution contains duplicate feedback."
            )

        return self