from pydantic import BaseModel, Field

from domain.data_preparation_issue import DataPreparationTreatment


class DataPreparationTreatmentDecision(BaseModel):
    """A human-selected treatment for one detected preparation issue."""

    issue_id: str = Field(min_length=1)

    treatment: DataPreparationTreatment

    rationale: str = Field(min_length=1)
