from pydantic import BaseModel, Field, model_validator

from domain.data_preparation_treatment_decision import (
    DataPreparationTreatmentDecision,
)


class DataPreparationTreatmentPlan(BaseModel):
    """Human-selected treatments bound to reviewed preparation evidence."""

    dataset_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    evidence_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    reviewer: str = Field(min_length=1)

    decisions: list[DataPreparationTreatmentDecision] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def reject_duplicate_issue_decisions(self):
        issue_ids = [decision.issue_id for decision in self.decisions]

        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError(
                "A treatment plan cannot contain duplicate issue decisions."
            )

        return self
