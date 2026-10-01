from pydantic import BaseModel

from domain.data_understanding import DataUnderstandingArtifact
from domain.data_understanding_review import DataUnderstandingReview


class DataUnderstandingStageResult(BaseModel):
    """Result of deterministic evidence preparation and agent review."""

    artifact: DataUnderstandingArtifact
    review: DataUnderstandingReview
