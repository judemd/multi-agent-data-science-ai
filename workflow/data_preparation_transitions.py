from typing import Literal

from domain.data_preparation_review import DataPreparationReview
from domain.hitl_decision import HITLDecision

DataPreparationState = Literal[
    "data_preparation",
    "human_review",
    "modeling",
    "blocked",
    "revision",
]


def transition_after_data_preparation_hitl(
    decision: HITLDecision,
) -> DataPreparationState:
    """Determine the next state from a Data Preparation human decision."""

    if decision.decision == "approve":
        return "modeling"

    if decision.decision == "request_revision":
        return "revision"

    return "blocked"


def evaluate_and_transition_data_preparation(
    review: DataPreparationReview,
    decision: HITLDecision,
) -> DataPreparationState:
    """Evaluate the Data Preparation review and human decision."""

    return transition_after_data_preparation_hitl(decision)
