from typing import Literal

from domain.data_understanding_review import DataUnderstandingReview
from domain.hitl_decision import HITLDecision
from workflow.hitl_gate import evaluate_data_understanding_hitl

DataUnderstandingState = Literal[
    "data_understanding",
    "human_review",
    "next_stage",
    "blocked",
    "revision",
]


def transition_after_hitl(
    decision: HITLDecision,
) -> DataUnderstandingState:
    """Determine the next workflow state from a human decision."""

    if decision.decision == "approve":
        return "next_stage"

    if decision.decision == "request_revision":
        return "revision"

    return "blocked"


def evaluate_and_transition_data_understanding(
    review: DataUnderstandingReview,
    decision: HITLDecision,
) -> DataUnderstandingState:
    """Evaluate the HITL gate and return the resulting workflow state."""

    if review.requires_human_review:
        return "blocked"

    if evaluate_data_understanding_hitl(
        review,
        decision,
    ):
        return "next_stage"

    return transition_after_hitl(decision)
