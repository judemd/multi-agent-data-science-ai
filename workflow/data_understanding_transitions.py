from typing import Literal

from domain.hitl_decision import HITLDecision

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
