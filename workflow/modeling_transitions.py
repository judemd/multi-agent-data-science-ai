from typing import Literal

from domain.hitl_decision import HITLDecision
from domain.modeling_review import ModelingReview

ModelingState = Literal[
    "modeling",
    "human_review",
    "evaluation",
    "blocked",
    "revision",
]


def transition_after_modeling_hitl(
    decision: HITLDecision,
) -> ModelingState:
    """Determine the next state from a Modeling human decision."""

    if decision.decision == "approve":
        return "evaluation"

    if decision.decision == "request_revision":
        return "revision"

    return "blocked"


def evaluate_and_transition_modeling(
    review: ModelingReview,
    decision: HITLDecision,
) -> ModelingState:
    """Evaluate the Modeling review and human decision."""

    return transition_after_modeling_hitl(decision)
