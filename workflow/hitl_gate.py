from domain.data_understanding_review import DataUnderstandingReview
from domain.hitl_decision import HITLDecision


def evaluate_data_understanding_hitl(
    review: DataUnderstandingReview,
    decision: HITLDecision,
) -> bool:
    """
    Determine whether the workflow may progress past Data Understanding.

    Progression is permitted only when the human explicitly approves
    the review. The review's requires_human_review flag indicates that
    human review is required; it does not prevent human approval.
    """

    return decision.decision == "approve"
