from domain.data_understanding_review import DataUnderstandingReview
from domain.hitl_decision import HITLDecision


def evaluate_data_understanding_hitl(
    review: DataUnderstandingReview,
    decision: HITLDecision,
) -> bool:
    """
    Determine whether the workflow may progress past Data Understanding.

    Progression is permitted only when:
    - the review has been explicitly approved by a human; and
    - the review itself does not require further human review.
    """

    if decision.decision != "approve":
        return False

    return not review.requires_human_review
