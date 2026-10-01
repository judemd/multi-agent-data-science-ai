from domain.data_understanding_review import DataUnderstandingReview
from domain.hitl_decision import HITLDecision
from workflow.hitl_gate import evaluate_data_understanding_hitl


def test_approved_review_allows_progression():
    review = DataUnderstandingReview(
        observed_evidence=[
            "No material data-quality issue was identified."
        ],
        requires_human_review=False,
    )

    decision = HITLDecision(
        decision="approve",
        reviewer="data_scientist",
        rationale="The findings are sufficiently supported.",
    )

    assert evaluate_data_understanding_hitl(
        review,
        decision,
    ) is True


def test_rejected_review_blocks_progression():
    review = DataUnderstandingReview(
        observed_evidence=[
            "Potential duplicate customer identifiers were found."
        ],
        requires_human_review=False,
    )

    decision = HITLDecision(
        decision="reject",
        reviewer="data_scientist",
        rationale="The duplicate issue requires investigation.",
    )

    assert evaluate_data_understanding_hitl(
        review,
        decision,
    ) is False


def test_revision_request_blocks_progression():
    review = DataUnderstandingReview(
        requires_human_review=False,
    )

    decision = HITLDecision(
        decision="request_revision",
        reviewer="data_scientist",
        rationale="More explanation is required.",
    )

    assert evaluate_data_understanding_hitl(
        review,
        decision,
    ) is False


def test_review_requiring_human_review_can_progress_after_human_approval():
    review = DataUnderstandingReview(
        observed_evidence=[
            "Potential data-quality issue identified."
        ],
        requires_human_review=True,
    )

    decision = HITLDecision(
        decision="approve",
        reviewer="data_scientist",
        rationale="The review has been considered and progression is approved.",
    )

    assert evaluate_data_understanding_hitl(
        review,
        decision,
    ) is True