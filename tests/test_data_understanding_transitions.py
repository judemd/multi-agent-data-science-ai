from domain.data_understanding_review import DataUnderstandingReview
from domain.hitl_decision import HITLDecision
from workflow.data_understanding_transitions import (
    evaluate_and_transition_data_understanding,
    transition_after_hitl,
)


def test_approval_transitions_to_next_stage():
    decision = HITLDecision(
        decision="approve",
        reviewer="data_scientist",
        rationale="Evidence is sufficient.",
    )

    assert transition_after_hitl(decision) == "next_stage"


def test_rejection_blocks_workflow():
    decision = HITLDecision(
        decision="reject",
        reviewer="data_scientist",
        rationale="Evidence requires further investigation.",
    )

    assert transition_after_hitl(decision) == "blocked"


def test_revision_request_returns_to_revision():
    decision = HITLDecision(
        decision="request_revision",
        reviewer="data_scientist",
        rationale="Additional explanation is required.",
        feedback=[
            "Clarify the duplicate identifier finding.",
            "Investigate missing revenue values.",
        ],
    )

    assert transition_after_hitl(decision) == "revision"


def test_approved_review_allows_next_stage():
    review = DataUnderstandingReview(
        observed_evidence=[
            "The evidence is sufficiently supported.",
        ],
        requires_human_review=False,
    )

    decision = HITLDecision(
        decision="approve",
        reviewer="data_scientist",
        rationale="Evidence is sufficient.",
    )

    assert evaluate_and_transition_data_understanding(
        review,
        decision,
    ) == "next_stage"


def test_review_requiring_human_review_can_progress_after_human_approval():
    review = DataUnderstandingReview(
        observed_evidence=[
            "Further clarification is required.",
        ],
        requires_human_review=True,
    )

    decision = HITLDecision(
        decision="approve",
        reviewer="data_scientist",
        rationale="The review has been considered and progression is approved.",
    )

    assert evaluate_and_transition_data_understanding(
        review,
        decision,
    ) == "next_stage"


def test_revision_request_returns_revision_state():
    review = DataUnderstandingReview(
        requires_human_review=False,
    )

    decision = HITLDecision(
        decision="request_revision",
        reviewer="data_scientist",
        rationale="Additional explanation is required.",
    )

    assert evaluate_and_transition_data_understanding(
        review,
        decision,
    ) == "revision"


def test_rejection_returns_blocked_state():
    review = DataUnderstandingReview(
        requires_human_review=False,
    )

    decision = HITLDecision(
        decision="reject",
        reviewer="data_scientist",
        rationale="Evidence requires further investigation.",
    )

    assert evaluate_and_transition_data_understanding(
        review,
        decision,
    ) == "blocked"
