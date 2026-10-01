from domain.hitl_decision import HITLDecision
from workflow.data_understanding_transitions import (
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
    )

    assert transition_after_hitl(decision) == "revision"
