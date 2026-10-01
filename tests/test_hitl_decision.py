import pytest
from pydantic import ValidationError

from domain.hitl_decision import HITLDecision


def test_accepts_approval():
    decision = HITLDecision(
        decision="approve",
        reviewer="data_scientist",
        rationale="The findings are sufficiently supported.",
    )

    assert decision.decision == "approve"
    assert decision.reviewer == "data_scientist"
    assert decision.feedback == []


def test_accepts_rejection():
    decision = HITLDecision(
        decision="reject",
        reviewer="data_scientist",
        rationale="The evidence requires further investigation.",
        feedback=[
            "Investigate duplicate customer identifiers.",
        ],
    )

    assert decision.decision == "reject"
    assert len(decision.feedback) == 1


def test_accepts_revision_request():
    decision = HITLDecision(
        decision="request_revision",
        reviewer="data_scientist",
        rationale="Clarify the treatment of missing revenue.",
        feedback=[
            "Explain whether missing revenue is structurally expected.",
        ],
    )

    assert decision.decision == "request_revision"


def test_rejects_unknown_decision():
    with pytest.raises(ValidationError):
        HITLDecision(
            decision="maybe",
            reviewer="data_scientist",
            rationale="Unclear.",
        )


def test_rejects_empty_reviewer():
    with pytest.raises(ValidationError):
        HITLDecision(
            decision="approve",
            reviewer="",
            rationale="Approved.",
        )


def test_rejects_empty_rationale():
    with pytest.raises(ValidationError):
        HITLDecision(
            decision="approve",
            reviewer="data_scientist",
            rationale="",
        )
