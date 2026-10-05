from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.hitl_decision import HITLDecision
from workflow.data_preparation_transitions import (
    evaluate_and_transition_data_preparation,
)


def build_review(
    *,
    requires_human_review: bool,
) -> DataPreparationReview:
    return DataPreparationReview(
        observed_evidence=[
            "Missing revenue values were detected.",
        ],
        proposed_actions=[
            DataPreparationAction(
                operation="impute_missing",
                column="revenue",
                strategy="median",
                reason="Investigate the appropriate missing-value treatment.",
            ),
        ],
        rationale=[
            "The artifact contains missing revenue values.",
        ],
        requires_human_review=requires_human_review,
    )


def build_decision(decision: str) -> HITLDecision:
    return HITLDecision(
        decision=decision,
        reviewer="data_scientist",
        rationale="Decision based on the preparation review.",
    )


def test_approved_preparation_can_progress_when_review_allows_it():
    result = evaluate_and_transition_data_preparation(
        build_review(requires_human_review=False),
        build_decision("approve"),
    )

    assert result == "modeling"


def test_review_requiring_human_review_can_progress_after_approval():
    result = evaluate_and_transition_data_preparation(
        build_review(requires_human_review=True),
        build_decision("approve"),
    )

    assert result == "modeling"


def test_revision_request_returns_revision():
    result = evaluate_and_transition_data_preparation(
        build_review(requires_human_review=False),
        build_decision("request_revision"),
    )

    assert result == "revision"


def test_rejection_blocks_preparation():
    result = evaluate_and_transition_data_preparation(
        build_review(requires_human_review=False),
        build_decision("reject"),
    )

    assert result == "blocked"


