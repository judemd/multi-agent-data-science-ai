from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview


def test_data_preparation_review_accepts_structured_response():
    review = DataPreparationReview(
        observed_evidence=[
            "Revenue contains missing values.",
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
            "Missing revenue values may affect downstream modeling.",
        ],
        risks_and_limitations=[
            "The correct treatment depends on business meaning.",
        ],
        human_review_questions=[
            "Should missing revenue values be retained or imputed?",
        ],
        requires_human_review=True,
    )

    assert review.observed_evidence
    assert review.proposed_actions
    assert review.rationale
    assert review.requires_human_review is True


