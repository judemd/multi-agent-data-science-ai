from domain.data_understanding_review import DataUnderstandingReview


def test_data_understanding_review_defaults_to_human_review():
    review = DataUnderstandingReview()

    assert review.requires_human_review is True
    assert review.observed_evidence == []
    assert review.interpretation == []
    assert review.risks_and_limitations == []
    assert review.human_review_questions == []
    assert review.recommended_next_investigation == []


def test_data_understanding_review_accepts_structured_findings():
    review = DataUnderstandingReview(
        observed_evidence=[
            "Missing values were detected in revenue."
        ],
        interpretation=[
            "Revenue completeness requires investigation."
        ],
        risks_and_limitations=[
            "Missing revenue may affect downstream analysis."
        ],
        human_review_questions=[
            "Should missing revenue values be retained?"
        ],
        recommended_next_investigation=[
            "Investigate the business meaning of missing revenue."
        ],
    )

    assert len(review.observed_evidence) == 1
    assert len(review.interpretation) == 1
    assert len(review.risks_and_limitations) == 1
    assert len(review.human_review_questions) == 1
    assert len(review.recommended_next_investigation) == 1
    assert review.requires_human_review is True


def test_human_review_can_explicitly_be_marked_complete():
    review = DataUnderstandingReview(
        observed_evidence=["No material quality issue detected."],
        requires_human_review=False,
    )

    assert review.requires_human_review is False
