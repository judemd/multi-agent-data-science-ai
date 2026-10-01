import pytest
from pydantic import ValidationError

from tools.data_understanding_review_validator import (
    validate_data_understanding_review,
)


def test_valid_review_is_accepted():
    review = validate_data_understanding_review(
        {
            "observed_evidence": [
                "Missing revenue values were detected."
            ],
            "interpretation": [
                "Revenue completeness requires investigation."
            ],
            "risks_and_limitations": [
                "Missing revenue may affect downstream analysis."
            ],
            "human_review_questions": [
                "How should missing revenue be treated?"
            ],
            "recommended_next_investigation": [
                "Investigate the business meaning of missing revenue."
            ],
            "requires_human_review": True,
        }
    )

    assert review.requires_human_review is True
    assert len(review.observed_evidence) == 1


def test_empty_review_is_still_valid_but_requires_review():
    review = validate_data_understanding_review({})

    assert review.requires_human_review is True
    assert review.observed_evidence == []


def test_invalid_review_field_type_is_rejected():
    with pytest.raises(ValidationError):
        validate_data_understanding_review(
            {
                "observed_evidence": "not a list",
            }
        )
