import json

import pytest
from pydantic import ValidationError

from agents.data_preparation import data_preparation_agent
from agents.data_preparation_runner import _parse_review
from domain.data_preparation_review import DataPreparationReview


def test_preparation_agent_constrains_output_to_review_schema():
    assert data_preparation_agent.output_schema is DataPreparationReview


def test_parse_review_accepts_category_normalization_without_custom_mapping():
    review = _parse_review(json.dumps({
        "proposed_actions": [{
            "operation": "normalize_categories",
            "column": "contract",
            "strategy": None,
            "reason": "Observed case variants need human review.",
        }],
        "requires_human_review": True,
    }))
    assert review.proposed_actions[0].strategy is None
    assert review.requires_human_review is True


@pytest.mark.parametrize("strategy", [
    {"month-to-month": ["MONTH-TO-MONTH", "Month-to-month"]},
    ["MONTH-TO-MONTH", "Month-to-month"],
])
def test_parse_review_rejects_custom_mapping_in_strategy(strategy):
    with pytest.raises(ValidationError):
        _parse_review(json.dumps({
            "proposed_actions": [{
                "operation": "normalize_categories",
                "column": "contract",
                "strategy": strategy,
                "reason": "Observed case variants need human review.",
            }],
        }))
