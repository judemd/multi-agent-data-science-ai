import pytest
from pydantic import ValidationError

from tools.artifact_validator import validate_problem_framing


def test_valid_problem_framing_output_is_accepted():
    artifact = validate_problem_framing(
        {
            "business_problem": "Customer churn is increasing.",
            "business_objective": "Improve retention decision-making.",
            "target_outcome": "Identify customers at risk of leaving.",
            "success_criteria": [
                "Provide actionable churn-risk insights."
            ],
            "assumptions": [],
            "constraints": [],
            "risks": [],
            "questions_for_human": [],
        }
    )

    assert artifact.business_problem == "Customer churn is increasing."


def test_invalid_problem_framing_output_is_rejected():
    with pytest.raises(ValidationError):
        validate_problem_framing(
            {
                "business_problem": "",
                "business_objective": "Improve retention.",
                "target_outcome": "Reduce churn.",
            }
        )
