import pytest
from pydantic import ValidationError

from domain.problem_framing import ProblemFramingArtifact


def test_problem_framing_artifact_accepts_valid_data():
    artifact = ProblemFramingArtifact(
        business_problem="Customer churn is reducing recurring revenue.",
        business_objective="Identify customers at risk of churn.",
        target_outcome="Support proactive retention decisions.",
        success_criteria=["Produce an actionable churn-risk assessment."],
    )

    assert artifact.business_problem == "Customer churn is reducing recurring revenue."
    assert len(artifact.success_criteria) == 1


def test_problem_framing_artifact_rejects_empty_business_problem():
    with pytest.raises(ValidationError):
        ProblemFramingArtifact(
            business_problem="",
            business_objective="Identify customers at risk of churn.",
            target_outcome="Support proactive retention decisions.",
        )
