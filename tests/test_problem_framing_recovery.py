"""Tests for transparent recovery of Problem Framing content."""

import pytest
from pydantic import ValidationError

from tools.problem_framing_recovery import reconstruct_problem_framing


def test_reconstructs_original_business_context_without_invention():
    context = {
        "business_problem": "Understand customer churn.",
        "business_objective": "Identify churn-related factors.",
        "desired_outcome": "Produce human-reviewed evidence.",
        "success_metrics": "",
        "constraints": "",
        "risks": "",
    }

    artifact = reconstruct_problem_framing(context)

    assert artifact.business_problem == context["business_problem"]
    assert artifact.business_objective == context["business_objective"]
    assert artifact.target_outcome == context["desired_outcome"]
    assert artifact.success_criteria == []
    assert artifact.assumptions == []
    assert artifact.constraints == []
    assert artifact.risks == []
    assert artifact.questions_for_human == []
    assert context["desired_outcome"] == "Produce human-reviewed evidence."


def test_missing_required_context_is_rejected():
    with pytest.raises(ValidationError):
        reconstruct_problem_framing(
            {
                "business_problem": "Understand churn.",
                "business_objective": "Identify factors.",
            }
        )
