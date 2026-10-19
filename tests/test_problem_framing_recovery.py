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


def test_recovery_preserves_optional_business_context():
    artifact = reconstruct_problem_framing({
        "business_problem": "Churn", "business_objective": "Prioritize retention",
        "desired_outcome": "Retention pilot", "success_metrics": "Measure pilot lift",
        "constraints": "Weekly outreach capacity", "risks": "Data drift",
    })
    assert artifact.success_criteria == ["Measure pilot lift"]
    assert artifact.constraints == ["Weekly outreach capacity"]
    assert artifact.risks == ["Data drift"]


def build_recovery_inputs():
    from domain.hitl_decision import HITLDecision
    from domain.project_state import ProjectState
    from workflow.states import WorkflowState

    project = ProjectState(
        project_id="project-recovery", project_name="Churn",
        current_state=WorkflowState.FINALIZATION,
        selected_model="Random Forest", selected_feature_columns=["tenure"],
    )
    artifact = reconstruct_problem_framing({
        "business_problem": "Churn", "business_objective": "Prioritize retention",
        "desired_outcome": "Retention pilot",
    })
    decision = HITLDecision(decision="approve", reviewer="Reviewer", rationale="Reviewed")
    return project, artifact, decision


def test_recovery_records_new_approval_and_preserves_project():
    from workflow.problem_framing_recovery import approve_recovered_framing

    project, artifact, decision = build_recovery_inputs()
    recovered = approve_recovered_framing(project, artifact, decision)
    assert project.problem_framing is None
    assert recovered.problem_framing == artifact
    assert recovered.problem_framing_recovery_decision == decision
    assert "does not establish historical" in recovered.problem_framing_provenance
    assert recovered.selected_model == project.selected_model
    assert recovered.selected_feature_columns == project.selected_feature_columns
    assert recovered.current_state == project.current_state


@pytest.mark.parametrize("change", ["reject", "blank_reviewer", "blank_rationale", "wrong_state", "existing"])
def test_recovery_requires_explicit_valid_approval(change):
    from workflow.problem_framing_recovery import approve_recovered_framing
    from workflow.states import WorkflowState

    project, artifact, decision = build_recovery_inputs()
    if change == "reject":
        decision.decision = "reject"
    elif change == "blank_reviewer":
        decision.reviewer = " "
    elif change == "blank_rationale":
        decision.rationale = " "
    elif change == "wrong_state":
        project.current_state = WorkflowState.MODELING
    else:
        project.problem_framing = artifact
    with pytest.raises(ValueError):
        approve_recovered_framing(project, artifact, decision)
