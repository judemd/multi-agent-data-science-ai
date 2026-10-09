"""Tests for the human-controlled Finalization handoff gate."""

import pytest

from domain.finalization import FinalizationArtifact
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tests.test_finalization_stage import build_project
from workflow.finalization_stage import run_finalization_stage
from workflow.finalization_workflow import apply_handoff_decision
from workflow.states import WorkflowState


def pending_handoff(tmp_path):
    project = build_project(tmp_path)
    return run_finalization_stage(project)


def human_decision(action, feedback=None):
    return HITLDecision(
        decision=action,
        reviewer="Human Reviewer",
        rationale="Reviewed the handoff evidence.",
        feedback=feedback or [],
    )


def test_approve_completes_handoff_without_deployment(tmp_path):
    project = pending_handoff(tmp_path)

    result = apply_handoff_decision(
        project,
        human_decision("approve"),
    )

    assert result.current_state == WorkflowState.COMPLETE
    assert result.handoff_decision.decision == "approve"
    assert result.finalization is not None
    assert result.finalization.deployment_approved is False
    assert result.finalization.requires_human_handoff_approval is True


def test_request_revision_returns_to_finalization(tmp_path):
    project = pending_handoff(tmp_path)

    result = apply_handoff_decision(
        project,
        human_decision(
            "request_revision",
            ["Clarify the limitations in the handoff."],
        ),
    )

    assert result.current_state == WorkflowState.FINALIZATION
    assert result.finalization is None
    assert result.handoff_decision.decision == "request_revision"


def test_reject_blocks_handoff(tmp_path):
    project = pending_handoff(tmp_path)

    result = apply_handoff_decision(
        project,
        human_decision("reject"),
    )

    assert result.current_state == WorkflowState.BLOCKED
    assert result.handoff_decision.decision == "reject"
    assert result.finalization is not None


def test_handoff_requires_pending_state_without_mutation(tmp_path):
    project = build_project(tmp_path)
    original = project.model_dump()

    with pytest.raises(ValueError, match="AWAITING_HANDOFF_APPROVAL"):
        apply_handoff_decision(project, human_decision("approve"))

    assert project.model_dump() == original


def test_handoff_requires_evidence_without_mutation(tmp_path):
    project = pending_handoff(tmp_path)
    project.finalization = None
    original = project.model_dump()

    with pytest.raises(ValueError, match="Finalization evidence"):
        apply_handoff_decision(project, human_decision("approve"))

    assert project.model_dump() == original


def test_handoff_rejects_blank_reviewer_without_mutation(tmp_path):
    project = pending_handoff(tmp_path)
    original = project.model_dump()

    decision = human_decision("approve")
    decision.reviewer = "   "

    with pytest.raises(ValueError, match="reviewer"):
        apply_handoff_decision(project, decision)

    assert project.model_dump() == original


def test_handoff_revision_requires_feedback_without_mutation(tmp_path):
    project = pending_handoff(tmp_path)
    original = project.model_dump()

    with pytest.raises(ValueError, match="feedback"):
        apply_handoff_decision(
            project,
            human_decision("request_revision"),
        )

    assert project.model_dump() == original


def test_handoff_approval_survives_json_roundtrip(tmp_path):
    project = pending_handoff(tmp_path)
    approved = apply_handoff_decision(
        project,
        human_decision("approve"),
    )

    restored = ProjectState.model_validate_json(
        approved.model_dump_json()
    )

    assert restored.current_state == WorkflowState.COMPLETE
    assert restored.handoff_decision.decision == "approve"
    assert restored.finalization.deployment_approved is False
    assert restored.finalization.unresolved_risks
def test_handoff_rejects_changed_source_dataset(tmp_path):
    project = pending_handoff(tmp_path)
    original = project.model_dump()

    with open(project.dataset_path, "a", encoding="utf-8") as file:
        file.write("40,east,0\n")

    with pytest.raises(ValueError, match="source dataset fingerprint mismatch"):
        apply_handoff_decision(project, human_decision("approve"))

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.handoff_decision is None


def test_handoff_rejects_changed_prepared_dataset(tmp_path):
    project = pending_handoff(tmp_path)
    original = project.model_dump()

    with open(project.prepared_dataset_path, "a", encoding="utf-8") as file:
        file.write("40,east,0\n")

    with pytest.raises(ValueError, match="prepared dataset fingerprint mismatch"):
        apply_handoff_decision(project, human_decision("approve"))

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.handoff_decision is None
def test_handoff_rejects_modified_evaluation_metrics(tmp_path):
    project = pending_handoff(tmp_path)
    project.finalization.evaluation.model_metrics.accuracy = 0.99
    original = project.model_dump()

    with pytest.raises(ValueError, match="Finalization evidence fingerprint mismatch"):
        apply_handoff_decision(project, human_decision("approve"))

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.handoff_decision is None


def test_handoff_rejects_removed_evaluation_limitation(tmp_path):
    project = pending_handoff(tmp_path)
    project.finalization.unresolved_risks.clear()
    original = project.model_dump()

    with pytest.raises(ValueError, match="Finalization evidence fingerprint mismatch"):
        apply_handoff_decision(project, human_decision("approve"))

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.handoff_decision is None


def test_handoff_requires_recorded_evidence_fingerprint(tmp_path):
    project = pending_handoff(tmp_path)
    project.finalization_evidence_fingerprint = None
    original = project.model_dump()

    with pytest.raises(ValueError, match="recorded Finalization evidence fingerprint"):
        apply_handoff_decision(project, human_decision("approve"))

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.handoff_decision is None
def test_handoff_revision_blocks_rebuild_until_feedback_resolved(tmp_path):
    project = pending_handoff(tmp_path)

    original_fingerprint = project.finalization_evidence_fingerprint
    assert original_fingerprint is not None

    revised = apply_handoff_decision(
        project,
        human_decision(
            "request_revision",
            ["Clarify the limitations in the handoff."],
        ),
    )

    assert revised.current_state == WorkflowState.FINALIZATION
    assert revised.finalization is None
    assert revised.finalization_evidence_fingerprint is None
    assert revised.handoff_decision.decision == "request_revision"
    assert revised.handoff_decision.feedback == [
        "Clarify the limitations in the handoff."
    ]

    state_before_rebuild = revised.model_dump()

    with pytest.raises(
        ValueError,
        match="revision feedback remains unresolved",
    ):
        run_finalization_stage(revised)

    assert revised.model_dump() == state_before_rebuild
    assert revised.current_state == WorkflowState.FINALIZATION
    assert revised.finalization is None
    assert revised.finalization_evidence_fingerprint is None
    assert revised.handoff_decision.feedback == [
        "Clarify the limitations in the handoff."
    ]