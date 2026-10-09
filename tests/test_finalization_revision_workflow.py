"""Tests for human-controlled Finalization revision resolution."""

import pytest

from domain.finalization_revision import (
    FinalizationFeedbackResolution,
    FinalizationRevisionResolution,
)
from tests.test_finalization_workflow import (
    human_decision,
    pending_handoff,
)
from workflow.finalization_revision_workflow import (
    resolve_finalization_revision,
)
from workflow.finalization_workflow import apply_handoff_decision
from workflow.states import WorkflowState


def test_revision_resolution_rejects_missing_feedback_without_mutation(
    tmp_path,
):
    project = pending_handoff(tmp_path)

    apply_handoff_decision(
        project,
        human_decision(
            "request_revision",
            [
                "Clarify model limitations.",
                "Document the handoff risks.",
            ],
        ),
    )

    assert project.current_state == WorkflowState.FINALIZATION

    resolution = FinalizationRevisionResolution(
        reviewer="Resolution Reviewer",
        rationale="Reviewed the proposed corrections.",
        items=[
            FinalizationFeedbackResolution(
                feedback="Clarify model limitations.",
                resolution="Expanded the limitations explanation.",
                status="addressed",
            ),
        ],
    )

    original_state = project.model_dump()

    with pytest.raises(
        ValueError,
        match="Every original Finalization feedback item",
    ):
        resolve_finalization_revision(project, resolution)

    assert project.model_dump() == original_state
    assert project.handoff_decision.decision == "request_revision"
    assert project.finalization_revision_resolution is None
    assert project.finalization_revision_history == []
    assert project.handoff_decision_history == []
def test_resolved_revision_rebuilds_evidence_and_requires_new_approval(
    tmp_path,
):
    from tools.finalization_evidence_fingerprint import (
        fingerprint_finalization_evidence,
    )
    from workflow.finalization_stage import run_finalization_stage

    project = pending_handoff(tmp_path)
    original_fingerprint = project.finalization_evidence_fingerprint

    apply_handoff_decision(
        project,
        human_decision(
            "request_revision",
            [
                "Clarify model limitations.",
                "Document the handoff risks.",
            ],
        ),
    )

    resolution = FinalizationRevisionResolution(
        reviewer="Resolution Reviewer",
        rationale="Verified that both documentation requests were addressed.",
        items=[
            FinalizationFeedbackResolution(
                feedback="Clarify model limitations.",
                resolution="Added a detailed explanation of model limitations.",
                status="addressed",
            ),
            FinalizationFeedbackResolution(
                feedback="Document the handoff risks.",
                resolution="Expanded the handoff risk documentation.",
                status="addressed",
            ),
        ],
    )

    resolve_finalization_revision(project, resolution)

    assert project.current_state == WorkflowState.FINALIZATION
    assert project.handoff_decision is None
    assert len(project.handoff_decision_history) == 1
    assert project.handoff_decision_history[0].decision == "request_revision"
    assert project.handoff_decision_history[0].feedback == [
        "Clarify model limitations.",
        "Document the handoff risks.",
    ]
    assert project.finalization_revision_resolution == resolution
    assert project.finalization_revision_history == [resolution]

    rebuilt = run_finalization_stage(project)

    assert rebuilt.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert rebuilt.handoff_decision is None
    assert rebuilt.finalization is not None
    assert rebuilt.finalization.revision_resolution == resolution
    assert rebuilt.finalization.deployment_approved is False
    assert rebuilt.finalization.requires_human_handoff_approval is True

    assert rebuilt.finalization_evidence_fingerprint == (
        fingerprint_finalization_evidence(rebuilt.finalization)
    )
    assert rebuilt.finalization_evidence_fingerprint != original_fingerprint

    assert len(rebuilt.handoff_decision_history) == 1
    assert rebuilt.finalization_revision_history == [resolution]
def test_revision_resolution_rejects_unresolved_items(tmp_path):
    project = pending_handoff(tmp_path)

    apply_handoff_decision(
        project,
        human_decision(
            "request_revision",
            [
                "Clarify model limitations.",
                "Document the handoff risks.",
            ],
        ),
    )

    resolution = FinalizationRevisionResolution(
        reviewer="Resolution Reviewer",
        rationale="Reviewed the current documentation.",
        items=[
            FinalizationFeedbackResolution(
                feedback="Clarify model limitations.",
                resolution="Added an explanation of model limitations.",
                status="addressed",
            ),
            FinalizationFeedbackResolution(
                feedback="Document the handoff risks.",
                resolution="Risk documentation is still incomplete.",
                status="unresolved",
            ),
        ],
    )

    original_state = project.model_dump()

    with pytest.raises(
        ValueError,
        match="All Finalization revision items must be addressed",
    ):
        resolve_finalization_revision(project, resolution)

    assert project.model_dump() == original_state
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.handoff_decision.decision == "request_revision"
    assert project.finalization is None
    assert project.finalization_evidence_fingerprint is None
    assert project.finalization_revision_resolution is None
    assert project.finalization_revision_history == []
    assert project.handoff_decision_history == []
def test_second_revision_cannot_reuse_previous_resolution(tmp_path):
    from workflow.finalization_stage import run_finalization_stage

    project = pending_handoff(tmp_path)

    apply_handoff_decision(
        project,
        human_decision(
            "request_revision",
            ["Clarify model limitations."],
        ),
    )

    first_resolution = FinalizationRevisionResolution(
        reviewer="First Resolution Reviewer",
        rationale="Confirmed the first clarification.",
        items=[
            FinalizationFeedbackResolution(
                feedback="Clarify model limitations.",
                resolution="Expanded the model limitations explanation.",
                status="addressed",
            ),
        ],
    )

    resolve_finalization_revision(project, first_resolution)
    run_finalization_stage(project)

    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.finalization.revision_resolution == first_resolution

    apply_handoff_decision(
        project,
        human_decision(
            "request_revision",
            ["Explain the remaining operational risks."],
        ),
    )

    assert project.current_state == WorkflowState.FINALIZATION
    assert project.finalization_revision_resolution is None
    assert project.finalization is None
    assert project.finalization_evidence_fingerprint is None
    assert project.finalization_revision_history == [first_resolution]

    state_before_rebuild = project.model_dump()

    with pytest.raises(
        ValueError,
        match="revision feedback remains unresolved",
    ):
        run_finalization_stage(project)

    assert project.model_dump() == state_before_rebuild

    second_resolution = FinalizationRevisionResolution(
        reviewer="Second Resolution Reviewer",
        rationale="Confirmed the operational risk explanation.",
        items=[
            FinalizationFeedbackResolution(
                feedback="Explain the remaining operational risks.",
                resolution="Documented the remaining operational risks.",
                status="addressed",
            ),
        ],
    )

    resolve_finalization_revision(project, second_resolution)

    assert len(project.handoff_decision_history) == 2
    assert project.handoff_decision_history[0].feedback == [
        "Clarify model limitations."
    ]
    assert project.handoff_decision_history[1].feedback == [
        "Explain the remaining operational risks."
    ]
    assert project.finalization_revision_history == [
        first_resolution,
        second_resolution,
    ]

    run_finalization_stage(project)

    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.handoff_decision is None
    assert project.finalization.revision_resolution == second_resolution
    assert project.finalization.deployment_approved is False