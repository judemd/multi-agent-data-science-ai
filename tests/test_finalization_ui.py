"""Streamlit tests for Finalization evidence review."""

from streamlit.testing.v1 import AppTest

from tests.test_finalization_stage import build_project
from workflow.finalization_stage import run_finalization_stage


def test_finalization_review_displays_handoff_evidence(tmp_path):
    project = run_finalization_stage(build_project(tmp_path))

    def render(artifact, fingerprint):
        from ui_finalization import render_finalization_review

        render_finalization_review(artifact, fingerprint)

    at = AppTest.from_function(
        render,
        args=(
            project.finalization,
            project.finalization_evidence_fingerprint,
        ),
    ).run()

    assert not at.exception

    rendered_text = " ".join(
        str(element.value)
        for collection in (
            at.title,
            at.subheader,
            at.markdown,
            at.warning,
            at.info,
            at.caption,
            at.code,
        )
        for element in collection
    )

    assert "Finalization" in rendered_text
    assert "Finalization Test" in rendered_text
    assert "Logistic Regression" in rendered_text
    assert "Single holdout split." in rendered_text
    assert "does not authorize production deployment" in rendered_text
    assert project.finalization_evidence_fingerprint in rendered_text

    assert project.handoff_decision is None
def test_finalization_ui_approval_persists_completed_handoff(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    from domain.project_state import ProjectState
    from tools.project_store import ProjectStore
    from workflow.states import WorkflowState

    project = run_finalization_stage(build_project(tmp_path))
    original_fingerprint = project.finalization_evidence_fingerprint

    def render(project):
        import streamlit as st
        from ui_finalization import render_finalization_hitl_controls

        st.session_state["project_state"] = project
        render_finalization_hitl_controls(project)

    at = AppTest.from_function(
        render,
        args=(project,),
    ).run()

    assert not at.exception
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL

    at.text_input[0].set_value("Final Reviewer")
    at.text_area[0].set_value(
        "Reviewed the evidence, risks, and limitations."
    )

    at.button[0].click().run()

    assert not at.exception
    assert project.current_state == WorkflowState.COMPLETE
    assert project.handoff_decision.decision == "approve"
    assert project.handoff_decision.reviewer == "Final Reviewer"
    assert project.finalization_evidence_fingerprint == original_fingerprint
    assert project.finalization.deployment_approved is False

    restored = ProjectStore().load(project.project_id)

    assert isinstance(restored, ProjectState)
    assert restored.current_state == WorkflowState.COMPLETE
    assert restored.handoff_decision.decision == "approve"
    assert restored.finalization_evidence_fingerprint == original_fingerprint
    assert restored.finalization.deployment_approved is False
def test_finalization_ui_revision_persists_feedback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    from tools.project_store import ProjectStore
    from workflow.states import WorkflowState

    project = run_finalization_stage(build_project(tmp_path))

    def render(project):
        import streamlit as st
        from ui_finalization import render_finalization_hitl_controls

        st.session_state["project_state"] = project
        render_finalization_hitl_controls(project)

    at = AppTest.from_function(
        render,
        args=(project,),
    ).run()

    assert not at.exception

    at.text_input[0].set_value("Final Reviewer")
    at.text_area[0].set_value(
        "The handoff needs clearer documentation."
    )
    at.text_area[1].set_value(
        "Explain the low model recall and its business implications."
    )

    at.button[1].click().run()

    assert not at.exception
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.handoff_decision.decision == "request_revision"
    assert project.handoff_decision.feedback == [
        "Explain the low model recall and its business implications."
    ]
    assert project.finalization is None
    assert project.finalization_evidence_fingerprint is None

    restored = ProjectStore().load(project.project_id)

    assert restored.current_state == WorkflowState.FINALIZATION
    assert restored.handoff_decision.decision == "request_revision"
    assert restored.handoff_decision.feedback == [
        "Explain the low model recall and its business implications."
    ]
    assert restored.finalization is None
    assert restored.finalization_evidence_fingerprint is None
def test_finalization_ui_rejection_persists_blocked_handoff(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    from tools.project_store import ProjectStore
    from workflow.states import WorkflowState

    project = run_finalization_stage(build_project(tmp_path))
    original_fingerprint = project.finalization_evidence_fingerprint

    def render(project):
        import streamlit as st
        from ui_finalization import render_finalization_hitl_controls

        st.session_state["project_state"] = project
        render_finalization_hitl_controls(project)

    at = AppTest.from_function(
        render,
        args=(project,),
    ).run()

    assert not at.exception
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL

    at.text_input[0].set_value("Final Reviewer")
    at.text_area[0].set_value(
        "The unresolved model limitations make this handoff unacceptable."
    )

    at.button[2].click().run()

    assert not at.exception
    assert project.current_state == WorkflowState.BLOCKED
    assert project.handoff_decision.decision == "reject"
    assert project.handoff_decision.reviewer == "Final Reviewer"
    assert project.finalization is not None
    assert project.finalization_evidence_fingerprint == original_fingerprint
    assert project.finalization.deployment_approved is False

    restored = ProjectStore().load(project.project_id)

    assert restored.current_state == WorkflowState.BLOCKED
    assert restored.handoff_decision.decision == "reject"
    assert restored.handoff_decision.rationale == (
        "The unresolved model limitations make this handoff unacceptable."
    )
    assert restored.finalization is not None
    assert restored.finalization_evidence_fingerprint == original_fingerprint
    assert restored.finalization.deployment_approved is False