"""Streamlit tests for human Finalization revision resolution."""

from streamlit.testing.v1 import AppTest

from tests.test_finalization_workflow import (
    human_decision,
    pending_handoff,
)
from workflow.finalization_workflow import apply_handoff_decision
from workflow.states import WorkflowState


def test_revision_ui_requires_resolution_for_every_feedback_item(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

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

    original_state = project.model_dump()
    saved_projects = []

    monkeypatch.setattr(
        "ui_finalization_revision.ProjectStore.save",
        lambda self, saved_project: saved_projects.append(
            saved_project.model_copy(deep=True)
        ),
    )

    def render(project):
        import streamlit as st
        from ui_finalization_revision import (
            render_finalization_revision_controls,
            render_finalization_revision_review,
        )

        st.session_state["project_state"] = project
        render_finalization_revision_review(project)
        render_finalization_revision_controls(project)

    at = AppTest.from_function(
        render,
        args=(project,),
    ).run()

    assert not at.exception
    assert any(
        "Clarify model limitations." in str(item.value)
        for item in at.markdown
    )

    at.text_input[0].set_value("Resolution Reviewer")
    at.text_area[0].set_value("Reviewed the proposed corrections.")
    at.text_area[1].set_value(
        "Added a detailed explanation of model limitations."
    )
    # Leave the second requested change unanswered.

    at.button[0].click().run()

    assert not at.exception
    assert any(
        "Every feedback item requires a documented resolution."
        in item.value
        for item in at.warning
    )

    assert project.model_dump() == original_state
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.handoff_decision.decision == "request_revision"
    assert project.finalization_revision_resolution is None
    assert saved_projects == []
def test_revision_ui_persists_completed_resolutions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    from tools.project_store import ProjectStore

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

    def render(project):
        import streamlit as st
        from ui_finalization_revision import (
            render_finalization_revision_controls,
        )

        st.session_state["project_state"] = project
        render_finalization_revision_controls(project)

    at = AppTest.from_function(
        render,
        args=(project,),
    ).run()

    assert not at.exception

    at.text_input[0].set_value("Resolution Reviewer")
    at.text_area[0].set_value(
        "Reviewed and confirmed both documentation changes."
    )
    at.text_area[1].set_value(
        "Expanded the explanation of model limitations."
    )
    at.text_area[2].set_value(
        "Documented the remaining handoff risks."
    )

    at.button[0].click().run()

    assert not at.exception
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.handoff_decision is None
    assert project.finalization is None
    assert project.finalization_evidence_fingerprint is None

    assert len(project.handoff_decision_history) == 1
    assert project.handoff_decision_history[0].decision == "request_revision"
    assert project.handoff_decision_history[0].feedback == [
        "Clarify model limitations.",
        "Document the handoff risks.",
    ]

    resolution = project.finalization_revision_resolution

    assert resolution is not None
    assert resolution.reviewer == "Resolution Reviewer"
    assert len(resolution.items) == 2
    assert all(item.status == "addressed" for item in resolution.items)
    assert project.finalization_revision_history == [resolution]

    restored = ProjectStore().load(project.project_id)

    assert restored.current_state == WorkflowState.FINALIZATION
    assert restored.handoff_decision is None
    assert restored.handoff_decision_history[0].decision == "request_revision"
    assert restored.finalization_revision_resolution == resolution
    assert restored.finalization_revision_history == [resolution]
    assert restored.finalization is None
    assert restored.finalization_evidence_fingerprint is None