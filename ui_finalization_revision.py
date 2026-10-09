"""Streamlit review and resolution of Finalization revision feedback."""

import streamlit as st

from domain.finalization_revision import (
    FinalizationFeedbackResolution,
    FinalizationRevisionResolution,
)
from domain.project_state import ProjectState
from tools.project_store import ProjectStore
from workflow.finalization_revision_workflow import (
    resolve_finalization_revision,
)
from workflow.states import WorkflowState


def render_finalization_revision_review(project: ProjectState) -> None:
    """Display the outstanding human revision request."""

    if project.current_state != WorkflowState.FINALIZATION:
        return

    decision = project.handoff_decision

    if decision is None or decision.decision != "request_revision":
        return

    st.subheader("Finalization — Revision Required")

    st.warning(
        "The previous handoff was returned for revision. "
        "Finalization cannot rebuild until every feedback item "
        "has a documented, human-reviewed resolution."
    )

    st.write(f"**Original reviewer:** {decision.reviewer}")
    st.write(f"**Reason for revision:** {decision.rationale}")

    st.markdown("#### Requested changes")

    for index, feedback in enumerate(decision.feedback, start=1):
        st.write(f"{index}. {feedback}")

    st.info(
        "Resolving these requests does not approve the POC handoff "
        "or authorize production deployment. A revised handoff "
        "requires a new human approval."
    )
def render_finalization_revision_controls(project: ProjectState) -> None:
    """Collect human-reviewed responses to outstanding feedback."""

    if project.current_state != WorkflowState.FINALIZATION:
        return

    decision = project.handoff_decision

    if decision is None or decision.decision != "request_revision":
        return

    st.subheader("Resolve Finalization Revision")

    with st.form("finalization_revision_resolution"):
        reviewer = st.text_input(
            "Resolution reviewer *",
            key="finalization_resolution_reviewer",
        )
        rationale = st.text_area(
            "Overall resolution rationale *",
            key="finalization_resolution_rationale",
        )

        responses = []

        for index, feedback in enumerate(decision.feedback):
            st.markdown(f"**Requested change {index + 1}:** {feedback}")

            response = st.text_area(
                f"How was request {index + 1} addressed? *",
                key=f"finalization_resolution_item_{index}",
            )

            responses.append((feedback, response))

        submitted = st.form_submit_button(
            "Confirm Resolutions",
            type="primary",
        )

    if not submitted:
        return

    if not reviewer.strip() or not rationale.strip():
        st.warning("Reviewer and overall rationale are required.")
        return

    if any(not response.strip() for _, response in responses):
        st.warning("Every feedback item requires a documented resolution.")
        return

    resolution = FinalizationRevisionResolution(
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        items=[
            FinalizationFeedbackResolution(
                feedback=feedback,
                resolution=response.strip(),
                status="addressed",
            )
            for feedback, response in responses
        ],
    )

    try:
        resolve_finalization_revision(project, resolution)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["project_state"] = project
    ProjectStore().save(project)

    st.success(
        "Revision resolutions recorded. Finalization can now rebuild "
        "the handoff package for a new human approval."
    )
    st.rerun()