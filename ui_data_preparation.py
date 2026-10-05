import streamlit as st
import pandas as pd

from tools.data_preparation_executor import (
    execute_data_preparation_actions,
)

from domain.data_preparation_review import DataPreparationReview
from domain.hitl_decision import HITLDecision
from tools.project_store import ProjectStore
from workflow.data_preparation_workflow import apply_data_preparation_decision
from workflow.states import WorkflowState

PROJECT_STORE = ProjectStore()


def render_data_preparation_review(
    review: DataPreparationReview,
) -> None:
    """Display the Data Preparation Agent's structured review."""

    st.subheader("Data Preparation Agent Review")

    st.caption(
        "The agent proposed preparation actions from deterministic evidence. "
        "Human approval is required before any preparation is performed."
    )

    observed_tab, actions_tab, rationale_tab, risks_tab, questions_tab = (
        st.tabs(
            [
                "Observed Evidence",
                "Proposed Actions",
                "Rationale",
                "Risks & Limitations",
                "Human Review Questions",
            ]
        )
    )

    with observed_tab:
        if review.observed_evidence:
            for item in review.observed_evidence:
                st.write(f"- {item}")
        else:
            st.info("No observed evidence was returned.")

    with actions_tab:
        if review.proposed_actions:
            for item in review.proposed_actions:
                st.write(f"- {item}")
        else:
            st.info("No preparation actions were proposed.")

    with rationale_tab:
        if review.rationale:
            for item in review.rationale:
                st.write(f"- {item}")
        else:
            st.info("No rationale was returned.")

    with risks_tab:
        if review.risks_and_limitations:
            for item in review.risks_and_limitations:
                st.warning(item)
        else:
            st.success("No material risks or limitations were identified.")

    with questions_tab:
        if review.human_review_questions:
            for item in review.human_review_questions:
                st.write(f"- {item}")
        else:
            st.info("No human review questions were returned.")

    if review.requires_human_review:
        st.warning(
            "Human review is required before preparation can proceed."
        )


def render_data_preparation_hitl_controls(
    project,
    dataframe,
) -> None:
    """Collect and apply the human Data Preparation decision."""

    review = project.data_preparation_review

    if review is None:
        st.error("No Data Preparation review is available.")
        return

    is_blocked = project.current_state == WorkflowState.BLOCKED
    existing_decision = project.data_preparation_decision

    if existing_decision is not None:
        if "data_preparation_reviewer" not in st.session_state:
            st.session_state["data_preparation_reviewer"] = (
                existing_decision.reviewer
            )

        if "data_preparation_review_rationale" not in st.session_state:
            st.session_state["data_preparation_review_rationale"] = (
                existing_decision.rationale
            )

        if "data_preparation_feedback" not in st.session_state:
            st.session_state["data_preparation_feedback"] = "\n".join(
                existing_decision.feedback
            )

    st.subheader("Data Preparation Human Review")

    if is_blocked:
        st.error(
            "The workflow is blocked by the human rejection. "
            "The review and decision history are preserved below."
        )

    with st.form("data_preparation_human_review"):
        reviewer = st.text_input(
            "Reviewer *",
            placeholder="Enter reviewer name or role.",
            key="data_preparation_reviewer",
            disabled=is_blocked,
        )

        rationale = st.text_area(
            "Reviewer rationale *",
            placeholder="Explain the reason for your decision.",
            key="data_preparation_review_rationale",
            disabled=is_blocked,
        )

        feedback_text = st.text_area(
            "Requested changes / feedback",
            placeholder="Required when requesting a revision.",
            key="data_preparation_feedback",
            disabled=is_blocked,
        )

        approve_column, revision_column, reject_column = st.columns(3)

        with approve_column:
            approve = st.form_submit_button(
                "Approve Preparation",
                type="primary",
                disabled=is_blocked,
            )

        with revision_column:
            request_revision = st.form_submit_button(
                "Request Revision",
                disabled=is_blocked,
            )

        with reject_column:
            reject = st.form_submit_button(
                "Reject",
                disabled=is_blocked,
            )

    decision_value = None

    if approve:
        decision_value = "approve"
    elif request_revision:
        decision_value = "request_revision"
    elif reject:
        decision_value = "reject"

    if decision_value is None:
        return

    if not reviewer.strip():
        st.warning("Enter the reviewer before submitting a decision.")
        return

    if not rationale.strip():
        st.warning("Enter a rationale before submitting a decision.")
        return

    feedback = []

    if decision_value == "request_revision":
        feedback = [
            item.strip()
            for item in feedback_text.splitlines()
            if item.strip()
        ]

        if not feedback:
            st.warning(
                "Provide at least one requested change before requesting "
                "a revision."
            )
            return

    decision = HITLDecision(
        decision=decision_value,
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        feedback=feedback,
    )

    try:
        apply_data_preparation_decision(
            project,
            decision,
            dataframe,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["project_state"] = project
    PROJECT_STORE.save(project)

    if project.current_state.value == "MODELING":
        st.success(
            "Preparation approved. The workflow may proceed to Modeling."
        )
        return

    if project.current_state.value == "BLOCKED":
        st.error(
            "The workflow has been blocked by the human rejection."
        )
        return

    if project.current_state.value == "DATA_PREPARATION":
        st.warning(
            "Revision requested. Further Data Preparation work is required."
        )
        st.rerun()