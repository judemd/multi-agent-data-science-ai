import streamlit as st
import pandas as pd

from tools.data_preparation_executor import (
    execute_data_preparation_actions,
)

from domain.data_preparation_review import DataPreparationReview
from domain.hitl_decision import HITLDecision
from tools.data_preparation_treatment_plan_builder import (
    build_data_preparation_treatment_plan,
)
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

    artifact = project.data_preparation

    if artifact is None:
        st.error("No Data Preparation evidence is available.")
        return

    st.markdown("#### Detected Data Quality Issues")

    if not artifact.issues:
        st.info("No typed data quality issues were detected.")
    else:
        st.caption(
            f"{len(artifact.issues)} issue(s) require a treatment decision "
            "before preparation can be approved."
        )

        for issue in artifact.issues:
            issue_label = issue.column or "Entire dataset"

            with st.expander(
                f"{issue.issue_type} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â {issue_label}",
                expanded=False,
            ):
                st.write(f"**Issue ID:** {issue.issue_id}")
                st.write(f"**Evidence:** {issue.evidence}")
                st.write(
                    "**Permitted treatments:** "
                    + ", ".join(issue.allowed_treatments)
                )

                if issue.requires_explicit_human_decision:
                    st.warning("Explicit human decision required.")

    if is_blocked:
        st.error(
            "The workflow is blocked by the human rejection. "
            "The review and decision history are preserved below."
        )

    with st.form("data_preparation_human_review"):
        selected_treatments = {}

        if artifact.issues:
            st.markdown("#### Human Treatment Decisions")
            st.caption(
                "Choose a treatment and explain your reasoning for "
                "every detected issue. No treatment is selected automatically."
            )

            for issue in artifact.issues:
                issue_id = issue.issue_id

                if not issue_id:
                    st.error(
                        "A detected issue has no stable ID. "
                        "Preparation approval cannot proceed."
                    )
                    return

                st.markdown(
                    f"**{issue.issue_type} ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â "
                    f"{issue.column or 'Entire dataset'}**"
                )

                treatment = st.selectbox(
                    f"Treatment for {issue_id} *",
                    options=[None, *issue.allowed_treatments],
                    format_func=lambda value: (
                        "Select a treatment..."
                        if value is None
                        else value.replace("_", " ").title()
                    ),
                    key=f"preparation_treatment_{issue_id}",
                    disabled=is_blocked,
                )

                treatment_rationale = st.text_area(
                    f"Treatment rationale for {issue_id} *",
                    key=f"preparation_treatment_rationale_{issue_id}",
                    placeholder="Explain why this treatment is appropriate.",
                    disabled=is_blocked,
                )

                selected_treatments[issue_id] = (
                    treatment,
                    treatment_rationale,
                )

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

    if decision_value == "approve":
        dataset_fingerprint = (
            project.data_preparation_dataset_fingerprint
        )
        evidence_fingerprint = (
            project.data_preparation_evidence_fingerprint
        )

        if not dataset_fingerprint or not evidence_fingerprint:
            st.error(
                "Preparation fingerprints are missing. "
                "The review must be regenerated before approval."
            )
            return

        try:
            treatment_plan = build_data_preparation_treatment_plan(
                artifact=artifact,
                dataset_fingerprint=dataset_fingerprint,
                evidence_fingerprint=evidence_fingerprint,
                reviewer=reviewer,
                selections=selected_treatments,
            )
        except ValueError as exc:
            st.error(f"Invalid treatment plan: {exc}")
            return

    decision = HITLDecision(
        decision=decision_value,
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        feedback=feedback,
    )

    previous_treatment_plan = project.data_preparation_treatment_plan

    if decision_value == "approve":
        project.data_preparation_treatment_plan = treatment_plan

    try:
        apply_data_preparation_decision(
            project,
            decision,
            dataframe,
        )
    except (ValueError, OSError) as exc:
        project.data_preparation_treatment_plan = previous_treatment_plan
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