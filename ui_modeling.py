import streamlit as st

from domain.hitl_decision import HITLDecision
from domain.modeling_review import ModelingReview
from tools.project_store import ProjectStore
from workflow.modeling_workflow import (
    SUPPORTED_MODELS,
    apply_modeling_decision,
)
from workflow.states import WorkflowState

PROJECT_STORE = ProjectStore()


def render_modeling_review(
    review: ModelingReview,
) -> None:
    """Display the Modeling Agent's structured review."""

    st.subheader("Modeling Agent Review")

    st.caption(
        "The agent proposed modeling approaches from deterministic evidence. "
        "Human approval is required before model evaluation can proceed."
    )

    (
        observed_tab,
        models_tab,
        recommended_tab,
        rationale_tab,
        validation_tab,
        risks_tab,
        questions_tab,
    ) = st.tabs(
        [
            "Observed Evidence",
            "Proposed Models",
            "Recommended Model",
            "Rationale",
            "Validation Strategy",
            "Risks & Limitations",
            "Human Review Questions",
        ]
    )

    with observed_tab:
        if review.observed_evidence:
            for item in review.observed_evidence:
                st.write(f"- {item}")
        else:
            st.info("No observed evidence was returned.")

    with models_tab:
        if review.proposed_models:
            for model in review.proposed_models:
                st.markdown(f"**{model.name}**")
                st.write(
                    f"**Model family:** {model.model_family}"
                )
                st.write(f"**Rationale:** {model.rationale}")

                if model.considerations:
                    st.write("**Considerations:**")
                    for item in model.considerations:
                        st.write(f"- {item}")

                st.divider()
        else:
            st.info("No modeling candidates were proposed.")

    with recommended_tab:
        if review.recommended_model:
            st.success(review.recommended_model)
        else:
            st.warning(
                "The agent did not provide a recommended model."
            )

    with rationale_tab:
        if review.rationale:
            for item in review.rationale:
                st.write(f"- {item}")
        else:
            st.info("No rationale was returned.")

    with validation_tab:
        if review.validation_strategy:
            for item in review.validation_strategy:
                st.write(f"- {item}")
        else:
            st.info("No validation strategy was returned.")

    with risks_tab:
        if review.risks_and_limitations:
            for item in review.risks_and_limitations:
                st.warning(item)
        else:
            st.success(
                "No material risks or limitations were identified."
            )

    with questions_tab:
        if review.human_review_questions:
            for item in review.human_review_questions:
                st.write(f"- {item}")
        else:
            st.info("No human review questions were returned.")

    if review.requires_human_review:
        st.warning(
            "Human review is required before model evaluation can proceed."
        )


def render_modeling_hitl_controls(
    project,
) -> None:
    """Collect and apply the human Modeling decision."""

    review = project.modeling_review

    if review is None:
        st.error("No Modeling review is available.")
        return

    is_blocked = project.current_state == WorkflowState.BLOCKED
    existing_decision = project.modeling_decision

    if existing_decision is not None:
        if "modeling_reviewer" not in st.session_state:
            st.session_state["modeling_reviewer"] = (
                existing_decision.reviewer
            )

        if "modeling_review_rationale" not in st.session_state:
            st.session_state["modeling_review_rationale"] = (
                existing_decision.rationale
            )

        if "modeling_feedback" not in st.session_state:
            st.session_state["modeling_feedback"] = "\n".join(
                existing_decision.feedback
            )

    st.subheader("Modeling Human Review")

    if is_blocked:
        st.error(
            "The workflow is blocked by the human rejection. "
            "The review and decision history are preserved below."
        )

    eligible_models = list(dict.fromkeys(
        model.name
        for model in review.proposed_models
        if model.name in SUPPORTED_MODELS
    ))

    recommended_index = (
        eligible_models.index(review.recommended_model)
        if review.recommended_model in eligible_models
        else 0
    )

    available_features = (
        list(project.modeling.feature_columns)
        if project.modeling is not None
        else []
    )

    previously_selected = project.selected_feature_columns or []
    default_features = [
        column
        for column in previously_selected
        if column in available_features
    ]

    with st.form("modeling_human_review"):
        selected_features = st.multiselect(
            "Approved features for training *",
            options=available_features,
            default=default_features,
            help=(
                "Select only features approved for model training. "
                "Exclude identifiers, leakage-prone columns, and any "
                "features that should not be used."
            ),
            disabled=is_blocked or not available_features,
        )

        st.caption(
            f"{len(selected_features)} of "
            f"{len(available_features)} features selected."
        )

        selected_model = st.selectbox(
            "Select model for training *",
            options=eligible_models,
            index=recommended_index if eligible_models else None,
            placeholder="No supported models were proposed",
            disabled=is_blocked or not eligible_models,
        )

        reviewer = st.text_input(
            "Reviewer *",
            placeholder="Enter reviewer name or role.",
            key="modeling_reviewer",
            disabled=is_blocked,
        )

        rationale = st.text_area(
            "Reviewer rationale *",
            placeholder="Explain the reason for your decision.",
            key="modeling_review_rationale",
            disabled=is_blocked,
        )

        feedback_text = st.text_area(
            "Requested changes / feedback",
            placeholder="Required when requesting a revision.",
            key="modeling_feedback",
            disabled=is_blocked,
        )

        approve_column, revision_column, reject_column = st.columns(3)

        with approve_column:
            approve = st.form_submit_button(
                "Approve Modeling",
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

    if decision_value == "approve" and selected_model is None:
        st.warning(
            "Select a supported model before approving Modeling."
        )
        return

    if decision_value == "approve" and not selected_features:
        st.warning(
            "Select at least one approved feature before approving Modeling."
        )
        return

    if not reviewer.strip():
        st.warning(
            "Enter the reviewer before submitting a decision."
        )
        return

    if not rationale.strip():
        st.warning(
            "Enter a rationale before submitting a decision."
        )
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
        apply_modeling_decision(
            project,
            decision,
            selected_model=(
                selected_model
                if decision_value == "approve"
                else None
            ),
            selected_feature_columns=(
                selected_features
                if decision_value == "approve"
                else None
            ),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["project_state"] = project
    PROJECT_STORE.save(project)

    if project.current_state == WorkflowState.EVALUATION:
        st.success(
            "Modeling approved. The workflow may proceed to Evaluation."
        )
        return

    if project.current_state == WorkflowState.BLOCKED:
        st.error(
            "The workflow has been blocked by the human rejection."
        )
        return

    if project.current_state == WorkflowState.MODELING:
        st.warning(
            "Revision requested. Further Modeling work is required."
        )
        st.rerun()
