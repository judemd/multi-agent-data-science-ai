"""Streamlit presentation of deterministic model evaluation evidence."""

import pandas as pd
import streamlit as st

from domain.evaluation import EvaluationArtifact
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.project_store import ProjectStore
from workflow.evaluation_workflow import apply_evaluation_decision
from workflow.states import WorkflowState


def render_evaluation_review(evaluation: EvaluationArtifact) -> None:
    """Display model results without making a deployment decision."""

    st.subheader("Evaluation Review")

    st.caption(
        "These results describe a single stratified holdout evaluation. "
        "Human review is required before the workflow can proceed."
    )

    st.write(f"**Selected model:** {evaluation.selected_model}")
    st.write(f"**Target:** {evaluation.target_column}")
    st.write(f"**Training rows:** {evaluation.train_row_count:,}")
    st.write(f"**Test rows:** {evaluation.test_row_count:,}")

    st.markdown("#### Model vs. Baseline")

    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    )

    metrics_table = pd.DataFrame(
        {
            "Metric": metric_names,
            "Selected model": [
                getattr(evaluation.model_metrics, name)
                for name in metric_names
            ],
            "Baseline": [
                getattr(evaluation.baseline_metrics, name)
                for name in metric_names
            ],
        }
    )

    st.dataframe(
        metrics_table.style.format(
            {
                "Selected model": "{:.3f}",
                "Baseline": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    features_tab, exclusions_tab, limitations_tab = st.tabs(
        [
            "Approved Features",
            "Excluded Columns",
            "Limitations",
        ]
    )

    with features_tab:
        st.write(
            f"**{len(evaluation.feature_columns)} approved features**"
        )
        for column in evaluation.feature_columns:
            st.write(f"- {column}")

    with exclusions_tab:
        if evaluation.excluded_columns:
            for column in evaluation.excluded_columns:
                st.write(f"- {column}")
        else:
            st.info("No feature columns were excluded.")

    with limitations_tab:
        if evaluation.limitations:
            for limitation in evaluation.limitations:
                st.warning(limitation)
        else:
            st.info("No limitations were recorded.")

    st.info(
        "Evaluation evidence is ready for human GO / ITERATE / NO-GO review. "
        "No deployment or finalization decision has been made."
    )


def render_evaluation_hitl_controls(project: ProjectState) -> None:
    """Collect a human GO, ITERATE, or NO-GO decision."""

    if project.current_state != WorkflowState.AWAITING_GO_NO_GO:
        return

    if project.evaluation is None:
        st.error("Evaluation evidence is required before a decision.")
        return

    st.subheader("Evaluation Human Decision")
    st.caption(
        "GO proceeds to Finalization, ITERATE returns to Modeling, "
        "and NO-GO blocks the workflow. GO does not deploy a model."
    )

    with st.form("evaluation_human_decision"):
        reviewer = st.text_input(
            "Reviewer *",
            key="evaluation_reviewer",
        )

        rationale = st.text_area(
            "Decision rationale *",
            key="evaluation_rationale",
        )

        feedback_text = st.text_area(
            "Specific revision requests (required for ITERATE)",
            key="evaluation_feedback",
        )

        go_column, iterate_column, no_go_column = st.columns(3)

        with go_column:
            go = st.form_submit_button(
                "GO",
                type="primary",
            )

        with iterate_column:
            iterate = st.form_submit_button("ITERATE")

        with no_go_column:
            no_go = st.form_submit_button("NO-GO")

    if not any((go, iterate, no_go)):
        return

    if not reviewer.strip():
        st.warning("Enter the reviewer before submitting a decision.")
        return

    if not rationale.strip():
        st.warning("Enter a rationale before submitting a decision.")
        return

    if iterate:
        decision_value = "request_revision"
    elif no_go:
        decision_value = "reject"
    else:
        decision_value = "approve"

    feedback = [
        line.strip()
        for line in feedback_text.splitlines()
        if line.strip()
    ] if iterate else []

    if iterate and not feedback:
        st.warning("ITERATE requires at least one specific revision request.")
        return

    decision = HITLDecision(
        decision=decision_value,
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        feedback=feedback,
    )

    try:
        apply_evaluation_decision(project, decision)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["project_state"] = project
    ProjectStore().save(project)

    if project.current_state == WorkflowState.FINALIZATION:
        st.success("GO approved. Proceeding to Finalization.")
    elif project.current_state == WorkflowState.MODELING:
        st.info("ITERATE requested. Returning to Modeling.")
    else:
        st.warning("NO-GO recorded. The workflow is blocked.")

    st.rerun()