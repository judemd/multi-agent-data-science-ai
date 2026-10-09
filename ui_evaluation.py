"""Streamlit presentation of deterministic model evaluation evidence."""

import pandas as pd
import streamlit as st

from domain.evaluation import EvaluationArtifact
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.executive_evaluation_summary import (
    build_executive_evaluation_summary,
)
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

    metrics_columns = {
        "Metric": metric_names,
        "Model (default threshold)": [
            getattr(evaluation.model_metrics, name)
            for name in metric_names
        ],
    }

    if evaluation.threshold_metrics is not None:
        metrics_columns["Model (selected threshold)"] = [
            getattr(evaluation.threshold_metrics, name)
            for name in metric_names
        ]

    metrics_columns["Baseline"] = [
        getattr(evaluation.baseline_metrics, name)
        for name in metric_names
    ]

    metrics_table = pd.DataFrame(metrics_columns)
    numeric_formats = {
        column: "{:.3f}"
        for column in metrics_table.columns
        if column != "Metric"
    }

    st.dataframe(
        metrics_table.style.format(numeric_formats),
        hide_index=True,
        use_container_width=True,
    )

    if (
        evaluation.selected_threshold is not None
        and evaluation.validation_f1 is not None
        and evaluation.threshold_metrics is not None
    ):
        st.write(
            "**Validation-selected classification threshold:** "
            f"{evaluation.selected_threshold:.3f}"
        )
        st.caption(
            f"Training-only validation F1: {evaluation.validation_f1:.3f}. "
            "The threshold was selected before evaluating the held-out "
            "test set. Lower thresholds can identify more potential "
            "churners but may also increase false-positive predictions."
        )
    else:
        st.caption(
            "Historical evaluation: no validation-selected threshold "
            "was recorded."
        )

    summary = build_executive_evaluation_summary(evaluation)

    st.markdown("#### Executive Evaluation Summary ? Decision Pending")
    st.write(f"**Executive assessment:** {summary.assessment}")
    st.write(f"**Measured performance:** {summary.performance_summary}")

    st.markdown("**Business implications**")
    for implication in summary.business_implications:
        st.write(f"- {implication}")

    st.markdown("**Recommended actions**")
    for action in summary.recommended_actions:
        st.write(f"- {action}")

    st.markdown("**Risks and limitations**")
    if summary.risks_and_limitations:
        for limitation in summary.risks_and_limitations:
            st.write(f"- {limitation}")
    else:
        st.write("No evaluation limitations were recorded.")

    st.write(
        "**Human Evaluation decision:** "
        f"{summary.evaluation_human_decision}"
    )
    st.caption(
        "This is a read-only preview. No human decision has been "
        "recorded by this summary, and deployment is not authorized."
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