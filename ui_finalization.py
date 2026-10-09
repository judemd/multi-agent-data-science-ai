"""Streamlit review of the verified Finalization handoff package."""

import pandas as pd
import streamlit as st

from domain.finalization import FinalizationArtifact
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.project_store import ProjectStore
from workflow.finalization_workflow import apply_handoff_decision
from workflow.states import WorkflowState


def render_finalization_review(
    artifact: FinalizationArtifact,
    evidence_fingerprint: str,
) -> None:
    """Present handoff evidence without approving deployment."""

    st.subheader("Finalization — POC Handoff Review")

    st.warning(
        "Handoff approval completes this POC only. "
        "It does not authorize production deployment."
    )

    st.write(f"**Project:** {artifact.project_name}")
    st.write(f"**Revision:** {artifact.revision}")
    st.code(evidence_fingerprint, language=None)

    st.markdown("#### Business Framing")
    st.write(
        f"**Business problem:** "
        f"{artifact.problem_framing.business_problem}"
    )
    st.write(
        f"**Business objective:** "
        f"{artifact.problem_framing.business_objective}"
    )
    st.write(
        f"**Target outcome:** "
        f"{artifact.problem_framing.target_outcome}"
    )

    summary = artifact.executive_evaluation_summary

    if summary is not None:
        st.markdown("#### Executive Evaluation Summary")

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
            "**Recorded human Evaluation decision:** "
            f"{summary.evaluation_human_decision}"
        )

        if not summary.deployment_authorized:
            st.caption(
                "Production deployment is not authorized by this "
                "evaluation or POC handoff."
            )
    else:
        st.caption(
            "This historical handoff package does not contain "
            "an Executive Evaluation Summary."
        )

    st.markdown("#### Dataset Provenance")
    st.write(f"**Source:** {artifact.source_dataset_path}")
    st.write(f"**Prepared:** {artifact.prepared_dataset_path}")
    st.write(
        f"**Prepared dataset SHA-256:** "
        f"`{artifact.prepared_dataset_fingerprint}`"
    )

    st.markdown("#### Human-Approved Preparation")

    treatment_rows = [
        {
            "Issue": item.issue_id,
            "Treatment": item.treatment,
            "Rationale": item.rationale,
        }
        for item in artifact.treatment_plan.decisions
    ]

    if treatment_rows:
        st.dataframe(
            pd.DataFrame(treatment_rows),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No explicit treatment decisions were recorded.")

    st.write(
        "**Preparation reviewer:** "
        f"{artifact.preparation_decision.reviewer}"
    )
    st.write(
        "**Modeling reviewer:** "
        f"{artifact.modeling_decision.reviewer}"
    )
    st.write(
        "**Evaluation reviewer:** "
        f"{artifact.evaluation_decision.reviewer}"
    )

    evaluation = artifact.evaluation

    st.markdown("#### Model Evaluation")
    st.write(f"**Selected model:** {evaluation.selected_model}")
    st.write(f"**Target:** {evaluation.target_column}")
    st.write(f"**Training rows:** {evaluation.train_row_count:,}")
    st.write(f"**Test rows:** {evaluation.test_row_count:,}")

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
            "The selected threshold was evaluated on the held-out test set. "
            "Threshold adjustment can change the balance between "
            "detected churners and false-positive predictions."
        )
    else:
        st.caption(
            "Historical evaluation: no validation-selected threshold "
            "was recorded."
        )

    st.markdown("#### Unresolved Risks and Limitations")

    if artifact.unresolved_risks:
        for risk in artifact.unresolved_risks:
            st.warning(risk)
    else:
        st.info("No unresolved risks were recorded.")

    with st.expander("Inspect complete handoff evidence"):
        st.json(artifact.model_dump(mode="json"))

    st.caption(
        "This package requires a separate human handoff decision. "
        "No model deployment is performed."
    )
def render_finalization_hitl_controls(project: ProjectState) -> None:
    """Collect an explicit human decision on the POC handoff."""

    if project.current_state != WorkflowState.AWAITING_HANDOFF_APPROVAL:
        return

    if project.finalization is None:
        st.error("Finalization evidence is required before handoff review.")
        return

    if not project.finalization_evidence_fingerprint:
        st.error("The Finalization evidence fingerprint is missing.")
        return

    st.subheader("Finalization Human Decision")

    st.caption(
        "Approve completes the POC handoff, Request Revision returns "
        "to Finalization, and Reject blocks the workflow. "
        "None of these actions authorizes deployment."
    )

    with st.form("finalization_human_decision"):
        reviewer = st.text_input(
            "Reviewer *",
            key="finalization_reviewer",
        )

        rationale = st.text_area(
            "Decision rationale *",
            key="finalization_rationale",
        )

        feedback_text = st.text_area(
            "Specific revision requests (required for Request Revision)",
            key="finalization_feedback",
        )

        approve_column, revise_column, reject_column = st.columns(3)

        with approve_column:
            approve = st.form_submit_button(
                "Approve Handoff",
                type="primary",
            )

        with revise_column:
            revise = st.form_submit_button("Request Revision")

        with reject_column:
            reject = st.form_submit_button("Reject Handoff")

    if not any((approve, revise, reject)):
        return

    if not reviewer.strip():
        st.warning("Enter the reviewer before submitting a decision.")
        return

    if not rationale.strip():
        st.warning("Enter a rationale before submitting a decision.")
        return

    if revise:
        decision_value = "request_revision"
    elif reject:
        decision_value = "reject"
    else:
        decision_value = "approve"

    feedback = (
        [
            line.strip()
            for line in feedback_text.splitlines()
            if line.strip()
        ]
        if revise
        else []
    )

    if revise and not feedback:
        st.warning(
            "Request Revision requires at least one specific change."
        )
        return

    decision = HITLDecision(
        decision=decision_value,
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        feedback=feedback,
    )

    try:
        apply_handoff_decision(project, decision)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["project_state"] = project
    ProjectStore().save(project)

    if project.current_state == WorkflowState.COMPLETE:
        st.success(
            "POC handoff approved and completed. "
            "Production deployment is not authorized."
        )
    elif project.current_state == WorkflowState.FINALIZATION:
        st.info(
            "Handoff revision requested. Returning to Finalization."
        )
    else:
        st.warning(
            "Handoff rejected. The workflow is blocked."
        )

    st.rerun()