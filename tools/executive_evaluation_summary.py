"""Deterministic executive summary from recorded evaluation evidence."""

from domain.evaluation import EvaluationArtifact
from domain.executive_evaluation_summary import ExecutiveEvaluationSummary
from domain.hitl_decision import HITLDecision


DECISION_LABELS = {
    "approve": "GO",
    "request_revision": "ITERATE",
    "reject": "NO-GO",
}


def build_executive_evaluation_summary(
    evaluation: EvaluationArtifact,
    decision: HITLDecision | None = None,
) -> ExecutiveEvaluationSummary:
    """Summarize recorded evidence without claiming deployment readiness."""

    label = (
        DECISION_LABELS[decision.decision]
        if decision is not None
        else None
    )
    model = evaluation.model_metrics
    baseline = evaluation.baseline_metrics

    performance = (
        f"{evaluation.selected_model} was evaluated on "
        f"{evaluation.test_row_count:,} held-out rows. "
        f"At the default classification threshold, accuracy was "
        f"{model.accuracy:.3f}, precision {model.precision:.3f}, "
        f"recall {model.recall:.3f}, F1 {model.f1:.3f}, "
        f"and ROC-AUC {model.roc_auc:.3f}. "
        f"The baseline achieved accuracy {baseline.accuracy:.3f}, "
        f"precision {baseline.precision:.3f}, "
        f"recall {baseline.recall:.3f}, F1 {baseline.f1:.3f}, "
        f"and ROC-AUC {baseline.roc_auc:.3f}."
    )

    if (
        evaluation.selected_threshold is not None
        and evaluation.validation_f1 is not None
        and evaluation.threshold_metrics is not None
    ):
        selected = evaluation.threshold_metrics
        performance += (
            f" A classification threshold of "
            f"{evaluation.selected_threshold:.3f} was selected using "
            f"training-only validation F1 "
            f"({evaluation.validation_f1:.3f}). "
            f"At that threshold, held-out precision was "
            f"{selected.precision:.3f}, recall {selected.recall:.3f}, "
            f"F1 {selected.f1:.3f}, and accuracy "
            f"{selected.accuracy:.3f}. "
            f"Threshold selection did not use the held-out test labels."
        )

    implications = [
        (
            f"At the default threshold, recall of "
            f"{model.recall:.1%} means the model identified "
            f"approximately that share of actual positive cases "
            f"in the held-out test set."
        ),
        (
            f"At the default threshold, precision of "
            f"{model.precision:.1%} means approximately that share "
            f"of positive predictions were correct on the test set."
        ),
        (
            "For a churn-retention use case, higher recall can identify "
            "more potential churners, while lower precision can lead "
            "to more unnecessary retention outreach. Actual campaign "
            "costs and benefits have not been measured."
        ),
    ]

    if evaluation.threshold_metrics is not None:
        selected = evaluation.threshold_metrics
        implications.append(
            "At the validation-selected threshold, held-out recall "
            f"was {selected.recall:.1%} and precision was "
            f"{selected.precision:.1%}; compare these with the "
            "default-threshold results before choosing an operating "
            "point."
        )

    actions = [
        "Define acceptable recall, precision, outreach capacity, "
        "and intervention costs with business stakeholders.",
        "Verify that every selected feature is available at the "
        "intended prediction time and assess potential leakage.",
        "Validate performance across additional splits or "
        "cross-validation and relevant customer segments.",
        "Require a separate human deployment-readiness review; "
        "the POC handoff does not authorize deployment.",
    ]

    assessment = (
        f"Human Evaluation decision: {label}. "
        if label is not None
        else "Human Evaluation decision: PENDING. "
    )
    assessment += (
        "This is a POC evidence and handoff assessment, "
        "not authorization to deploy a model."
    )

    decision_description = (
        f"{label} ({decision.decision}) by {decision.reviewer}; "
        f"rationale: {decision.rationale}"
        if decision is not None
        else "PENDING: Human GO / ITERATE / NO-GO review is required."
    )

    return ExecutiveEvaluationSummary(
        assessment=assessment,
        performance_summary=performance,
        business_implications=implications,
        recommended_actions=actions,
        risks_and_limitations=list(evaluation.limitations),
        evaluation_human_decision=decision_description,
        deployment_authorized=False,
    )
