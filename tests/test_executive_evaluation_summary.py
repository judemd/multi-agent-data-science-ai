"""Tests for deterministic executive evaluation summaries."""

import pytest

from domain.evaluation import EvaluationArtifact, EvaluationMetrics
from domain.hitl_decision import HITLDecision
from tools.executive_evaluation_summary import (
    build_executive_evaluation_summary,
)


def make_metrics(accuracy, precision, recall, f1, roc_auc):
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
    )


def make_evaluation(with_threshold=False):
    fields = {}

    if with_threshold:
        fields = {
            "selected_threshold": 0.275,
            "validation_f1": 0.520,
            "threshold_metrics": make_metrics(
                0.710, 0.390, 0.720, 0.506, 0.711
            ),
        }

    return EvaluationArtifact(
        selected_model="Random Forest",
        target_column="churned",
        feature_columns=["age", "tenure_months"],
        train_row_count=24052,
        test_row_count=6013,
        model_metrics=make_metrics(
            0.802, 0.615, 0.081, 0.143, 0.711
        ),
        baseline_metrics=make_metrics(
            0.796, 0.0, 0.0, 0.0, 0.5
        ),
        limitations=["Single holdout split."],
        **fields,
    )


def make_decision(value="approve"):
    return HITLDecision(
        decision=value,
        reviewer="Human Reviewer",
        rationale="Reviewed the measured evidence.",
    )


def test_historical_evaluation_without_threshold():
    summary = build_executive_evaluation_summary(
        make_evaluation(), make_decision()
    )

    assert "0.802" in summary.performance_summary
    assert "0.081" in summary.performance_summary
    assert "0.796" in summary.performance_summary
    assert "0.275" not in summary.performance_summary
    assert summary.risks_and_limitations == ["Single holdout split."]
    assert summary.deployment_authorized is False


def test_threshold_comparison_uses_recorded_evidence():
    summary = build_executive_evaluation_summary(
        make_evaluation(with_threshold=True),
        make_decision(),
    )

    assert "0.275" in summary.performance_summary
    assert "0.520" in summary.performance_summary
    assert "0.720" in summary.performance_summary
    assert "0.390" in summary.performance_summary
    assert "training-only validation" in summary.performance_summary
    assert "did not use the held-out test labels" in (
        summary.performance_summary
    )
    assert summary.deployment_authorized is False


@pytest.mark.parametrize(
    ("decision", "label"),
    [
        ("approve", "GO"),
        ("request_revision", "ITERATE"),
        ("reject", "NO-GO"),
    ],
)
def test_human_decision_mapping(decision, label):
    summary = build_executive_evaluation_summary(
        make_evaluation(), make_decision(decision)
    )

    assert f"Human Evaluation decision: {label}" in summary.assessment
    assert f"{label} ({decision})" in summary.evaluation_human_decision
    assert "Human Reviewer" in summary.evaluation_human_decision
    assert "Reviewed the measured evidence." in (
        summary.evaluation_human_decision
    )
    assert summary.deployment_authorized is False


def test_pending_review_preview_does_not_invent_approval():
    evaluation = make_evaluation(with_threshold=True)
    original = evaluation.model_dump()

    summary = build_executive_evaluation_summary(evaluation)

    assert "Human Evaluation decision: PENDING" in summary.assessment
    assert summary.evaluation_human_decision == (
        "PENDING: Human GO / ITERATE / NO-GO review is required."
    )
    assert "Human Reviewer" not in summary.evaluation_human_decision
    assert "0.275" in summary.performance_summary
    assert summary.deployment_authorized is False
    assert evaluation.model_dump() == original


def test_summary_does_not_mutate_source_evidence():
    evaluation = make_evaluation(with_threshold=True)
    decision = make_decision()
    original_evaluation = evaluation.model_dump()
    original_decision = decision.model_dump()

    build_executive_evaluation_summary(evaluation, decision)

    assert evaluation.model_dump() == original_evaluation
    assert decision.model_dump() == original_decision
