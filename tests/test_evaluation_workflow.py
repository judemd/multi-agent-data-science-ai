"""Tests for human-controlled Evaluation transitions."""

import pytest

from domain.evaluation import EvaluationArtifact, EvaluationMetrics
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from workflow.evaluation_workflow import apply_evaluation_decision
from workflow.states import WorkflowState


def build_project() -> ProjectState:
    metrics = EvaluationMetrics(
        accuracy=0.8,
        precision=0.7,
        recall=0.6,
        f1=0.65,
        roc_auc=0.75,
    )

    return ProjectState(
        project_id="evaluation-decision-test",
        project_name="Evaluation Decision Test",
        current_state=WorkflowState.AWAITING_GO_NO_GO,
        selected_model="Logistic Regression",
        selected_feature_columns=["age", "region"],
        evaluation=EvaluationArtifact(
            selected_model="Logistic Regression",
            target_column="churn",
            feature_columns=["age", "region"],
            excluded_columns=["customer_id"],
            train_row_count=80,
            test_row_count=20,
            model_metrics=metrics,
            baseline_metrics=metrics,
        ),
    )


def make_decision(
    decision: str,
    *,
    reviewer: str = "Human Reviewer",
    rationale: str = "Reviewed evaluation evidence.",
    feedback: list[str] | None = None,
) -> HITLDecision:
    return HITLDecision(
        decision=decision,
        reviewer=reviewer,
        rationale=rationale,
        feedback=feedback or [],
    )


def test_go_advances_to_finalization_and_preserves_evidence():
    project = build_project()
    original_evaluation = project.evaluation.model_copy(deep=True)

    result = apply_evaluation_decision(
        project,
        make_decision("approve"),
    )

    assert result is project
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.evaluation == original_evaluation
    assert project.evaluation_decision.decision == "approve"
    assert project.evaluation_history == [original_evaluation]
    assert len(project.evaluation_decision_history) == 1


def test_iterate_returns_to_modeling_and_preserves_history():
    project = build_project()
    original_evaluation = project.evaluation.model_copy(deep=True)

    result = apply_evaluation_decision(
        project,
        make_decision(
            "request_revision",
            feedback=["Improve positive-class recall."],
        ),
    )

    assert result is project
    assert project.current_state == WorkflowState.MODELING
    assert project.revision == 2
    assert project.evaluation is None
    assert project.evaluation_history == [original_evaluation]
    assert project.evaluation_decision.decision == "request_revision"
    assert project.evaluation_decision_history[0].feedback == [
        "Improve positive-class recall."
    ]
    assert project.modeling is None
    assert project.modeling_review is None
    assert project.modeling_decision is None
    assert project.selected_model is None
    assert project.selected_feature_columns is None


def test_no_go_blocks_and_preserves_evidence():
    project = build_project()

    apply_evaluation_decision(
        project,
        make_decision("reject"),
    )

    assert project.current_state == WorkflowState.BLOCKED
    assert project.evaluation is not None
    assert project.evaluation_decision.decision == "reject"
    assert len(project.evaluation_history) == 1
    assert len(project.evaluation_decision_history) == 1


@pytest.mark.parametrize(
    "state",
    [
        WorkflowState.MODELING,
        WorkflowState.EVALUATION,
        WorkflowState.FINALIZATION,
        WorkflowState.BLOCKED,
    ],
)
def test_rejects_incorrect_state_without_mutation(state):
    project = build_project()
    project.current_state = state
    original = project.model_dump()

    with pytest.raises(ValueError, match="awaiting GO / NO-GO review"):
        apply_evaluation_decision(
            project,
            make_decision("approve"),
        )

    assert project.model_dump() == original


def test_requires_evaluation_evidence_without_mutation():
    project = build_project()
    project.evaluation = None
    original = project.model_dump()

    with pytest.raises(ValueError, match="Evaluation evidence is required"):
        apply_evaluation_decision(
            project,
            make_decision("approve"),
        )

    assert project.model_dump() == original


@pytest.mark.parametrize(
    ("reviewer", "rationale", "message"),
    [
        ("   ", "Valid rationale", "reviewer cannot be blank"),
        ("Human Reviewer", "   ", "rationale cannot be blank"),
    ],
)
def test_rejects_blank_review_fields_without_mutation(
    reviewer,
    rationale,
    message,
):
    project = build_project()
    original = project.model_dump()

    with pytest.raises(ValueError, match=message):
        apply_evaluation_decision(
            project,
            make_decision(
                "approve",
                reviewer=reviewer,
                rationale=rationale,
            ),
        )

    assert project.model_dump() == original


@pytest.mark.parametrize(
    "feedback",
    [
        [],
        ["   "],
    ],
)
def test_iterate_requires_specific_feedback_without_mutation(feedback):
    project = build_project()
    original = project.model_dump()

    with pytest.raises(ValueError, match="ITERATE requires"):
        apply_evaluation_decision(
            project,
            make_decision(
                "request_revision",
                feedback=feedback,
            ),
        )

    assert project.model_dump() == original


def test_evaluation_decision_history_survives_json_roundtrip():
    project = build_project()

    apply_evaluation_decision(
        project,
        make_decision(
            "request_revision",
            feedback=["Review feature selection."],
        ),
    )

    restored = ProjectState.model_validate_json(
        project.model_dump_json()
    )

    assert restored.current_state == WorkflowState.MODELING
    assert restored.evaluation is None
    assert len(restored.evaluation_history) == 1
    assert len(restored.evaluation_decision_history) == 1
    assert (
        restored.evaluation_decision_history[0].decision
        == "request_revision"
    )
def test_historical_evaluation_artifact_remains_valid():
    artifact = build_project().evaluation

    assert artifact.selected_threshold is None
    assert artifact.validation_f1 is None
    assert artifact.threshold_metrics is None

    restored = EvaluationArtifact.model_validate(
        artifact.model_dump(exclude_none=True)
    )
    assert restored == artifact


def test_threshold_evaluation_artifact_round_trips():
    artifact = build_project().evaluation.model_copy(
        update={
            "selected_threshold": 0.35,
            "validation_f1": 0.72,
            "threshold_metrics": EvaluationMetrics(
                accuracy=0.76,
                precision=0.55,
                recall=0.68,
                f1=0.61,
                roc_auc=0.75,
            ),
        }
    )

    restored = EvaluationArtifact.model_validate(
        artifact.model_dump(mode="json")
    )

    assert restored == artifact
    assert restored.selected_threshold == pytest.approx(0.35)
    assert restored.validation_f1 == pytest.approx(0.72)
    assert restored.threshold_metrics.recall == pytest.approx(0.68)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_evaluation_artifact_rejects_invalid_threshold(threshold):
    payload = build_project().evaluation.model_dump()
    payload["selected_threshold"] = threshold

    with pytest.raises(ValueError):
        EvaluationArtifact.model_validate(payload)
