"""Tests for deterministic Finalization handoff assembly."""

import pytest

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_issue import DataPreparationIssue
from domain.data_preparation_treatment_decision import (
    DataPreparationTreatmentDecision,
)
from domain.data_preparation_treatment_plan import (
    DataPreparationTreatmentPlan,
)
from domain.evaluation import EvaluationArtifact, EvaluationMetrics
from domain.hitl_decision import HITLDecision
from domain.modeling import ModelingArtifact
from domain.problem_framing import ProblemFramingArtifact
from domain.project_state import ProjectState
from tools.dataset_fingerprint import fingerprint_dataset_file
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)
from workflow.finalization_stage import run_finalization_stage
from workflow.states import WorkflowState


def approved_decision():
    return HITLDecision(
        decision="approve",
        reviewer="Human Reviewer",
        rationale="Reviewed and approved the evidence.",
    )


def build_project(tmp_path):
    source_path = tmp_path / "source.csv"
    source_path.write_text(
        "age,region,churn\n20,north,0\n30,south,1\n",
        encoding="utf-8",
    )

    prepared_path = tmp_path / "prepared.csv"
    prepared_path.write_text(
        "age,region,churn\n20,north,0\n30,south,1\n",
        encoding="utf-8",
    )

    issue = DataPreparationIssue(
        issue_id="missing_values:region",
        issue_type="missing_values",
        column="region",
        evidence="Some source records have missing regions.",
        allowed_treatments=["retain", "mode"],
        requires_explicit_human_decision=True,
    )

    preparation = DataPreparationArtifact(
        file_name="source.csv",
        row_count=2,
        column_count=3,
        issues=[issue],
    )

    source_fingerprint = fingerprint_dataset_file(source_path)
    evidence_fingerprint = fingerprint_preparation_evidence(preparation)

    treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=source_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="Human Reviewer",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id=issue.issue_id,
                treatment="retain",
                rationale="Retained for further investigation.",
            )
        ],
    )

    metrics = EvaluationMetrics(
        accuracy=0.8,
        precision=0.7,
        recall=0.6,
        f1=0.65,
        roc_auc=0.75,
    )

    return ProjectState(
        project_id="finalization-test",
        project_name="Finalization Test",
        current_state=WorkflowState.FINALIZATION,
        dataset_path=str(source_path),
        prepared_dataset_path=str(prepared_path),
        prepared_dataset_fingerprint=fingerprint_dataset_file(
            prepared_path
        ),
        data_preparation_dataset_fingerprint=source_fingerprint,
        data_preparation_evidence_fingerprint=evidence_fingerprint,
        problem_framing=ProblemFramingArtifact(
            business_problem="Customer churn",
            business_objective="Understand churn risk",
            target_outcome="Support retention decisions",
        ),
        data_preparation=preparation,
        data_preparation_treatment_plan=treatment_plan,
        data_preparation_decision=approved_decision(),
        modeling=ModelingArtifact(
            file_name="prepared.csv",
            row_count=2,
            column_count=3,
            target_column="churn",
            target_dtype="int64",
            target_missing_count=0,
            target_unique_count=2,
            target_distribution={"0": 1, "1": 1},
            feature_columns=["age", "region"],
        ),
        modeling_decision=approved_decision(),
        selected_model="Logistic Regression",
        selected_feature_columns=["age", "region"],
        evaluation=EvaluationArtifact(
            selected_model="Logistic Regression",
            target_column="churn",
            feature_columns=["age", "region"],
            train_row_count=80,
            test_row_count=20,
            model_metrics=metrics,
            baseline_metrics=metrics,
            limitations=["Single holdout split."],
        ),
        evaluation_decision=approved_decision(),
    )


def test_finalization_creates_handoff_pending_human_approval(tmp_path):
    project = build_project(tmp_path)

    result = run_finalization_stage(project)

    assert result is project
    assert project.current_state == WorkflowState.AWAITING_HANDOFF_APPROVAL
    assert project.finalization is not None
    assert project.handoff_decision is None
    assert project.finalization.deployment_approved is False
    assert project.finalization.requires_human_handoff_approval is True
    assert project.finalization.evaluation.selected_model == (
        "Logistic Regression"
    )
    assert any(
        "Retained or pending issue" in risk
        for risk in project.finalization.unresolved_risks
    )
    assert any(
        "Problem Framing" in risk
        for risk in project.finalization.unresolved_risks
    )
    assert any(
        "Single holdout split." in risk
        for risk in project.finalization.unresolved_risks
    )

    restored = ProjectState.model_validate_json(project.model_dump_json())
    assert restored.finalization == project.finalization


def test_finalization_rejects_wrong_state_without_mutation(tmp_path):
    project = build_project(tmp_path)
    project.current_state = WorkflowState.EVALUATION
    original = project.model_dump()

    with pytest.raises(ValueError, match="only run"):
        run_finalization_stage(project)

    assert project.model_dump() == original


def test_finalization_requires_problem_framing_without_mutation(tmp_path):
    project = build_project(tmp_path)
    project.problem_framing = None
    original = project.model_dump()

    with pytest.raises(ValueError, match="Problem Framing"):
        run_finalization_stage(project)

    assert project.model_dump() == original


def test_finalization_requires_evaluation_go_without_mutation(tmp_path):
    project = build_project(tmp_path)
    project.evaluation_decision = HITLDecision(
        decision="reject",
        reviewer="Human Reviewer",
        rationale="Performance not sufficient.",
    )
    original = project.model_dump()

    with pytest.raises(ValueError, match="approved Evaluation"):
        run_finalization_stage(project)

    assert project.model_dump() == original


def test_finalization_rejects_changed_prepared_dataset(tmp_path):
    project = build_project(tmp_path)
    original = project.model_dump()

    with open(project.prepared_dataset_path, "a", encoding="utf-8") as file:
        file.write("40,east,0\n")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_finalization_stage(project)

    assert project.model_dump() == original


def test_finalization_rejects_model_selection_mismatch(tmp_path):
    project = build_project(tmp_path)
    project.selected_model = "Random Forest"
    original = project.model_dump()

    with pytest.raises(ValueError, match="model selection"):
        run_finalization_stage(project)

    assert project.model_dump() == original
def test_finalization_rejects_modified_preparation_evidence(tmp_path):
    project = build_project(tmp_path)
    project.data_preparation.issues[0].evidence = (
        "Evidence changed after approval."
    )
    original = project.model_dump()

    with pytest.raises(ValueError, match="preparation evidence fingerprint mismatch"):
        run_finalization_stage(project)

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.finalization is None


def test_finalization_rejects_unknown_treatment_issue(tmp_path):
    project = build_project(tmp_path)
    project.data_preparation_treatment_plan.decisions[0].issue_id = (
        "unknown:region"
    )
    original = project.model_dump()

    with pytest.raises(ValueError, match="unknown|Unknown|not found"):
        run_finalization_stage(project)

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.finalization is None


def test_finalization_rejects_disallowed_treatment(tmp_path):
    project = build_project(tmp_path)
    project.data_preparation_treatment_plan.decisions[0].treatment = (
        "convert_date"
    )
    original = project.model_dump()

    with pytest.raises(ValueError, match="treatment|Treatment|permitted|allowed"):
        run_finalization_stage(project)

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.finalization is None
def test_finalization_rejects_changed_source_dataset(tmp_path):
    project = build_project(tmp_path)
    original = project.model_dump()

    with open(project.dataset_path, "a", encoding="utf-8") as file:
        file.write("40,east,0\n")

    with pytest.raises(ValueError, match="source dataset fingerprint mismatch"):
        run_finalization_stage(project)

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.finalization is None
def test_finalization_rejects_missing_treatment_decision(tmp_path):
    project = build_project(tmp_path)

    # The detected issue requires an explicit human treatment decision.
    project.data_preparation_treatment_plan.decisions = []

    original = project.model_dump()

    with pytest.raises(ValueError):
        run_finalization_stage(project)

    assert project.model_dump() == original
    assert project.current_state == WorkflowState.FINALIZATION
    assert project.finalization is None