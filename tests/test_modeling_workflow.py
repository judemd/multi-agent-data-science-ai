import pytest

from domain.hitl_decision import HITLDecision
from domain.modeling import ModelingArtifact
from domain.modeling_review import ModelingReview
from domain.project_state import ProjectState
from workflow.modeling_workflow import apply_modeling_decision
from workflow.states import WorkflowState


def build_project() -> ProjectState:
    return ProjectState(
        project_id="project-005",
        project_name="Customer Churn",
        current_state=WorkflowState.AWAITING_MODEL_SELECTION,
        modeling=ModelingArtifact(
            file_name="customers.csv",
            row_count=100,
            column_count=3,
            target_column="churn",
            target_dtype="int64",
            target_missing_count=0,
            target_unique_count=2,
            target_distribution={"0": 80, "1": 20},
            feature_columns=["age", "region"],
        ),
        modeling_review=ModelingReview(
            observed_evidence=[
                "The target contains two distinct classes.",
            ],
            proposed_models=[
                {
                    "name": "Logistic Regression",
                    "model_family": "linear classification",
                    "rationale": "Suitable baseline for a binary target.",
                    "considerations": [
                        "Requires appropriate numeric encoding.",
                    ],
                }
            ],
            recommended_model="Logistic Regression",
            rationale=[
                "The target structure supports a binary classification approach."
            ],
            validation_strategy=[
                "Use a stratified validation strategy appropriate for the target."
            ],
            requires_human_review=False,
        ),
    )


def build_decision(decision: str) -> HITLDecision:
    return HITLDecision(
        decision=decision,
        reviewer="data_scientist",
        rationale="Decision based on the modeling review.",
    )


def test_approval_moves_project_to_evaluation():
    project = apply_modeling_decision(
        build_project(),
        build_decision("approve"),
        selected_model="Logistic Regression",
        selected_feature_columns=["age", "region"],
    )

    assert project.current_state == WorkflowState.EVALUATION


def test_revision_returns_project_to_modeling():
    project = apply_modeling_decision(
        build_project(),
        build_decision("request_revision"),
    )

    assert project.current_state == WorkflowState.MODELING


def test_rejection_moves_project_to_blocked():
    project = apply_modeling_decision(
        build_project(),
        build_decision("reject"),
    )

    assert project.current_state == WorkflowState.BLOCKED


def test_decision_requires_existing_review():
    project = build_project()
    project.modeling_review = None

    with pytest.raises(
        ValueError,
        match="without a review",
    ):
        apply_modeling_decision(
            project,
            build_decision("approve"),
        )


def test_decision_requires_model_selection_state():
    project = build_project()
    project.current_state = WorkflowState.MODELING

    with pytest.raises(
        ValueError,
        match="awaiting model selection",
    ):
        apply_modeling_decision(
            project,
            build_decision("approve"),
        )


def test_review_requiring_human_review_can_progress_after_approval():
    project = build_project()
    project.modeling_review.requires_human_review = True

    project = apply_modeling_decision(
        project,
        build_decision("approve"),
        selected_model="Logistic Regression",
        selected_feature_columns=["age", "region"],
    )

    assert project.current_state == WorkflowState.EVALUATION


def test_revision_clears_previous_modeling_review():
    project = build_project()

    project.modeling = ModelingArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
        target_column="churn",
        target_dtype="int64",
        target_missing_count=0,
        target_unique_count=2,
        target_distribution={
            "0": 80,
            "1": 20,
        },
    )

    project.modeling_review = ModelingReview(
        observed_evidence=["Previous evidence"],
        proposed_models=[],
        recommended_model="Previous Model",
        rationale=["Previous rationale"],
        validation_strategy=["Previous validation strategy"],
        requires_human_review=True,
    )

    updated = apply_modeling_decision(
        project,
        build_decision("request_revision"),
    )

    assert updated.current_state == WorkflowState.MODELING
    assert updated.modeling is None
    assert updated.modeling_review is None
    assert updated.modeling_decision is not None
    assert updated.modeling_decision.decision == "request_revision"


def test_approval_persists_selected_model():
    project = apply_modeling_decision(
        build_project(),
        build_decision("approve"),
        selected_model="Logistic Regression",
        selected_feature_columns=["age", "region"],
    )

    assert project.current_state == WorkflowState.EVALUATION
    assert project.selected_model == "Logistic Regression"


def test_approval_requires_explicit_model_selection():
    project = build_project()

    with pytest.raises(ValueError, match="explicit model selection"):
        apply_modeling_decision(
            project,
            build_decision("approve"),
        )

    assert project.current_state == WorkflowState.AWAITING_MODEL_SELECTION
    assert project.selected_model is None


def test_approval_rejects_unsupported_model():
    project = build_project()

    with pytest.raises(ValueError, match="Unsupported model selection"):
        apply_modeling_decision(
            project,
            build_decision("approve"),
            selected_model="Unsupported Model",
        )

    assert project.current_state == WorkflowState.AWAITING_MODEL_SELECTION
    assert project.selected_model is None


def test_approval_rejects_model_not_proposed_by_agent():
    project = build_project()

    with pytest.raises(ValueError, match="not proposed"):
        apply_modeling_decision(
            project,
            build_decision("approve"),
            selected_model="Random Forest",
        )

    assert project.current_state == WorkflowState.AWAITING_MODEL_SELECTION
    assert project.selected_model is None


def test_revision_clears_selected_model():
    project = build_project()
    project.selected_model = "Logistic Regression"

    updated = apply_modeling_decision(
        project,
        build_decision("request_revision"),
    )

    assert updated.current_state == WorkflowState.MODELING
    assert updated.selected_model is None


def test_approval_persists_selected_features():
    project = apply_modeling_decision(
        build_project(),
        build_decision("approve"),
        selected_model="Logistic Regression",
        selected_feature_columns=["age"],
    )

    assert project.current_state == WorkflowState.EVALUATION
    assert project.selected_feature_columns == ["age"]


@pytest.mark.parametrize(
    "features, error",
    [
        (None, "Explicit feature selection"),
        ([], "Explicit feature selection"),
        (["age", "age"], "contain duplicates"),
        (["unknown"], "Unknown selected feature"),
        (["churn"], "Target column cannot"),
    ],
)
def test_approval_rejects_invalid_feature_selection(features, error):
    project = build_project()

    with pytest.raises(ValueError, match=error):
        apply_modeling_decision(
            project,
            build_decision("approve"),
            selected_model="Logistic Regression",
            selected_feature_columns=features,
        )

    assert project.current_state == WorkflowState.AWAITING_MODEL_SELECTION
    assert project.modeling_decision is None
    assert project.selected_model is None
    assert project.selected_feature_columns is None


def test_approval_requires_modeling_evidence():
    project = build_project()
    project.modeling = None

    with pytest.raises(ValueError, match="Modeling evidence is required"):
        apply_modeling_decision(
            project,
            build_decision("approve"),
            selected_model="Logistic Regression",
            selected_feature_columns=["age"],
        )

    assert project.current_state == WorkflowState.AWAITING_MODEL_SELECTION


def test_revision_clears_selected_features():
    project = build_project()
    project.selected_model = "Logistic Regression"
    project.selected_feature_columns = ["age", "region"]

    updated = apply_modeling_decision(
        project,
        build_decision("request_revision"),
    )

    assert updated.current_state == WorkflowState.MODELING
    assert updated.selected_model is None
    assert updated.selected_feature_columns is None