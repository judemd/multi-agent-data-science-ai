from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from workflow.modeling_transitions import (
    evaluate_and_transition_modeling,
)
from workflow.states import WorkflowState


SUPPORTED_MODELS = frozenset({
    "Logistic Regression",
    "Random Forest",
})


def apply_modeling_decision(
    project: ProjectState,
    decision: HITLDecision,
    selected_model: str | None = None,
    selected_feature_columns: list[str] | None = None,
) -> ProjectState:
    """Apply a Modeling human decision to persisted project state."""

    if project.current_state != WorkflowState.AWAITING_MODEL_SELECTION:
        raise ValueError(
            "Modeling decisions can only be applied while "
            "awaiting model selection."
        )

    if project.modeling_review is None:
        raise ValueError(
            "Cannot apply a Modeling decision without a review."
        )

    if decision.decision == "approve":
        if selected_model is None:
            raise ValueError(
                "An explicit model selection is required for approval."
            )

        if selected_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model selection: {selected_model}"
            )

        proposed_names = {
            model.name
            for model in project.modeling_review.proposed_models
        }

        if selected_model not in proposed_names:
            raise ValueError(
                "Selected model was not proposed by the Modeling Agent."
            )

        if not selected_feature_columns:
            raise ValueError(
                "Explicit feature selection is required for approval."
            )

        if len(selected_feature_columns) != len(set(selected_feature_columns)):
            raise ValueError(
                "Selected feature columns contain duplicates."
            )

        if project.modeling is None:
            raise ValueError(
                "Modeling evidence is required for feature approval."
            )

        if project.modeling.target_column in selected_feature_columns:
            raise ValueError(
                "Target column cannot be selected as a feature."
            )

        available_features = set(project.modeling.feature_columns)
        unknown_features = set(selected_feature_columns) - available_features

        if unknown_features:
            raise ValueError(
                f"Unknown selected feature columns: {sorted(unknown_features)}"
            )

    next_state = evaluate_and_transition_modeling(
        project.modeling_review,
        decision,
    )

    project.modeling_decision = decision

    if next_state == "evaluation":
        project.selected_model = selected_model
        project.selected_feature_columns = list(selected_feature_columns)
        project.current_state = WorkflowState.EVALUATION

    elif next_state == "revision":
        project.current_state = WorkflowState.MODELING
        project.modeling = None
        project.modeling_review = None
        project.selected_model = None
        project.selected_feature_columns = None

    else:
        project.current_state = WorkflowState.BLOCKED

    return project
