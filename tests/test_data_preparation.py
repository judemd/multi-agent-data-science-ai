from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.project_state import ProjectState
from workflow.states import WorkflowState


def test_data_preparation_artifact_accepts_valid_evidence():
    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
        columns=[
            "customer_id",
            "revenue",
            "signup_date",
        ],
        datatype_summary={
            "customer_id": "object",
            "revenue": "float64",
            "signup_date": "object",
        },
        missing_value_summary={
            "revenue": 4,
        },
        candidate_date_columns=[
            "signup_date",
        ],
        potential_identifier_columns=[
            "customer_id",
        ],
        preparation_questions=[
            "Should signup_date be converted to a date datatype?",
        ],
    )

    assert artifact.file_name == "customers.csv"
    assert artifact.row_count == 100
    assert artifact.candidate_date_columns == ["signup_date"]
    assert artifact.potential_identifier_columns == ["customer_id"]


def test_project_state_supports_data_preparation_artifact():
    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
    )

    state = ProjectState(
        project_id="project-001",
        project_name="Customer Churn",
        current_state=WorkflowState.DATA_PREPARATION,
        data_preparation=artifact,
    )

    assert state.current_state == WorkflowState.DATA_PREPARATION
    assert state.data_preparation == artifact


def test_project_state_defaults_without_data_preparation():
    state = ProjectState(
        project_id="project-001",
        project_name="Customer Churn",
    )

    assert state.data_preparation is None

def test_project_state_persists_data_preparation_artifact_and_review():
    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
    )

    review = DataPreparationReview(
        observed_evidence=[
            "Revenue contains missing values.",
        ],
        proposed_actions=[
            DataPreparationAction(
                operation="impute_missing",
                column="revenue",
                strategy="median",
                reason="Investigate the appropriate missing-value treatment.",
            ),
        ],
        requires_human_review=True,
    )

    state = ProjectState(
        project_id="project-001",
        project_name="Customer Churn",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        data_preparation=artifact,
        data_preparation_review=review,
    )

    assert state.current_state == WorkflowState.AWAITING_PREPARATION_APPROVAL
    assert state.data_preparation == artifact
    assert state.data_preparation_review == review


