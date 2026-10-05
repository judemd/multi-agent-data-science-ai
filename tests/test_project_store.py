from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from tools.project_store import ProjectStore
from workflow.states import WorkflowState


def test_project_can_be_saved_and_loaded(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))

    original = ProjectState(
        project_id="project-001",
        project_name="Customer Churn POC",
        dataset_path="datasets/customer_churn_poc.csv",
    )

    saved_path = store.save(original)
    loaded = store.load("project-001")

    assert saved_path.exists()
    assert loaded.project_id == original.project_id
    assert loaded.project_name == original.project_name
    assert loaded.dataset_path == original.dataset_path
    assert loaded.current_state == WorkflowState.PROBLEM_FRAMING
    assert loaded.revision == 1

def test_data_preparation_artifact_and_review_survive_save_and_load(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))

    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
        columns=[
            "customer_id",
            "revenue",
            "segment",
        ],
        missing_value_summary={
            "revenue": 4,
        },
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
        rationale=[
            "The artifact identifies missing revenue values.",
        ],
        requires_human_review=True,
    )

    original = ProjectState(
        project_id="project-002",
        project_name="Customer Churn Preparation",
        dataset_path="datasets/customers.csv",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        data_preparation=artifact,
        data_preparation_review=review,
    )
    original.prepared_dataset_path = "data/projects/prepared_dataset.csv"
    
    store.save(original)

    loaded = store.load("project-002")

    assert loaded.current_state == WorkflowState.AWAITING_PREPARATION_APPROVAL
    assert loaded.data_preparation == artifact
    assert loaded.data_preparation_review == review
    assert (
        loaded.prepared_dataset_path
        == "data/projects/prepared_dataset.csv"
    )


def test_data_preparation_decision_survives_save_and_load(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))

    decision = HITLDecision(
        decision="approve",
        reviewer="data_scientist",
        rationale="The proposed preparation actions are appropriate.",
    )

    original = ProjectState(
        project_id="project-005",
        project_name="Customer Churn",
        current_state=WorkflowState.MODELING,
        data_preparation_decision=decision,
    )

    store.save(original)

    loaded = store.load("project-005")

    assert loaded.data_preparation_decision == decision
def test_rejected_data_preparation_decision_survives_save_and_load(tmp_path):
    store = ProjectStore(storage_dir=str(tmp_path))

    decision = HITLDecision(
        decision="reject",
        reviewer="data_scientist",
        rationale="The proposed preparation requires unacceptable changes.",
    )

    original = ProjectState(
        project_id="project-006",
        project_name="Customer Churn",
        current_state=WorkflowState.BLOCKED,
        data_preparation_decision=decision,
    )

    store.save(original)

    loaded = store.load("project-006")

    assert loaded.current_state == WorkflowState.BLOCKED
    assert loaded.data_preparation_decision == decision


