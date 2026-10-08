import pytest
import pandas as pd

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.hitl_decision import HITLDecision
from domain.project_state import ProjectState
from workflow.data_preparation_workflow import (
    apply_data_preparation_decision,
)
from workflow.states import WorkflowState


def build_project() -> ProjectState:
    return ProjectState(
        project_id="project-004",
        project_name="Customer Churn",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        data_preparation_review=DataPreparationReview(
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
            requires_human_review=False,
        ),
    )


def build_decision(decision: str) -> HITLDecision:
    return HITLDecision(
        decision=decision,
        reviewer="data_scientist",
        rationale="Decision based on the preparation review.",
    )

def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 2],
            "revenue": [10.0, None, 20.0],
        }
    )

def build_approvable_project(tmp_path):
    """Create a project with valid evidence and a human treatment plan."""
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    dataframe = build_dataframe()
    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="Revenue contains one missing value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint
    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="median",
                rationale="Approve median imputation for missing revenue.",
            ),
        ],
    )

    return project, dataframe

def test_approval_moves_project_to_modeling(tmp_path):
    project, dataframe = build_approvable_project(tmp_path)

    project = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    assert project.current_state == WorkflowState.MODELING


def test_revision_returns_project_to_data_preparation():
    project = apply_data_preparation_decision(
        build_project(),
        build_decision("request_revision"),
    )

    assert project.current_state == WorkflowState.DATA_PREPARATION


def test_rejection_moves_project_to_blocked():
    project = apply_data_preparation_decision(
        build_project(),
        build_decision("reject"),
    )

    assert project.current_state == WorkflowState.BLOCKED


def test_decision_requires_existing_review():
    project = build_project()
    project.data_preparation_review = None

    with pytest.raises(
        ValueError,
        match="without a review",
    ):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
        )


def test_decision_requires_approval_state():
    project = build_project()
    project.current_state = WorkflowState.DATA_PREPARATION

    with pytest.raises(
        ValueError,
        match="awaiting preparation approval",
    ):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
        )


def test_review_requiring_human_review_can_progress_after_approval(tmp_path):
    project, dataframe = build_approvable_project(tmp_path)
    project.data_preparation_review.requires_human_review = True

    project = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    assert project.current_state == WorkflowState.MODELING
    
def test_revision_clears_previous_data_preparation_review():
    project = build_project()

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=100,
        column_count=3,
    )

    project.data_preparation_review = DataPreparationReview(
        observed_evidence=["Previous evidence"],
        requires_human_review=True,
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("request_revision"),
    )

    assert updated.current_state == WorkflowState.DATA_PREPARATION
    assert updated.data_preparation is None
    assert updated.data_preparation_review is None

def test_approval_does_not_progress_when_execution_fails(tmp_path):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    dataframe = build_dataframe()
    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="Revenue contains one missing value.",
                allowed_treatments=["investigate", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="investigate",
                rationale="Investigate the missing values before proceeding.",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Unsupported approved treatment"):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            dataframe,
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None

def test_approval_requires_human_treatment_plan():
    project = build_project()

    with pytest.raises(ValueError, match="human treatment plan"):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            build_dataframe(),
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None


def test_approval_rejects_stale_treatment_plan(tmp_path):
    project, dataframe = build_approvable_project(tmp_path)

    original_fingerprint = (
        project.data_preparation_treatment_plan.evidence_fingerprint
    )
    stale_fingerprint = (
        "a" * 64 if original_fingerprint != "a" * 64 else "b" * 64
    )

    project.data_preparation_treatment_plan.evidence_fingerprint = (
        stale_fingerprint
    )

    with pytest.raises(ValueError, match="evidence fingerprint"):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            dataframe,
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None

def test_approval_rejects_modified_preparation_evidence(tmp_path):
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    from tools.dataset_fingerprint import fingerprint_dataset_file

    dataset_path = tmp_path / "customers.csv"
    build_dataframe().to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
    )

    original_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = original_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=original_fingerprint,
        reviewer="data_scientist",
        decisions=[],
    )

    # Simulate evidence being modified after human review.
    project.data_preparation.row_count = 999

    with pytest.raises(ValueError, match="evidence fingerprint"):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            build_dataframe(),
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None


def test_approval_rejects_modified_dataset_file(tmp_path):
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    dataset_path = tmp_path / "customers.csv"
    original_dataframe = build_dataframe()
    original_dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[],
    )

    # The reviewed file changes before the approval is submitted.
    dataset_path.write_text(
        "customer_id,revenue\n1,999\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset fingerprint"):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            original_dataframe,
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None


def test_approval_rejects_dataframe_different_from_persisted_dataset(tmp_path):
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    dataset_path = tmp_path / "customers.csv"
    original_dataframe = build_dataframe()
    original_dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[],
    )

    # The file is unchanged, but the in-memory DataFrame is different.
    modified_dataframe = original_dataframe.copy()
    modified_dataframe.loc[0, "revenue"] = 999.0

    with pytest.raises(ValueError, match="persisted project dataset"):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            modified_dataframe,
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None



def test_human_retain_decision_overrides_agent_proposed_actions(
    tmp_path, monkeypatch
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "revenue": [10.0, 10.0, 20.0],
        }
    )

    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="exact_duplicates:dataset",
                issue_type="exact_duplicates",
                column=None,
                evidence="One exact duplicate row exists.",
                allowed_treatments=["remove_duplicates", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    project.data_preparation_review.proposed_actions = [
        DataPreparationAction(
            operation="remove_duplicates",
            reason="Agent recommends removing exact duplicates.",
        ),
    ]

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="exact_duplicates:dataset",
                treatment="retain",
                rationale="Duplicates are approved for retention.",
            ),
        ],
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    prepared = pd.read_csv(updated.prepared_dataset_path)

    assert updated.current_state == WorkflowState.MODELING
    assert len(prepared) == 3
    assert prepared.equals(dataframe)



def test_human_approved_duplicate_removal_is_executed(
    tmp_path, monkeypatch
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "revenue": [10.0, 10.0, 20.0],
        }
    )

    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="exact_duplicates:dataset",
                issue_type="exact_duplicates",
                column=None,
                evidence="One exact duplicate row exists.",
                allowed_treatments=["remove_duplicates", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    project.data_preparation_review.proposed_actions = [
        DataPreparationAction(
            operation="remove_duplicates",
            reason="Agent recommends removing exact duplicates.",
        ),
    ]

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="exact_duplicates:dataset",
                treatment="remove_duplicates",
                rationale="Human approved removal of exact duplicates.",
            ),
        ],
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    prepared = pd.read_csv(updated.prepared_dataset_path)

    assert updated.current_state == WorkflowState.MODELING
    assert len(prepared) == 2
    assert prepared.equals(dataframe.drop_duplicates().reset_index(drop=True))




def test_approval_does_not_progress_when_output_write_fails(
    tmp_path, monkeypatch
):
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = build_dataframe()
    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[],
    )

    def fail_csv_write(self, *args, **kwargs):
        raise OSError("Simulated prepared dataset write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv_write)

    with pytest.raises(
        OSError,
        match="Simulated prepared dataset write failure",
    ):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            dataframe,
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None


def test_partial_output_write_preserves_existing_prepared_dataset(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = build_dataframe()
    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[],
    )

    prepared_path = (
        tmp_path / "data" / "projects"
        / f"{project.project_id}_prepared.csv"
    )
    prepared_path.parent.mkdir(parents=True, exist_ok=True)

    original_contents = b"existing,approved\n1,42\n"
    prepared_path.write_bytes(original_contents)

    def partially_write_then_fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b"customer_id,revenue\n1,")
        raise OSError("Simulated partial CSV write")

    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        partially_write_then_fail,
    )

    with pytest.raises(OSError, match="Simulated partial CSV write"):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            dataframe,
        )

    assert prepared_path.read_bytes() == original_contents
    assert list(
        prepared_path.parent.glob(
            f".{project.project_id}_prepared_*.csv"
        )
    ) == []

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None


def test_human_approved_median_imputation_is_executed(
    tmp_path, monkeypatch
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "revenue": [10.0, None, 30.0, 50.0],
        }
    )

    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=4,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="Revenue contains one missing value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="median",
                rationale="Use the observed numeric median.",
            ),
        ],
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    prepared = pd.read_csv(updated.prepared_dataset_path)

    assert updated.current_state == WorkflowState.MODELING
    assert prepared["revenue"].tolist() == [
        10.0,
        30.0,
        30.0,
        50.0,
    ]
    assert prepared["revenue"].isna().sum() == 0
    assert prepared["customer_id"].tolist() == [1, 2, 3, 4]


def test_human_approved_median_rejects_nonnumeric_column(
    tmp_path, monkeypatch
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "segment": ["retail", None, "business"],
        }
    )

    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:segment",
                issue_type="missing_values",
                column="segment",
                evidence="Segment contains one missing value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:segment",
                treatment="median",
                rationale="Attempt median imputation on segment.",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Median imputation requires a numeric column",
    ):
        apply_data_preparation_decision(
            project,
            build_decision("approve"),
            dataframe,
        )

    assert project.current_state == (
        WorkflowState.AWAITING_PREPARATION_APPROVAL
    )
    assert project.data_preparation_decision is None
    assert project.prepared_dataset_path is None


def test_human_approved_mean_imputation_is_executed(
    tmp_path, monkeypatch
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "revenue": [10.0, None, 20.0, 60.0],
        }
    )

    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)
    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=4,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="Revenue contains one missing value.",
                allowed_treatments=["mean", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="mean",
                rationale="Use the observed numeric mean.",
            ),
        ],
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    prepared = pd.read_csv(updated.prepared_dataset_path)

    assert updated.current_state == WorkflowState.MODELING
    assert prepared["revenue"].tolist() == [
        10.0,
        30.0,
        20.0,
        60.0,
    ]
    assert prepared["revenue"].isna().sum() == 0
    assert prepared["customer_id"].tolist() == [1, 2, 3, 4]


def test_human_approved_mode_imputation_is_executed(
    tmp_path, monkeypatch
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "segment": ["business", None, "retail", "business"],
        }
    )

    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=4,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:segment",
                issue_type="missing_values",
                column="segment",
                evidence="Segment contains one missing value.",
                allowed_treatments=["mode", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:segment",
                treatment="mode",
                rationale="Use the most frequent observed category.",
            ),
        ],
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    prepared = pd.read_csv(updated.prepared_dataset_path)

    assert updated.current_state == WorkflowState.MODELING
    assert prepared["segment"].tolist() == [
        "business",
        "business",
        "retail",
        "business",
    ]
    assert prepared["segment"].isna().sum() == 0
    assert prepared["customer_id"].tolist() == [1, 2, 3, 4]


def test_human_approved_drop_rows_is_executed(tmp_path, monkeypatch):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "revenue": [10.0, None, 30.0, 40.0],
            "segment": ["retail", "business", None, "retail"],
        }
    )

    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    project = build_project()
    project.dataset_path = str(dataset_path)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=4,
        column_count=3,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="Revenue contains one missing value.",
                allowed_treatments=["drop_rows", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="drop_rows",
                rationale="Remove records with missing revenue.",
            ),
        ],
    )

    updated = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    prepared = pd.read_csv(updated.prepared_dataset_path)

    assert updated.current_state == WorkflowState.MODELING
    assert prepared["customer_id"].tolist() == [1, 3, 4]
    assert prepared["revenue"].tolist() == [10.0, 30.0, 40.0]

    # A missing value in an unrelated column must not remove its row.
    assert pd.isna(prepared.loc[1, "segment"])

    # The original DataFrame must remain unchanged.
    assert len(dataframe) == 4
    assert pd.isna(dataframe.loc[1, "revenue"])

def test_approved_trim_whitespace_executes_before_modeling(
    tmp_path,
    monkeypatch,
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)

    project, _ = build_approvable_project(tmp_path)

    dataframe = pd.DataFrame(
        {
            "status": [" Active ", "Pending  ", "  Active"],
            "revenue": [10.0, 20.0, 30.0],
        }
    )
    dataframe.to_csv(project.dataset_path, index=False)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="formatting_issue:status",
                issue_type="formatting_issue",
                column="status",
                evidence="Status values contain surrounding whitespace.",
                allowed_treatments=["trim_whitespace", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    from tools.dataset_fingerprint import fingerprint_dataset_file

    dataset_fingerprint = fingerprint_dataset_file(project.dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint
    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="formatting_issue:status",
                treatment="trim_whitespace",
                rationale="Approved removal of surrounding whitespace.",
            ),
        ],
    )

    result = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    assert result.current_state == WorkflowState.MODELING
    assert result.data_preparation_decision.decision == "approve"
    assert result.prepared_dataset_path is not None

    prepared = pd.read_csv(result.prepared_dataset_path)
    assert prepared["status"].tolist() == [
        "Active",
        "Pending",
        "Active",
    ]
    assert prepared["revenue"].tolist() == [10.0, 20.0, 30.0]

    original = pd.read_csv(project.dataset_path)
    assert original["status"].tolist() == [
        " Active ",
        "Pending  ",
        "  Active",
    ]

def test_approved_normalize_nulls_runs_before_imputation(tmp_path, monkeypatch):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import DataPreparationTreatmentPlan
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)
    project, _ = build_approvable_project(tmp_path)

    dataframe = pd.DataFrame(
        {
            "status": ["Active", "Active", None, " N/A "],
            "revenue": [10.0, 20.0, 30.0, 40.0],
        }
    )
    dataframe.to_csv(project.dataset_path, index=False)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=4,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:status",
                issue_type="missing_values",
                column="status",
                evidence="One genuine missing value.",
                allowed_treatments=["mode", "retain"],
            ),
            DataPreparationIssue(
                issue_id="fake_nulls:status",
                issue_type="fake_nulls",
                column="status",
                evidence="One textual null placeholder.",
                allowed_treatments=["normalize_nulls", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(project.dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:status",
                treatment="mode",
                rationale="Approved mode imputation.",
            ),
            DataPreparationTreatmentDecision(
                issue_id="fake_nulls:status",
                treatment="normalize_nulls",
                rationale="Approved null normalization.",
            ),
        ],
    )

    result = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    assert result.current_state == WorkflowState.MODELING
    assert result.data_preparation_decision.decision == "approve"
    assert result.prepared_dataset_path is not None

    prepared = pd.read_csv(result.prepared_dataset_path)

    assert prepared["status"].tolist() == [
        "Active",
        "Active",
        "Active",
        "Active",
    ]
    assert prepared["revenue"].tolist() == [10.0, 20.0, 30.0, 40.0]

    original = pd.read_csv(project.dataset_path)
    assert pd.isna(original.loc[2, "status"])
    assert original.loc[3, "status"] == " N/A "


def test_approved_normalize_categories_executes_before_modeling(
    tmp_path,
    monkeypatch,
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)
    project, _ = build_approvable_project(tmp_path)

    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "consumer",
                "Consumer",
                " Consumer ",
                "Business",
                "business",
                "business",
            ],
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        }
    )
    dataframe.to_csv(project.dataset_path, index=False)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=7,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="categorical_inconsistency:segment",
                issue_type="categorical_inconsistency",
                column="segment",
                evidence="Segment values contain case and whitespace variants.",
                allowed_treatments=["normalize_categories", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(project.dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint
    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="categorical_inconsistency:segment",
                treatment="normalize_categories",
                rationale="Approved consolidation of observed category variants.",
            ),
        ],
    )

    result = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    assert result.current_state == WorkflowState.MODELING
    assert result.data_preparation_decision.decision == "approve"
    assert result.prepared_dataset_path is not None

    prepared = pd.read_csv(result.prepared_dataset_path)

    assert prepared["segment"].tolist() == [
        "Consumer",
        "Consumer",
        "Consumer",
        "Consumer",
        "business",
        "business",
        "business",
    ]
    assert prepared["revenue"].tolist() == [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
    ]

    original = pd.read_csv(project.dataset_path)
    assert original["segment"].tolist() == [
        "Consumer",
        "consumer",
        "Consumer",
        " Consumer ",
        "Business",
        "business",
        "business",
    ]


def test_category_normalization_precedes_mode_imputation(
    tmp_path,
    monkeypatch,
):
    from domain.data_preparation_issue import DataPreparationIssue
    from domain.data_preparation_treatment_decision import (
        DataPreparationTreatmentDecision,
    )
    from domain.data_preparation_treatment_plan import (
        DataPreparationTreatmentPlan,
    )
    from tools.dataset_fingerprint import fingerprint_dataset_file
    from tools.preparation_evidence_fingerprint import (
        fingerprint_preparation_evidence,
    )

    monkeypatch.chdir(tmp_path)
    project, _ = build_approvable_project(tmp_path)

    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "consumer",
                "Consumer",
                "Business",
                "Business",
                None,
            ],
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    dataframe.to_csv(project.dataset_path, index=False)

    project.data_preparation = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=6,
        column_count=2,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:segment",
                issue_type="missing_values",
                column="segment",
                evidence="One genuine missing value.",
                allowed_treatments=["mode", "retain"],
            ),
            DataPreparationIssue(
                issue_id="categorical_inconsistency:segment",
                issue_type="categorical_inconsistency",
                column="segment",
                evidence="Consumer and consumer are case variants.",
                allowed_treatments=["normalize_categories", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(project.dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    project.data_preparation_dataset_fingerprint = dataset_fingerprint
    project.data_preparation_evidence_fingerprint = evidence_fingerprint

    # Intentionally list imputation before categorical normalization.
    project.data_preparation_treatment_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="data_scientist",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:segment",
                treatment="mode",
                rationale="Approved mode imputation.",
            ),
            DataPreparationTreatmentDecision(
                issue_id="categorical_inconsistency:segment",
                treatment="normalize_categories",
                rationale="Approved consolidation of category variants.",
            ),
        ],
    )

    result = apply_data_preparation_decision(
        project,
        build_decision("approve"),
        dataframe,
    )

    assert result.current_state == WorkflowState.MODELING
    assert result.data_preparation_decision.decision == "approve"
    assert result.prepared_dataset_path is not None

    prepared = pd.read_csv(result.prepared_dataset_path)

    assert prepared["segment"].tolist() == [
        "Consumer",
        "Consumer",
        "Consumer",
        "Business",
        "Business",
        "Consumer",
    ]

    original = pd.read_csv(project.dataset_path)
    assert pd.isna(original.loc[5, "segment"])
    assert original.loc[1, "segment"] == "consumer"