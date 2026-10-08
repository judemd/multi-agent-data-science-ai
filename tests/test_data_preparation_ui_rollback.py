import pandas as pd
from streamlit.testing.v1 import AppTest

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_issue import DataPreparationIssue
from domain.data_preparation_review import DataPreparationReview
from domain.data_preparation_treatment_decision import (
    DataPreparationTreatmentDecision,
)
from domain.data_preparation_treatment_plan import (
    DataPreparationTreatmentPlan,
)
from domain.project_state import ProjectState
from tools.dataset_fingerprint import fingerprint_dataset_file
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)
from workflow.states import WorkflowState


def test_failed_ui_approval_restores_previous_treatment_plan(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame(
        {"revenue": [10.0, None, 30.0]}
    )
    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="One missing revenue value.",
                allowed_treatments=["retain", "investigate"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = fingerprint_dataset_file(dataset_path)
    evidence_fingerprint = fingerprint_preparation_evidence(artifact)

    previous_plan = DataPreparationTreatmentPlan(
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="Previous Reviewer",
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="retain",
                rationale="Previously reviewed and retained.",
            ),
        ],
    )

    project = ProjectState(
        project_id="ui-rollback-test",
        project_name="UI Rollback Test",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        dataset_path=str(dataset_path),
        data_preparation=artifact,
        data_preparation_review=DataPreparationReview(
            observed_evidence=["Revenue contains missing values."],
            requires_human_review=True,
        ),
        data_preparation_dataset_fingerprint=dataset_fingerprint,
        data_preparation_evidence_fingerprint=evidence_fingerprint,
        data_preparation_treatment_plan=previous_plan,
    )

    monkeypatch.setattr(
        "ui_data_preparation.PROJECT_STORE.save",
        lambda project: None,
    )

    def render(project, dataframe):
        import streamlit as st
        from ui_data_preparation import (
            render_data_preparation_hitl_controls,
        )

        st.session_state["project_state"] = project
        render_data_preparation_hitl_controls(project, dataframe)

    at = AppTest.from_function(
        render,
        args=(project, dataframe),
    ).run()

    assert not at.exception

    at.selectbox[0].set_value("investigate")
    at.text_area[0].set_value("Investigate before proceeding.")
    at.text_input[0].set_value("Current Reviewer")
    at.text_area[1].set_value("Attempting approval for investigation.")

    at.button[0].click().run()

    assert not at.exception
    assert project.current_state == WorkflowState.AWAITING_PREPARATION_APPROVAL
    assert project.data_preparation_decision is None
    assert project.data_preparation_treatment_plan == previous_plan
    assert any(
        "Unsupported approved treatment" in item.value
        for item in at.error
    )

def test_incomplete_ui_treatment_selection_blocks_approval(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame({"revenue": [10.0, None, 30.0]})
    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="One missing revenue value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    project = ProjectState(
        project_id="ui-incomplete-test",
        project_name="UI Incomplete Test",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        dataset_path=str(dataset_path),
        data_preparation=artifact,
        data_preparation_review=DataPreparationReview(
            observed_evidence=["Revenue contains missing values."],
            requires_human_review=True,
        ),
        data_preparation_dataset_fingerprint=(
            fingerprint_dataset_file(dataset_path)
        ),
        data_preparation_evidence_fingerprint=(
            fingerprint_preparation_evidence(artifact)
        ),
    )

    def render(project, dataframe):
        from ui_data_preparation import (
            render_data_preparation_hitl_controls,
        )

        render_data_preparation_hitl_controls(project, dataframe)

    at = AppTest.from_function(
        render,
        args=(project, dataframe),
    ).run()

    assert not at.exception

    # Leave the treatment selector unselected.
    at.text_input[0].set_value("Current Reviewer")
    at.text_area[1].set_value("Reviewed the preparation evidence.")
    at.button[0].click().run()

    assert not at.exception
    assert project.current_state == WorkflowState.AWAITING_PREPARATION_APPROVAL
    assert project.data_preparation_decision is None
    assert project.data_preparation_treatment_plan is None
    assert any(
        "Select a treatment" in item.value
        for item in at.error
    )

def test_successful_ui_approval_creates_prepared_dataset(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    dataframe = pd.DataFrame({"revenue": [10.0, None, 30.0]})
    dataset_path = tmp_path / "customers.csv"
    dataframe.to_csv(dataset_path, index=False)

    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="One missing revenue value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    project = ProjectState(
        project_id="ui-success-test",
        project_name="UI Success Test",
        current_state=WorkflowState.AWAITING_PREPARATION_APPROVAL,
        dataset_path=str(dataset_path),
        data_preparation=artifact,
        data_preparation_review=DataPreparationReview(
            observed_evidence=["Revenue contains missing values."],
            requires_human_review=True,
        ),
        data_preparation_dataset_fingerprint=(
            fingerprint_dataset_file(dataset_path)
        ),
        data_preparation_evidence_fingerprint=(
            fingerprint_preparation_evidence(artifact)
        ),
    )

    monkeypatch.setattr(
        "ui_data_preparation.PROJECT_STORE.save",
        lambda project: None,
    )

    def render(project, dataframe):
        from ui_data_preparation import (
            render_data_preparation_hitl_controls,
        )

        render_data_preparation_hitl_controls(project, dataframe)

    at = AppTest.from_function(
        render,
        args=(project, dataframe),
    ).run()

    assert not at.exception

    next(
        widget for widget in at.selectbox
        if widget.label == "Treatment for missing_values:revenue *"
    ).set_value("median")

    next(
        widget for widget in at.text_area
        if widget.label == "Treatment rationale for missing_values:revenue *"
    ).set_value("Median is appropriate for the missing revenue value.")

    next(
        widget for widget in at.text_input
        if widget.label == "Reviewer *"
    ).set_value("Current Reviewer")

    next(
        widget for widget in at.text_area
        if widget.label == "Reviewer rationale *"
    ).set_value("The approved treatment addresses the missing value.")

    next(
        widget for widget in at.button
        if widget.label == "Approve Preparation"
    ).click().run()

    assert not at.exception
    assert not at.error
    assert project.current_state == WorkflowState.MODELING
    assert project.data_preparation_decision is not None
    assert project.data_preparation_decision.decision == "approve"
    assert project.data_preparation_treatment_plan is not None
    assert project.data_preparation_treatment_plan.reviewer == "Current Reviewer"
    assert len(project.data_preparation_treatment_plan.decisions) == 1
    assert project.data_preparation_treatment_plan.decisions[0].treatment == "median"

    assert project.prepared_dataset_path is not None
    prepared = pd.read_csv(project.prepared_dataset_path)

    assert prepared["revenue"].tolist() == [10.0, 20.0, 30.0]
    assert not prepared["revenue"].isna().any()
