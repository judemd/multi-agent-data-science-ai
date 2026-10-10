from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from domain.project_state import ProjectState
from workflow.states import WorkflowState


def test_data_preparation_ui_module_imports():
    from ui_data_preparation import render_data_preparation_review

    assert callable(render_data_preparation_review)


def test_data_preparation_project_has_review():
    project = ProjectState(
        project_id="project-ui-001",
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
            requires_human_review=True,
        ),
    )

    assert project.data_preparation_review is not None
    assert project.current_state == WorkflowState.AWAITING_PREPARATION_APPROVAL





def test_data_preparation_ui_issue_labels_are_readable():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "ui_data_preparation.py"
    ).read_text(encoding="utf-8")

    assert source.count("{issue.issue_type} - ") == 2

    # Detect common signs of accidentally double-decoded UTF-8 text.
    for marker in ("\u00c3", "\u00c2", "\u00e2\u20ac", "\ufffd"):
        assert marker not in source
