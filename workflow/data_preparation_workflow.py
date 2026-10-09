from pathlib import Path

import pandas as pd

from domain.hitl_decision import HITLDecision
from domain.data_preparation_action import DataPreparationAction
from domain.project_state import ProjectState
from tools.data_preparation_executor import (
    execute_data_preparation_actions,
)
from tools.dataset_fingerprint import fingerprint_dataset_file
from tools.data_loader import load_dataset
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)
from tools.data_preparation_treatment_plan_validator import (
    validate_data_preparation_treatment_plan,
)
from workflow.data_preparation_transitions import (
    evaluate_and_transition_data_preparation,
)
from workflow.states import WorkflowState


def apply_data_preparation_decision(
    project: ProjectState,
    decision: HITLDecision,
    dataframe: pd.DataFrame | None = None,
) -> ProjectState:
    """Apply a Data Preparation human decision to persisted project state."""

    if project.current_state != WorkflowState.AWAITING_PREPARATION_APPROVAL:
        raise ValueError(
            "Data Preparation decisions can only be applied while "
            "awaiting preparation approval."
        )

    if project.data_preparation_review is None:
        raise ValueError(
            "Cannot apply a Data Preparation decision without a review."
        )

    if (
        decision.decision == "approve"
        and project.data_preparation_treatment_plan is None
    ):
        raise ValueError(
            "Approval requires a human treatment plan."
        )

    if decision.decision == "approve":
        if project.data_preparation is None:
            raise ValueError(
                "Approval requires a reviewed Data Preparation artifact."
            )

        if (
            project.data_preparation_dataset_fingerprint is None
            or project.data_preparation_evidence_fingerprint is None
        ):
            raise ValueError(
                "Approval requires stored preparation fingerprints."
            )

        if not project.dataset_path:
            raise ValueError(
                "Approval requires a persisted project dataset."
            )

        current_dataset_fingerprint = fingerprint_dataset_file(
            project.dataset_path
        )

        if (
            current_dataset_fingerprint
            != project.data_preparation_dataset_fingerprint
        ):
            raise ValueError(
                "Current dataset fingerprint does not match "
                "the reviewed dataset."
            )

        if dataframe is None:
            raise ValueError(
                "A dataframe is required before approved Data Preparation "
                "actions can be executed."
            )

        persisted_dataframe = load_dataset(project.dataset_path)

        try:
            pd.testing.assert_frame_equal(
                dataframe,
                persisted_dataframe,
                check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError(
                "The supplied DataFrame does not match "
                "the persisted project dataset."
            ) from exc

        current_evidence_fingerprint = fingerprint_preparation_evidence(
            project.data_preparation
        )

        if (
            current_evidence_fingerprint
            != project.data_preparation_evidence_fingerprint
        ):
            raise ValueError(
                "Current preparation evidence fingerprint does not match "
                "the reviewed preparation evidence."
            )

        plan = project.data_preparation_treatment_plan

        if (
            plan.evidence_fingerprint
            != project.data_preparation_evidence_fingerprint
        ):
            raise ValueError(
                "Treatment plan evidence fingerprint does not match "
                "the reviewed preparation evidence."
            )

        if (
            plan.dataset_fingerprint
            != project.data_preparation_dataset_fingerprint
        ):
            raise ValueError(
                "Treatment plan dataset fingerprint does not match "
                "the reviewed dataset."
            )

    if decision.decision == "approve":
        validate_data_preparation_treatment_plan(
            project.data_preparation_treatment_plan,
            project.data_preparation,
            project.data_preparation_dataset_fingerprint,
        )

    next_state = evaluate_and_transition_data_preparation(
        project.data_preparation_review,
        decision,
    )

    if next_state == "modeling":
        if dataframe is None:
            raise ValueError(
                "A dataframe is required before approved Data Preparation "
                "actions can be executed."
            )

        approved_actions = []

        issues_by_id = {
            issue.issue_id: issue
            for issue in project.data_preparation.issues
        }

        for treatment_decision in (
            project.data_preparation_treatment_plan.decisions
        ):
            treatment = treatment_decision.treatment
            issue = issues_by_id[treatment_decision.issue_id]

            if treatment == "retain":
                continue

            if treatment in ("median", "mean", "mode"):
                if issue.column is None:
                    raise ValueError(
                        f"{treatment.capitalize()} imputation "
                        "requires an issue column."
                    )

                approved_actions.append(
                    DataPreparationAction(
                        operation="impute_missing",
                        column=issue.column,
                        strategy=treatment,
                        reason=treatment_decision.rationale,
                    )
                )

            elif treatment == "remove_duplicates":
                approved_actions.append(
                    DataPreparationAction(
                        operation="remove_duplicates",
                        reason=treatment_decision.rationale,
                    )
                )

            elif treatment == "drop_rows":
                if issue.column is None:
                    raise ValueError(
                        "drop_rows requires an issue column."
                    )

                approved_actions.append(
                    DataPreparationAction(
                        operation="drop_rows",
                        column=issue.column,
                        reason=treatment_decision.rationale,
                    )
                )

            elif treatment == "normalize_nulls":
                if issue.column is None:
                    raise ValueError(
                        "normalize_nulls requires an issue column."
                    )

                approved_actions.append(
                    DataPreparationAction(
                        operation="normalize_nulls",
                        column=issue.column,
                        reason=treatment_decision.rationale,
                    )
                )

            elif treatment == "normalize_categories":
                if issue.column is None:
                    raise ValueError(
                        "normalize_categories requires an issue column."
                    )

                approved_actions.append(
                    DataPreparationAction(
                        operation="normalize_categories",
                        column=issue.column,
                        reason=treatment_decision.rationale,
                    )
                )

            elif treatment == "trim_whitespace":
                if issue.column is None:
                    raise ValueError(
                        "trim_whitespace requires an issue column."
                    )

                approved_actions.append(
                    DataPreparationAction(
                        operation="trim_whitespace",
                        column=issue.column,
                        reason=treatment_decision.rationale,
                    )
                )

            elif treatment == "exclude_feature":
                if issue.column is None:
                    raise ValueError(
                        "exclude_feature requires an issue column."
                    )

                approved_actions.append(
                    DataPreparationAction(
                        operation="exclude_feature",
                        column=issue.column,
                        reason=treatment_decision.rationale,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported approved treatment: {treatment!r}"
                )

        # Apply approved text cleaning before missing-value imputation.
        # Stable sorting preserves relative order within each priority.
        operation_priority = {
            "normalize_nulls": 0,
            "trim_whitespace": 1,
            "normalize_categories": 2,
            "impute_missing": 3,
        }

        approved_actions.sort(
            key=lambda action: operation_priority.get(
                action.operation,
                4,
            )
        )

        prepared_dataframe = execute_data_preparation_actions(
            dataframe,
            approved_actions,
        )

        prepared_path = (
            Path("data/projects")
            / f"{project.project_id}_prepared.csv"
        )
        prepared_path.parent.mkdir(parents=True, exist_ok=True)

        import os
        import tempfile

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix=f".{project.project_id}_prepared_",
                dir=prepared_path.parent,
                delete=False,
                encoding="utf-8",
                newline="",
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            prepared_dataframe.to_csv(
                temporary_path,
                index=False,
            )

            os.replace(temporary_path, prepared_path)
            temporary_path = None

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        prepared_fingerprint = fingerprint_dataset_file(prepared_path)

        project.prepared_dataset_path = str(prepared_path)
        project.prepared_dataset_fingerprint = prepared_fingerprint
        project.current_state = WorkflowState.MODELING

    elif next_state == "revision":
        project.current_state = WorkflowState.DATA_PREPARATION
        project.data_preparation = None
        project.data_preparation_review = None

    else:
        project.current_state = WorkflowState.BLOCKED

    project.data_preparation_decision = decision

    return project
