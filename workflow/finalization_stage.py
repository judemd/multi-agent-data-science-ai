"""Deterministic Finalization handoff assembly."""

from domain.finalization import FinalizationArtifact
from tools.executive_evaluation_summary import (
    build_executive_evaluation_summary,
)
from domain.project_state import ProjectState
from tools.data_preparation_treatment_plan_validator import (
    validate_data_preparation_treatment_plan,
)
from tools.dataset_fingerprint import fingerprint_dataset_file
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)
from tools.finalization_evidence_fingerprint import (
    fingerprint_finalization_evidence,
)
from workflow.states import WorkflowState


def run_finalization_stage(project: ProjectState) -> ProjectState:
    """Assemble verified evidence for separate human handoff approval."""

    if project.current_state != WorkflowState.FINALIZATION:
        raise ValueError("Finalization can only run from FINALIZATION.")

    if project.problem_framing is None:
        raise ValueError("Finalization requires Problem Framing evidence.")

    if project.problem_framing_provenance is not None:
        if not project.problem_framing_provenance.strip():
            raise ValueError(
                "Finalization requires meaningful Problem Framing provenance."
            )

        recovery_decision = project.problem_framing_recovery_decision
        if (
            recovery_decision is None
            or recovery_decision.decision != "approve"
        ):
            raise ValueError(
                "Finalization requires human approval of reconstructed "
                "Problem Framing evidence."
            )
    if project.data_preparation is None:
        raise ValueError("Finalization requires Data Preparation evidence.")

    if project.data_preparation_treatment_plan is None:
        raise ValueError("Finalization requires the approved treatment plan.")

    if project.evaluation is None:
        raise ValueError("Finalization requires Evaluation evidence.")

    decisions = {
        "Data Preparation": project.data_preparation_decision,
        "Modeling": project.modeling_decision,
        "Evaluation": project.evaluation_decision,
    }

    for stage_name, decision in decisions.items():
        if decision is None or decision.decision != "approve":
            raise ValueError(
                f"Finalization requires an approved {stage_name} decision."
            )
        if not decision.reviewer.strip() or not decision.rationale.strip():
            raise ValueError(
                f"Finalization requires complete {stage_name} reviewer evidence."
            )

    if not project.dataset_path:
        raise ValueError("Finalization requires a source dataset path.")

    if not project.prepared_dataset_path:
        raise ValueError("Finalization requires a prepared dataset path.")

    if not project.prepared_dataset_fingerprint:
        raise ValueError("Finalization requires an approved dataset fingerprint.")

    if not project.data_preparation_dataset_fingerprint:
        raise ValueError(
            "Finalization requires the approved source dataset fingerprint."
        )

    try:
        source_fingerprint = fingerprint_dataset_file(
            project.dataset_path
        )
    except OSError as exc:
        raise ValueError(
            "Finalization cannot read the approved source dataset."
        ) from exc

    if source_fingerprint != project.data_preparation_dataset_fingerprint:
        raise ValueError(
            "Finalization source dataset fingerprint mismatch."
        )

    plan = project.data_preparation_treatment_plan

    if (
        plan.dataset_fingerprint
        != project.data_preparation_dataset_fingerprint
        or plan.evidence_fingerprint
        != project.data_preparation_evidence_fingerprint
    ):
        raise ValueError(
            "Finalization treatment plan fingerprints do not match "
            "the approved preparation evidence."
        )

    current_evidence_fingerprint = fingerprint_preparation_evidence(
        project.data_preparation
    )

    if (
        current_evidence_fingerprint
        != project.data_preparation_evidence_fingerprint
    ):
        raise ValueError(
            "Finalization preparation evidence fingerprint mismatch."
        )

    validate_data_preparation_treatment_plan(
        plan,
        project.data_preparation,
        project.data_preparation_dataset_fingerprint,
    )

    try:
        current_fingerprint = fingerprint_dataset_file(
            project.prepared_dataset_path
        )
    except OSError as exc:
        raise ValueError(
            "Finalization cannot read the approved prepared dataset."
        ) from exc

    if current_fingerprint != project.prepared_dataset_fingerprint:
        raise ValueError(
            "Finalization fingerprint mismatch: the approved prepared "
            "dataset has changed."
        )

    if (
        project.selected_model != project.evaluation.selected_model
        or project.selected_feature_columns != project.evaluation.feature_columns
    ):
        raise ValueError(
            "Finalization model selection does not match Evaluation evidence."
        )

    if (
        project.modeling is None
        or project.modeling.target_column != project.evaluation.target_column
    ):
        raise ValueError(
            "Finalization target does not match Modeling evidence."
        )

    issues_by_id = {
        issue.issue_id: issue
        for issue in project.data_preparation.issues
    }

    unresolved_risks = [
        "Historical Problem Framing human approval provenance "
        "is not recorded in ProjectState."
    ]

    if project.problem_framing_provenance is not None:
        recovery_decision = project.problem_framing_recovery_decision
        unresolved_risks.append(
            "Problem Framing was reconstructed rather than recovered "
            "as an original structured artifact. Provenance: "
            f"{project.problem_framing_provenance} "
            "New reconstruction approval reviewer: "
            f"{recovery_decision.reviewer}; "
            f"rationale: {recovery_decision.rationale}. "
            "This approval does not establish historical approval."
        )
    for treatment_decision in plan.decisions:
        if treatment_decision.treatment not in ("retain", "investigate"):
            continue

        issue = issues_by_id.get(treatment_decision.issue_id)

        if issue is None:
            raise ValueError(
                "Finalization treatment plan references an unknown issue."
            )

        column = issue.column or "dataset-wide"

        unresolved_risks.append(
            f"Retained or pending issue: {issue.issue_type} "
            f"({column}); evidence: {issue.evidence}; "
            f"human rationale: {treatment_decision.rationale}"
        )

    unresolved_risks.extend(project.evaluation.limitations)
    unresolved_risks.append(
        "Evaluation GO authorizes handoff review only; "
        "it does not authorize deployment."
    )

    artifact = FinalizationArtifact(
        project_id=project.project_id,
        project_name=project.project_name,
        revision=project.revision,
        problem_framing=project.problem_framing.model_copy(deep=True),
        source_dataset_path=project.dataset_path,
        prepared_dataset_path=project.prepared_dataset_path,
        prepared_dataset_fingerprint=project.prepared_dataset_fingerprint,
        data_preparation=project.data_preparation.model_copy(deep=True),
        treatment_plan=plan.model_copy(deep=True),
        preparation_decision=project.data_preparation_decision.model_copy(
            deep=True
        ),
        evaluation=project.evaluation.model_copy(deep=True),
        modeling_decision=project.modeling_decision.model_copy(deep=True),
        evaluation_decision=project.evaluation_decision.model_copy(deep=True),
        unresolved_risks=unresolved_risks,
        executive_evaluation_summary=(
            build_executive_evaluation_summary(
                project.evaluation,
                project.evaluation_decision,
            )
        ),
        revision_resolution=(
            project.finalization_revision_resolution.model_copy(deep=True)
            if project.finalization_revision_resolution is not None
            else None
        ),
        deployment_approved=False,
        requires_human_handoff_approval=True,
    )

    if (
        project.handoff_decision is not None
        and project.handoff_decision.decision == "request_revision"
    ):
        raise ValueError(
            "Finalization revision feedback remains unresolved. "
            "A human-reviewed resolution is required before rebuilding "
            "the handoff package."
        )

    project.finalization = artifact
    project.finalization_evidence_fingerprint = (
        fingerprint_finalization_evidence(artifact)
    )
    project.handoff_decision = None
    project.current_state = WorkflowState.AWAITING_HANDOFF_APPROVAL

    return project