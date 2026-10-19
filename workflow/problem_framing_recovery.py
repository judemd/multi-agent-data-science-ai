"""Explicit human approval of recovered framing for an existing project."""

from domain.hitl_decision import HITLDecision
from domain.problem_framing import ProblemFramingArtifact
from domain.project_state import ProjectState
from workflow.states import WorkflowState


def approve_recovered_framing(
    project: ProjectState,
    artifact: ProblemFramingArtifact,
    decision: HITLDecision,
) -> ProjectState:
    if project.current_state != WorkflowState.FINALIZATION:
        raise ValueError("Framing recovery requires a project awaiting Finalization.")
    if project.problem_framing is not None:
        raise ValueError("Existing Problem Framing evidence cannot be overwritten.")
    if decision.decision != "approve":
        raise ValueError("Recovered framing requires explicit human approval.")
    if not decision.reviewer.strip() or not decision.rationale.strip():
        raise ValueError("Enter a reviewer and rationale for framing recovery.")
    if any(not value.strip() for value in (
        artifact.business_problem, artifact.business_objective, artifact.target_outcome
    )):
        raise ValueError("Complete the business problem, objective, and desired outcome.")

    recovered = project.model_copy(deep=True)
    recovered.problem_framing = artifact.model_copy(deep=True)
    recovered.problem_framing_provenance = (
        "Recovered from business context and reviewed by a human at Finalization; "
        "this does not establish historical Problem Framing approval."
    )
    recovered.problem_framing_recovery_decision = decision.model_copy(deep=True)
    return recovered
