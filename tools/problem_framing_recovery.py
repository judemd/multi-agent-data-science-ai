"""Reconstruct Problem Framing from preserved business context."""

from collections.abc import Mapping

from domain.problem_framing import ProblemFramingArtifact


def reconstruct_problem_framing(
    business_context: Mapping[str, str],
) -> ProblemFramingArtifact:
    """Reconstruct content only; this does not imply historical approval."""

    return ProblemFramingArtifact(
        business_problem=business_context.get("business_problem", ""),
        business_objective=business_context.get("business_objective", ""),
        target_outcome=business_context.get("desired_outcome", ""),
        success_criteria=[],
        assumptions=[],
        constraints=[],
        risks=[],
        questions_for_human=[],
    )
