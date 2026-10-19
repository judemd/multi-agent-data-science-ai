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
        success_criteria=[business_context["success_metrics"].strip()]
        if business_context.get("success_metrics", "").strip() else [],
        assumptions=[],
        constraints=[business_context["constraints"].strip()]
        if business_context.get("constraints", "").strip() else [],
        risks=[business_context["risks"].strip()]
        if business_context.get("risks", "").strip() else [],
        questions_for_human=[],
    )
