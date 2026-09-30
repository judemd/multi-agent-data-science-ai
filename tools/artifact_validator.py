from domain.problem_framing import ProblemFramingArtifact


def validate_problem_framing(
    data: dict,
) -> ProblemFramingArtifact:
    """Validate Business Analyst output against the workflow contract."""

    return ProblemFramingArtifact.model_validate(data)
