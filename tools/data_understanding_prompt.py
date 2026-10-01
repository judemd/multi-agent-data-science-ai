import json

from domain.data_understanding import DataUnderstandingArtifact


def build_data_understanding_prompt(
    artifact: DataUnderstandingArtifact,
) -> str:
    """Serialize validated profiling evidence for the Data Understanding Agent."""

    evidence = artifact.model_dump(mode="json")

    return (
        "Review the following validated data-understanding evidence. "
        "Treat it as observed evidence and do not invent statistics.\n\n"
        "DATA_UNDERSTANDING_ARTIFACT:\n"
        f"{json.dumps(evidence, indent=2, sort_keys=True)}"
    )
