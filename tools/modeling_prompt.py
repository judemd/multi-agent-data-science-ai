import json

from domain.modeling import ModelingArtifact


def build_modeling_prompt(
    artifact: ModelingArtifact,
) -> str:
    """Serialize validated modeling evidence for the modeling agent."""

    evidence = artifact.model_dump(mode="json")

    return (
        "Review the following validated modeling evidence. "
        "Treat it as observed evidence and do not invent statistics. "
        "Interpret the supplied evidence only; do not calculate new statistics. "
        "Do not train or evaluate models. "
        "Do not modify the dataset.\n\n"
        "MODELING_ARTIFACT:\n"
        f"{json.dumps(evidence, indent=2, sort_keys=True)}"
    )
