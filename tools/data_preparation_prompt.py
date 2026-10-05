import json

from domain.data_preparation import DataPreparationArtifact


def build_data_preparation_prompt(
    artifact: DataPreparationArtifact,
) -> str:
    """Serialize validated preparation evidence for the preparation agent."""

    evidence = artifact.model_dump(mode="json")

    return (
        "Review the following validated data-preparation evidence. "
        "Treat it as observed evidence and do not invent statistics. "
        "Interpret the supplied evidence only; do not calculate new statistics. "
        "Do not modify the dataset. "
        "Return a JSON object matching the DataPreparationReview schema. "
        "The proposed_actions field must contain structured action objects, "
        "not natural-language strings. "
        "Each action must use one of these operations: "
        "impute_missing, remove_duplicates, exclude_feature. "
        "For impute_missing, provide the affected column and a supported "
        "strategy such as median or mean. "
        "For exclude_feature, provide the affected column. "
        "For remove_duplicates, a column is normally not required. "
        "Every proposed action must include an evidence-based reason. "
        "Only propose an action when the supplied evidence supports it. "
        "Proposed actions are recommendations for human review; "
        "do not claim that any action has already been executed.\n\n"
        "DATA_PREPARATION_ARTIFACT:\n"
        f"{json.dumps(evidence, indent=2, sort_keys=True)}"
    )
