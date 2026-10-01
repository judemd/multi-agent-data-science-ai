from domain.data_understanding import DataUnderstandingArtifact
from tools.data_understanding_prompt import build_data_understanding_prompt


def prepare_data_understanding_stage(
    artifact: DataUnderstandingArtifact,
) -> str:
    """
    Prepare validated profiling evidence for the Data Understanding Agent.

    This function deliberately performs no analysis or transformation.
    """

    return build_data_understanding_prompt(artifact)
