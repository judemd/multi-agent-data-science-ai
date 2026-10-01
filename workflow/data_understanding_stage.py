import pandas as pd

from domain.data_understanding import DataUnderstandingArtifact
from tools.data_understanding_eda import add_eda_evidence
from tools.data_understanding_prompt import build_data_understanding_prompt


def prepare_data_understanding_stage(
    artifact: DataUnderstandingArtifact,
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> str:
    """Prepare complete evidence for the Data Understanding Agent.

    Deterministic EDA is calculated from the supplied DataFrame and merged
    into the existing validated artifact before the agent prompt is built.

    The agent receives structured evidence and is responsible for
    interpretation, not statistical calculation.
    """

    enriched_artifact = add_eda_evidence(
        artifact,
        dataframe,
        target_column=target_column,
    )

    return build_data_understanding_prompt(enriched_artifact)
