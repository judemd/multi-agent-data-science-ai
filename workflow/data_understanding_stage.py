import pandas as pd

from agents.data_understanding_runner import run_data_understanding_agent
from domain.data_understanding import DataUnderstandingArtifact
from domain.data_understanding_review import DataUnderstandingReview
from domain.data_understanding_stage_result import DataUnderstandingStageResult
from tools.data_understanding_eda import add_eda_evidence
from tools.data_understanding_prompt import build_data_understanding_prompt


def build_data_understanding_evidence(
    artifact: DataUnderstandingArtifact,
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> DataUnderstandingArtifact:
    """Build complete deterministic evidence without modifying the input artifact."""

    return add_eda_evidence(
        artifact,
        dataframe,
        target_column=target_column,
    )


def prepare_data_understanding_stage(
    artifact: DataUnderstandingArtifact,
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> str:
    """Prepare complete evidence for the Data Understanding Agent."""

    enriched_artifact = build_data_understanding_evidence(
        artifact,
        dataframe,
        target_column=target_column,
    )

    return build_data_understanding_prompt(enriched_artifact)


def run_data_understanding_stage(
    artifact: DataUnderstandingArtifact,
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> DataUnderstandingReview:
    """Run deterministic evidence preparation and agent interpretation."""

    enriched_artifact = build_data_understanding_evidence(
        artifact,
        dataframe,
        target_column=target_column,
    )

    prompt = build_data_understanding_prompt(
        enriched_artifact,
    )

    return run_data_understanding_agent(prompt)


def run_data_understanding_stage_with_artifact(
    artifact: DataUnderstandingArtifact,
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> DataUnderstandingStageResult:
    """Run the stage and return the exact evidence supplied to the agent."""

    enriched_artifact = build_data_understanding_evidence(
        artifact,
        dataframe,
        target_column=target_column,
    )

    prompt = build_data_understanding_prompt(
        enriched_artifact,
    )

    review = run_data_understanding_agent(prompt)

    return DataUnderstandingStageResult(
        artifact=enriched_artifact,
        review=review,
    )
