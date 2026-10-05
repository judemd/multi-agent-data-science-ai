import pandas as pd

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_action import DataPreparationAction
from domain.data_preparation_review import DataPreparationReview
from workflow.data_preparation_stage import (
    build_data_preparation_stage_artifact,
    prepare_data_preparation_stage,
    run_data_preparation_stage,
)


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "revenue": [100.0, None, 300.0],
            "segment": ["Enterprise", "enterprise ", "SMB"],
        }
    )


def test_stage_builds_preparation_artifact():
    result = build_data_preparation_stage_artifact(
        build_dataframe(),
        "customers.csv",
    )

    assert isinstance(result, DataPreparationArtifact)
    assert result.file_name == "customers.csv"
    assert result.missing_value_summary == {"revenue": 1}


def test_prepare_stage_contains_artifact_evidence():
    result = prepare_data_preparation_stage(
        build_dataframe(),
        "customers.csv",
    )

    assert isinstance(result, str)
    assert "DATA_PREPARATION_ARTIFACT" in result
    assert "revenue" in result
    assert "segment" in result


def test_run_stage_returns_agent_review(monkeypatch):
    expected_review = DataPreparationReview(
        observed_evidence=[
            "Revenue contains one missing value.",
        ],
        proposed_actions=[
            DataPreparationAction(
                operation="impute_missing",
                column="revenue",
                strategy="median",
                reason="Investigate the appropriate missing-value treatment.",
            ),
        ],
        rationale=[
            "The artifact identifies a missing revenue value.",
        ],
        requires_human_review=True,
    )

    def fake_run_agent(prompt: str) -> DataPreparationReview:
        assert "DATA_PREPARATION_ARTIFACT" in prompt
        assert "revenue" in prompt
        return expected_review

    monkeypatch.setattr(
        "workflow.data_preparation_stage.run_data_preparation_agent",
        fake_run_agent,
    )

    result = run_data_preparation_stage(
        build_dataframe(),
        "customers.csv",
    )

    assert result == expected_review
    assert result.requires_human_review is True


