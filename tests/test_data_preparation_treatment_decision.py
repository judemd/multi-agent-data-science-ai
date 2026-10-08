import pytest
from pydantic import ValidationError

from domain.data_preparation_treatment_decision import (
    DataPreparationTreatmentDecision,
)


def test_accepts_valid_human_treatment_decision():
    decision = DataPreparationTreatmentDecision(
        issue_id="missing_values:revenue",
        treatment="median",
        rationale="Revenue is numeric and median imputation is approved.",
    )

    assert decision.issue_id == "missing_values:revenue"
    assert decision.treatment == "median"


def test_accepts_retain_without_transformation():
    decision = DataPreparationTreatmentDecision(
        issue_id="outlier:revenue",
        treatment="retain",
        rationale="The observed values are valid business transactions.",
    )

    assert decision.treatment == "retain"


def test_rejects_empty_issue_id():
    with pytest.raises(ValidationError):
        DataPreparationTreatmentDecision(
            issue_id="",
            treatment="retain",
            rationale="Reviewed by the data scientist.",
        )


def test_rejects_unsupported_treatment():
    with pytest.raises(ValidationError):
        DataPreparationTreatmentDecision(
            issue_id="missing_values:revenue",
            treatment="delete_everything",
            rationale="Reviewed by the data scientist.",
        )


def test_rejects_empty_rationale():
    with pytest.raises(ValidationError):
        DataPreparationTreatmentDecision(
            issue_id="missing_values:revenue",
            treatment="median",
            rationale="",
        )
