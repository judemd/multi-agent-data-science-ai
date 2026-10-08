import pytest
from pydantic import ValidationError

from domain.data_preparation_treatment_plan import (
    DataPreparationTreatmentPlan,
)
from domain.data_preparation_treatment_decision import (
    DataPreparationTreatmentDecision,
)


def build_plan(**overrides):
    values = {
        "dataset_fingerprint": "a" * 64,
        "evidence_fingerprint": "b" * 64,
        "reviewer": "data_scientist",
        "decisions": [
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="median",
                rationale="Approved median imputation.",
            ),
        ],
    }
    values.update(overrides)
    return DataPreparationTreatmentPlan(**values)


def test_accepts_valid_treatment_plan():
    plan = build_plan()

    assert plan.reviewer == "data_scientist"
    assert len(plan.decisions) == 1
    assert plan.decisions[0].treatment == "median"


def test_accepts_empty_decision_list():
    plan = build_plan(decisions=[])

    assert plan.decisions == []


def test_rejects_duplicate_issue_decisions():
    decision = DataPreparationTreatmentDecision(
        issue_id="missing_values:revenue",
        treatment="retain",
        rationale="Retain after human review.",
    )

    with pytest.raises(ValidationError, match="duplicate issue decisions"):
        build_plan(decisions=[decision, decision])


def test_rejects_invalid_dataset_fingerprint():
    with pytest.raises(ValidationError):
        build_plan(dataset_fingerprint="invalid")


def test_rejects_invalid_evidence_fingerprint():
    with pytest.raises(ValidationError):
        build_plan(evidence_fingerprint="invalid")


def test_rejects_empty_reviewer():
    with pytest.raises(ValidationError):
        build_plan(reviewer="")
