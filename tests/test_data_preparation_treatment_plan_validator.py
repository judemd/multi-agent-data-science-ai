import pytest

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_issue import DataPreparationIssue
from domain.data_preparation_treatment_decision import (
    DataPreparationTreatmentDecision,
)
from domain.data_preparation_treatment_plan import (
    DataPreparationTreatmentPlan,
)
from tools.data_preparation_treatment_plan_validator import (
    validate_data_preparation_treatment_plan,
)
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)


DATASET_FINGERPRINT = "a" * 64


def build_artifact():
    return DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="Revenue has 1 missing value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )


def build_plan(artifact, **overrides):
    values = {
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "evidence_fingerprint": fingerprint_preparation_evidence(
            artifact
        ),
        "reviewer": "data_scientist",
        "decisions": [
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="median",
                rationale="Approved after reviewing the missing values.",
            ),
        ],
    }
    values.update(overrides)
    return DataPreparationTreatmentPlan(**values)


def test_accepts_complete_permitted_plan():
    artifact = build_artifact()
    plan = build_plan(artifact)

    validate_data_preparation_treatment_plan(
        plan, artifact, DATASET_FINGERPRINT
    )


def test_accepts_empty_plan_when_no_issues_exist():
    artifact = build_artifact()
    artifact.issues = []
    plan = build_plan(artifact, decisions=[])

    validate_data_preparation_treatment_plan(
        plan, artifact, DATASET_FINGERPRINT
    )


def test_rejects_stale_dataset_fingerprint():
    artifact = build_artifact()
    plan = build_plan(
        artifact,
        dataset_fingerprint="b" * 64,
    )

    with pytest.raises(ValueError, match="dataset fingerprint"):
        validate_data_preparation_treatment_plan(
            plan, artifact, DATASET_FINGERPRINT
        )


def test_rejects_stale_evidence_fingerprint():
    artifact = build_artifact()
    plan = build_plan(
        artifact,
        evidence_fingerprint="b" * 64,
    )

    with pytest.raises(ValueError, match="evidence fingerprint"):
        validate_data_preparation_treatment_plan(
            plan, artifact, DATASET_FINGERPRINT
        )


def test_rejects_unknown_issue_id():
    artifact = build_artifact()
    plan = build_plan(
        artifact,
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="unknown:revenue",
                treatment="retain",
                rationale="Reviewed.",
            ),
        ],
    )

    with pytest.raises(ValueError, match="unknown issue IDs"):
        validate_data_preparation_treatment_plan(
            plan, artifact, DATASET_FINGERPRINT
        )


def test_rejects_missing_issue_decision():
    artifact = build_artifact()
    plan = build_plan(artifact, decisions=[])

    with pytest.raises(ValueError, match="missing decisions"):
        validate_data_preparation_treatment_plan(
            plan, artifact, DATASET_FINGERPRINT
        )


def test_rejects_disallowed_treatment():
    artifact = build_artifact()
    plan = build_plan(
        artifact,
        decisions=[
            DataPreparationTreatmentDecision(
                issue_id="missing_values:revenue",
                treatment="exclude_feature",
                rationale="Reviewed.",
            ),
        ],
    )

    with pytest.raises(ValueError, match="not permitted"):
        validate_data_preparation_treatment_plan(
            plan, artifact, DATASET_FINGERPRINT
        )


def test_rejects_issue_without_stable_id():
    artifact = build_artifact()
    artifact.issues[0].issue_id = None
    plan = build_plan(artifact)

    with pytest.raises(ValueError, match="stable issue ID"):
        validate_data_preparation_treatment_plan(
            plan, artifact, DATASET_FINGERPRINT
        )


def test_rejects_duplicate_issue_ids_in_artifact():
    artifact = build_artifact()
    artifact.issues.append(
        artifact.issues[0].model_copy(deep=True)
    )
    plan = build_plan(artifact)

    with pytest.raises(ValueError, match="Duplicate preparation issue ID"):
        validate_data_preparation_treatment_plan(
            plan, artifact, DATASET_FINGERPRINT
        )
