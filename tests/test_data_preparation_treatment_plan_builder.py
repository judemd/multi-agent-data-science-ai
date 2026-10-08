import pytest

from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_issue import DataPreparationIssue
from tools.data_preparation_treatment_plan_builder import (
    build_data_preparation_treatment_plan,
)
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)


def test_builder_creates_valid_plan_from_explicit_human_selection():
    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="One missing revenue value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    dataset_fingerprint = "a" * 64
    evidence_fingerprint = fingerprint_preparation_evidence(artifact)

    plan = build_data_preparation_treatment_plan(
        artifact=artifact,
        dataset_fingerprint=dataset_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        reviewer="Data Scientist",
        selections={
            "missing_values:revenue": (
                "median",
                "Median is appropriate for this numeric feature.",
            ),
        },
    )

    assert plan.reviewer == "Data Scientist"
    assert plan.dataset_fingerprint == dataset_fingerprint
    assert plan.evidence_fingerprint == evidence_fingerprint
    assert len(plan.decisions) == 1
    assert plan.decisions[0].issue_id == "missing_values:revenue"
    assert plan.decisions[0].treatment == "median"
    assert (
        plan.decisions[0].rationale
        == "Median is appropriate for this numeric feature."
    )

def test_builder_rejects_missing_human_selection():
    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="One missing revenue value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Missing human treatment selection",
    ):
        build_data_preparation_treatment_plan(
            artifact=artifact,
            dataset_fingerprint="a" * 64,
            evidence_fingerprint=fingerprint_preparation_evidence(
                artifact
            ),
            reviewer="Data Scientist",
            selections={},
        )

def test_builder_rejects_impermissible_treatment():
    artifact = DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="One missing revenue value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="is not permitted",
    ):
        build_data_preparation_treatment_plan(
            artifact=artifact,
            dataset_fingerprint="a" * 64,
            evidence_fingerprint=fingerprint_preparation_evidence(
                artifact
            ),
            reviewer="Data Scientist",
            selections={
                "missing_values:revenue": (
                    "exclude_feature",
                    "Attempt to exclude this feature.",
                ),
            },
        )
