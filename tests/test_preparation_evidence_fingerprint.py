from domain.data_preparation import DataPreparationArtifact
from domain.data_preparation_issue import DataPreparationIssue
from tools.preparation_evidence_fingerprint import (
    fingerprint_preparation_evidence,
)


def build_artifact() -> DataPreparationArtifact:
    return DataPreparationArtifact(
        file_name="customers.csv",
        row_count=3,
        column_count=1,
        columns=["revenue"],
        datatype_summary={"revenue": "float64"},
        missing_value_summary={"revenue": 1},
        issues=[
            DataPreparationIssue(
                issue_id="missing_values:revenue",
                issue_type="missing_values",
                column="revenue",
                evidence="One missing value.",
                allowed_treatments=["median", "retain"],
                requires_explicit_human_decision=True,
            )
        ],
    )


def test_evidence_fingerprint_is_stable():
    artifact = build_artifact()

    assert fingerprint_preparation_evidence(
        artifact
    ) == fingerprint_preparation_evidence(artifact)


def test_evidence_fingerprint_ignores_dictionary_insertion_order():
    first = build_artifact()
    first.datatype_summary = {
        "revenue": "float64",
        "customer_id": "object",
    }

    second = build_artifact()
    second.datatype_summary = {
        "customer_id": "object",
        "revenue": "float64",
    }

    assert fingerprint_preparation_evidence(
        first
    ) == fingerprint_preparation_evidence(second)


def test_evidence_fingerprint_changes_with_issue_evidence():
    first = build_artifact()
    second = build_artifact()
    second.issues[0].evidence = "Two missing values."

    assert fingerprint_preparation_evidence(
        first
    ) != fingerprint_preparation_evidence(second)


def test_evidence_fingerprint_changes_with_allowed_treatments():
    first = build_artifact()
    second = build_artifact()
    second.issues[0].allowed_treatments = ["retain"]

    assert fingerprint_preparation_evidence(
        first
    ) != fingerprint_preparation_evidence(second)


def test_evidence_fingerprint_changes_with_human_review_requirement():
    first = build_artifact()
    second = build_artifact()
    second.issues[0].requires_explicit_human_decision = False

    assert fingerprint_preparation_evidence(
        first
    ) != fingerprint_preparation_evidence(second)


def test_evidence_fingerprint_changes_with_artifact_context():
    first = build_artifact()
    second = build_artifact()
    second.row_count = 4

    assert fingerprint_preparation_evidence(
        first
    ) != fingerprint_preparation_evidence(second)
