import pandas as pd

from domain.data_preparation import DataPreparationArtifact
from tools.data_preparation_evidence import (
    build_data_preparation_evidence,
)


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "revenue": [
                100.0,
                None,
                300.0,
                400.0,
            ],
            "segment": [
                "Enterprise",
                "enterprise ",
                "SMB",
                "SMB",
            ],
            "signup_date": [
                "2025-01-01",
                "2025-02-01",
                "2025-03-01",
                "2025-04-01",
            ],
        }
    )


def test_build_data_preparation_evidence_returns_artifact():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert isinstance(result, DataPreparationArtifact)
    assert result.file_name == "customers.csv"
    assert result.row_count == 4
    assert result.column_count == 4


def test_evidence_identifies_missing_values():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert result.missing_value_summary == {
        "revenue": 1,
    }


def test_evidence_identifies_categorical_inconsistency():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert "segment" in result.categorical_inconsistencies


def test_evidence_identifies_candidate_date_column():
    result = build_data_preparation_evidence(
        build_dataframe(),
        "customers.csv",
    )

    assert "signup_date" in result.candidate_date_columns


def test_evidence_does_not_modify_dataframe():
    dataframe = build_dataframe()
    before = dataframe.copy(deep=True)

    build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    pd.testing.assert_frame_equal(dataframe, before)


def test_evidence_includes_typed_missing_value_issues():
    dataframe = pd.DataFrame(
        {
            "revenue": [10.0, None, 30.0],
            "contract_type": ["Monthly", None, "Annual"],
            "customer_id": [1, 2, 3],
        }
    )

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    assert len(artifact.issues) == 2

    revenue_issue = next(
        issue
        for issue in artifact.issues
        if issue.column == "revenue"
    )

    assert revenue_issue.issue_type == "missing_values"
    assert revenue_issue.allowed_treatments == [
        "median",
        "mean",
        "mode",
        "drop_rows",
        "retain",
    ]

    contract_issue = next(
        issue
        for issue in artifact.issues
        if issue.column == "contract_type"
    )

    assert contract_issue.issue_type == "missing_values"
    assert contract_issue.allowed_treatments == [
        "mode",
        "constant",
        "drop_rows",
        "retain",
    ]


def test_evidence_includes_typed_categorical_inconsistency_issue():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "consumer",
                "Business",
            ],
            "customer_id": [1, 2, 3],
        }
    )

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    categorical_issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "categorical_inconsistency"
    ]

    assert len(categorical_issues) == 1

    issue = categorical_issues[0]

    assert issue.column == "segment"
    assert issue.allowed_treatments == [
        "normalize_categories",
        "retain",
    ]


def test_evidence_includes_typed_numeric_conversion_issue():
    dataframe = pd.DataFrame(
        {
            "tenure_months": [
                "12",
                "18",
                "24",
                "36",
                "unknown",
            ],
            "customer_id": [1, 2, 3, 4, 5],
        }
    )

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    numeric_issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "numeric_conversion"
    ]

    assert len(numeric_issues) == 1

    issue = numeric_issues[0]

    assert issue.column == "tenure_months"
    assert issue.allowed_treatments == [
        "convert_numeric",
        "retain",
    ]
    assert "4 of 5" in issue.evidence


def test_evidence_includes_typed_exact_duplicate_issue():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C002", "C003"],
            "value": [100, 200, 200, 300],
        }
    )

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    duplicate_issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "exact_duplicates"
    ]

    assert len(duplicate_issues) == 1

    issue = duplicate_issues[0]

    assert issue.column is None
    assert issue.allowed_treatments == [
        "remove_duplicates",
        "retain",
    ]
    assert "1 exact duplicate row(s)" in issue.evidence


def test_build_data_preparation_evidence_includes_identifier_issue():
    dataframe = pd.DataFrame(
        {
            "customer_id": [
                f"C{i:03d}"
                for i in range(1, 21)
            ],
            "segment": [
                "Consumer",
                "Business",
            ] * 10,
        }
    )

    artifact = build_data_preparation_evidence(
        dataframe=dataframe,
        file_name="customers.csv",
    )

    identifier_issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "identifier"
    ]

    assert len(identifier_issues) == 1
    assert identifier_issues[0].column == "customer_id"
    assert identifier_issues[0].requires_explicit_human_decision is True
    assert identifier_issues[0].allowed_treatments == [
        "exclude_feature",
        "retain",
        "investigate",
    ]

