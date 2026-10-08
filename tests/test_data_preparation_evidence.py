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



def test_evidence_includes_typed_date_conversion_issue():
    dataframe = pd.DataFrame(
        {
            "signup_date": [
                "2025-01-01",
                "2025-02-01",
                "2025-03-01",
                "2025-04-01",
            ],
            "segment": [
                "Consumer",
                "Business",
                "Consumer",
                "Business",
            ],
        }
    )

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    date_issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "date_conversion"
    ]

    assert len(date_issues) == 1
    assert date_issues[0].column == "signup_date"
    assert date_issues[0].allowed_treatments == [
        "convert_date",
        "retain",
    ]


def test_evidence_includes_typed_formatting_issue():
    dataframe = pd.DataFrame(
        {
            "segment": [
                " Consumer",
                "Business ",
                "Consumer",
                "Business",
            ],
            "revenue": [100, 200, 300, 400],
        }
    )
    original = dataframe.copy(deep=True)

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    formatting_issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "formatting_issue"
    ]

    assert len(formatting_issues) == 1
    assert formatting_issues[0].column == "segment"
    assert "2 value(s)" in formatting_issues[0].evidence
    assert formatting_issues[0].allowed_treatments == [
        "trim_whitespace",
        "retain",
    ]

    pd.testing.assert_frame_equal(dataframe, original)


def test_evidence_includes_typed_outlier_issue():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 105, 110, 115, 120, 1000],
            "segment": [
                "Consumer",
                "Business",
                "Consumer",
                "Business",
                "Consumer",
                "Business",
            ],
        }
    )
    original = dataframe.copy(deep=True)

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    outlier_issues = [
        issue
        for issue in artifact.issues
        if issue.issue_type == "outlier"
    ]

    assert len(outlier_issues) == 1
    assert outlier_issues[0].column == "revenue"
    assert "1 IQR outlier(s)" in outlier_issues[0].evidence
    assert "lower bound:" in outlier_issues[0].evidence
    assert "upper bound:" in outlier_issues[0].evidence
    assert outlier_issues[0].allowed_treatments == [
        "cap_outliers",
        "retain",
        "investigate",
    ]
    assert outlier_issues[0].requires_explicit_human_decision is True

    pd.testing.assert_frame_equal(dataframe, original)


def test_legacy_constant_columns_include_all_missing_columns():
    dataframe = pd.DataFrame(
        {
            "constant_segment": ["Consumer"] * 4,
            "entirely_missing": [None] * 4,
            "varying_revenue": [100, 200, 300, 400],
        }
    )

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    assert set(artifact.constant_columns) == {
        "constant_segment",
        "entirely_missing",
    }


def test_evidence_includes_typed_feature_variability_issues():
    dataframe = pd.DataFrame(
        {
            "constant_segment": ["Consumer"] * 20,
            "near_constant_status": ["Active"] * 19 + ["Inactive"],
            "entirely_missing": [None] * 20,
            "varying_region": ["North"] * 10 + ["South"] * 10,
        }
    )
    original = dataframe.copy(deep=True)

    artifact = build_data_preparation_evidence(
        dataframe,
        "customers.csv",
    )

    variability_issues = {
        issue.column: issue
        for issue in artifact.issues
        if issue.issue_type in {
            "constant_feature",
            "near_constant_feature",
        }
    }

    assert set(variability_issues) == {
        "constant_segment",
        "near_constant_status",
    }

    constant = variability_issues["constant_segment"]
    assert constant.issue_type == "constant_feature"
    assert constant.allowed_treatments == [
        "exclude_feature",
        "retain",
    ]
    assert constant.requires_explicit_human_decision is True

    near_constant = variability_issues["near_constant_status"]
    assert near_constant.issue_type == "near_constant_feature"
    assert "95.00%" in near_constant.evidence
    assert near_constant.allowed_treatments == [
        "exclude_feature",
        "retain",
        "investigate",
    ]
    assert near_constant.requires_explicit_human_decision is True

    assert set(artifact.constant_columns) == {
        "constant_segment",
        "entirely_missing",
    }

    pd.testing.assert_frame_equal(dataframe, original)
