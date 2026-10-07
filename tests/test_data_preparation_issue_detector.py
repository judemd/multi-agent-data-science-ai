import pandas as pd

from tools.data_preparation_issue_detector import (
    detect_categorical_inconsistency_issues,
    detect_exact_duplicate_issues,
    detect_identifier_issues,
    detect_missing_value_issues,
    detect_numeric_conversion_issues,
)


def test_detects_numeric_missing_values_with_numeric_treatments():
    dataframe = pd.DataFrame(
        {
            "revenue": [10.0, None, 30.0],
        }
    )

    issues = detect_missing_value_issues(dataframe)

    assert len(issues) == 1

    issue = issues[0]

    assert issue.issue_type == "missing_values"
    assert issue.column == "revenue"
    assert "1 missing or null-like values" in issue.evidence
    assert issue.allowed_treatments == [
        "median",
        "mean",
        "mode",
        "drop_rows",
        "retain",
    ]


def test_detects_categorical_missing_values_with_safe_treatments():
    dataframe = pd.DataFrame(
        {
            "contract_type": ["Monthly", None, "Annual"],
        }
    )

    issues = detect_missing_value_issues(dataframe)

    assert len(issues) == 1
    assert issues[0].column == "contract_type"
    assert issues[0].allowed_treatments == [
        "mode",
        "constant",
        "drop_rows",
        "retain",
    ]


def test_ignores_columns_without_missing_values():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "revenue": [10.0, 20.0, 30.0],
        }
    )

    issues = detect_missing_value_issues(dataframe)

    assert issues == []


def test_detects_missing_values_in_multiple_columns():
    dataframe = pd.DataFrame(
        {
            "revenue": [10.0, None, 30.0],
            "contract_type": ["Monthly", None, "Annual"],
            "customer_id": [1, 2, 3],
        }
    )

    issues = detect_missing_value_issues(dataframe)

    assert len(issues) == 2
    assert [issue.column for issue in issues] == [
        "revenue",
        "contract_type",
    ]


def test_detects_supported_textual_null_markers():
    dataframe = pd.DataFrame(
        {
            "status": [
                "active",
                "N/A",
                " null ",
                "",
                "   ",
                "none",
            ],
        }
    )

    issues = detect_missing_value_issues(dataframe)

    assert len(issues) == 1
    assert issues[0].column == "status"
    assert "5 missing or null-like values" in issues[0].evidence


def test_unknown_is_not_automatically_missing():
    dataframe = pd.DataFrame(
        {
            "status": [
                "active",
                "unknown",
                "inactive",
            ],
        }
    )

    issues = detect_missing_value_issues(dataframe)

    assert issues == []


def test_detects_categorical_inconsistency_issue():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "consumer",
                " Consumer ",
                "Business",
                "business",
            ],
        }
    )

    issues = detect_categorical_inconsistency_issues(dataframe)

    assert len(issues) == 1

    issue = issues[0]

    assert issue.issue_type == "categorical_inconsistency"
    assert issue.column == "segment"
    assert issue.allowed_treatments == [
        "normalize_categories",
        "retain",
    ]
    assert "Consumer" in issue.evidence
    assert "consumer" in issue.evidence


def test_consistent_categories_do_not_create_issue():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Consumer",
                "Business",
                "Business",
            ],
        }
    )

    issues = detect_categorical_inconsistency_issues(dataframe)

    assert issues == []


def test_null_markers_are_not_categorical_inconsistencies():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "consumer",
                "N/A",
                " n/a ",
                None,
            ],
        }
    )

    issues = detect_categorical_inconsistency_issues(dataframe)

    assert len(issues) == 1
    assert issues[0].column == "segment"
    assert "consumer" in issues[0].evidence
    assert "n/a" not in issues[0].evidence.lower()


def test_numeric_columns_are_not_checked_as_categories():
    dataframe = pd.DataFrame(
        {
            "revenue": [10, 20, 30],
        }
    )

    issues = detect_categorical_inconsistency_issues(dataframe)

    assert issues == []




def test_detects_numeric_conversion_issue():
    dataframe = pd.DataFrame(
        {
            "tenure_months": [
                "12",
                "18",
                "24",
                "36",
                "unknown",
            ],
        }
    )

    issues = detect_numeric_conversion_issues(dataframe)

    assert len(issues) == 1

    issue = issues[0]

    assert issue.issue_type == "numeric_conversion"
    assert issue.column == "tenure_months"
    assert issue.allowed_treatments == [
        "convert_numeric",
        "retain",
    ]
    assert "4 of 5" in issue.evidence
    assert "80.0%" in issue.evidence


def test_normal_categorical_column_is_not_numeric_conversion_issue():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Business",
                "Consumer",
                "Business",
            ],
        }
    )

    issues = detect_numeric_conversion_issues(dataframe)

    assert issues == []


def test_native_numeric_column_is_not_numeric_conversion_issue():
    dataframe = pd.DataFrame(
        {
            "tenure_months": [12, 18, 24, 36],
        }
    )

    issues = detect_numeric_conversion_issues(dataframe)

    assert issues == []



def test_detects_exact_duplicate_issue():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C002", "C003"],
            "value": [100, 200, 200, 300],
        }
    )

    issues = detect_exact_duplicate_issues(dataframe)

    assert len(issues) == 1

    issue = issues[0]

    assert issue.issue_type == "exact_duplicates"
    assert issue.column is None
    assert issue.allowed_treatments == [
        "remove_duplicates",
        "retain",
    ]
    assert "1 exact duplicate row(s)" in issue.evidence


def test_no_exact_duplicates_produces_no_issue():
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "value": [100, 200, 300],
        }
    )

    issues = detect_exact_duplicate_issues(dataframe)

    assert issues == []



def test_detects_potential_identifier_as_explicit_human_issue():
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

    issues = detect_identifier_issues(dataframe)

    assert len(issues) == 1

    issue = issues[0]

    assert issue.issue_type == "identifier"
    assert issue.column == "customer_id"
    assert issue.allowed_treatments == [
        "exclude_feature",
        "retain",
        "investigate",
    ]
    assert issue.requires_explicit_human_decision is True


def test_identifier_issue_is_not_created_below_threshold():
    dataframe = pd.DataFrame(
        {
            "reference": [
                "A",
                "B",
                "C",
                "D",
                "D",
            ],
        }
    )

    issues = detect_identifier_issues(dataframe)

    assert issues == []


def test_numeric_high_cardinality_column_is_not_identifier_issue():
    dataframe = pd.DataFrame(
        {
            "customer_number": [
                1001,
                1002,
                1003,
                1004,
            ],
        }
    )

    issues = detect_identifier_issues(dataframe)

    assert issues == []

