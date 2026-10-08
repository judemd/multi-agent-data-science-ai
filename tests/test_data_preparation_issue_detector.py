import pandas as pd

from tools.data_preparation_issue_detector import (
    detect_categorical_inconsistency_issues,
    detect_date_conversion_issues,
    detect_exact_duplicate_issues,
    detect_formatting_issues,
    detect_feature_variability_issues,
    detect_identifier_issues,
    detect_fake_null_issues,
    detect_missing_value_issues,
    detect_numeric_conversion_issues,
    detect_outlier_issues,
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
    assert "1 genuine missing values" in issue.evidence
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

    assert detect_missing_value_issues(dataframe) == []

    issues = detect_fake_null_issues(dataframe)

    assert len(issues) == 1
    assert issues[0].issue_type == "fake_nulls"
    assert issues[0].column == "status"
    assert "5 textual null placeholders" in issues[0].evidence
    assert issues[0].allowed_treatments == [
        "normalize_nulls",
        "retain",
    ]


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




def test_detects_date_conversion_issue_with_controlled_treatments():
    dataframe = pd.DataFrame(
        {
            "signup_date": [
                "2025-01-01",
                "2025-02-01",
                "2025-03-01",
                "2025-04-01",
            ],
        }
    )

    issues = detect_date_conversion_issues(dataframe)

    assert len(issues) == 1
    assert issues[0].issue_type == "date_conversion"
    assert issues[0].column == "signup_date"
    assert issues[0].allowed_treatments == [
        "convert_date",
        "retain",
    ]


def test_date_conversion_issues_ignore_categories_and_numeric_text():
    dataframe = pd.DataFrame(
        {
            "segment": [
                "Consumer",
                "Business",
                "Consumer",
                "Business",
            ],
            "reference": [
                "1001",
                "1002",
                "1003",
                "1004",
            ],
        }
    )

    issues = detect_date_conversion_issues(dataframe)

    assert issues == []


def test_date_conversion_issue_detection_does_not_modify_dataframe():
    dataframe = pd.DataFrame(
        {
            "signup_date": [
                "2025-01-01",
                None,
                "2025-03-01",
            ],
        }
    )
    before = dataframe.copy(deep=True)

    detect_date_conversion_issues(dataframe)

    pd.testing.assert_frame_equal(dataframe, before)


def test_detect_formatting_issues_reports_whitespace():
    dataframe = pd.DataFrame(
        {
            "segment": [
                " Consumer",
                "Business ",
                "Consumer",
                "Business",
            ],
        }
    )

    issues = detect_formatting_issues(dataframe)

    assert len(issues) == 1
    assert issues[0].issue_type == "formatting_issue"
    assert issues[0].column == "segment"
    assert "2 value(s)" in issues[0].evidence
    assert issues[0].allowed_treatments == [
        "trim_whitespace",
        "retain",
    ]


def test_detect_formatting_issues_ignores_clean_and_numeric_columns():
    dataframe = pd.DataFrame(
        {
            "segment": ["Consumer", "Business", "Consumer"],
            "revenue": [100, 200, 300],
        }
    )

    assert detect_formatting_issues(dataframe) == []


def test_detect_formatting_issues_does_not_modify_dataframe():
    dataframe = pd.DataFrame(
        {
            "segment": [" Consumer ", "Business", None],
        }
    )
    original = dataframe.copy(deep=True)

    detect_formatting_issues(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_detect_outlier_issues_reports_extreme_value():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 105, 110, 115, 120, 1000],
        }
    )

    issues = detect_outlier_issues(dataframe)

    assert len(issues) == 1
    assert issues[0].issue_type == "outlier"
    assert issues[0].column == "revenue"
    assert "1 IQR outlier(s)" in issues[0].evidence
    assert "lower bound:" in issues[0].evidence
    assert "upper bound:" in issues[0].evidence
    assert issues[0].allowed_treatments == [
        "cap_outliers",
        "retain",
        "investigate",
    ]
    assert issues[0].requires_explicit_human_decision is True


def test_detect_outlier_issues_ignores_clean_null_and_text_columns():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 105, 110, 115, 120, None],
            "empty_numeric": pd.Series(
                [float("nan")] * 6,
                dtype="float64",
            ),
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

    assert detect_outlier_issues(dataframe) == []


def test_detect_outlier_issues_does_not_modify_dataframe():
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 105, 110, 115, 120, 1000],
        }
    )
    original = dataframe.copy(deep=True)

    detect_outlier_issues(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_detect_feature_variability_issues_classifies_features():
    dataframe = pd.DataFrame(
        {
            "constant_segment": ["Consumer"] * 20,
            "near_constant_status": ["Active"] * 19 + ["Inactive"],
        }
    )

    issues = detect_feature_variability_issues(dataframe)

    assert len(issues) == 2

    by_column = {issue.column: issue for issue in issues}

    constant = by_column["constant_segment"]
    assert constant.issue_type == "constant_feature"
    assert "1 distinct non-null value(s)" in constant.evidence
    assert "100.00%" in constant.evidence
    assert constant.allowed_treatments == [
        "exclude_feature",
        "retain",
    ]
    assert constant.requires_explicit_human_decision is True

    near_constant = by_column["near_constant_status"]
    assert near_constant.issue_type == "near_constant_feature"
    assert "2 distinct non-null value(s)" in near_constant.evidence
    assert "95.00%" in near_constant.evidence
    assert near_constant.allowed_treatments == [
        "exclude_feature",
        "retain",
        "investigate",
    ]
    assert near_constant.requires_explicit_human_decision is True


def test_detect_feature_variability_issues_ignores_missing_and_varying():
    dataframe = pd.DataFrame(
        {
            "entirely_missing": [None] * 20,
            "varying_region": ["North"] * 10 + ["South"] * 10,
        }
    )

    assert detect_feature_variability_issues(dataframe) == []


def test_detect_feature_variability_issues_preserves_dataframe():
    dataframe = pd.DataFrame(
        {
            "status": ["Active"] * 19 + ["Inactive"],
        }
    )
    original = dataframe.copy(deep=True)

    detect_feature_variability_issues(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)

def test_genuine_missing_and_fake_nulls_are_separate_issues():
    dataframe = pd.DataFrame(
        {
            "status": ["Active", None, " N/A ", "unknown", "null"],
        }
    )

    missing_issues = detect_missing_value_issues(dataframe)
    fake_null_issues = detect_fake_null_issues(dataframe)

    assert len(missing_issues) == 1
    assert "1 genuine missing values" in missing_issues[0].evidence

    assert len(fake_null_issues) == 1
    assert "2 textual null placeholders" in fake_null_issues[0].evidence
    assert fake_null_issues[0].requires_explicit_human_decision
