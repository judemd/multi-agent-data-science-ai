from typing import Literal

from pydantic import BaseModel, Field


DataPreparationIssueType = Literal[
    "missing_values",
    "fake_nulls",
    "categorical_inconsistency",
    "formatting_issue",
    "numeric_conversion",
    "date_conversion",
    "outlier",
    "exact_duplicates",
    "conflicting_duplicates",
    "invalid_value",
    "business_rule_violation",
    "constant_feature",
    "near_constant_feature",
    "identifier",
    "leakage_risk",
    "sensitive_attribute",
    "proxy_risk",
    "redundant_feature",
]


DataPreparationTreatment = Literal[
    "median",
    "mean",
    "mode",
    "constant",
    "drop_rows",
    "retain",
    "normalize_nulls",
    "trim_whitespace",
    "normalize_categories",
    "convert_numeric",
    "convert_date",
    "remove_duplicates",
    "exclude_feature",
    "cap_outliers",
    "set_missing",
    "investigate",
]


class DataPreparationIssue(BaseModel):
    """A deterministic data-quality issue with controlled treatment options."""

    issue_type: DataPreparationIssueType

    column: str | None = None

    evidence: str = Field(
        min_length=1,
        description="Deterministic evidence describing the observed issue.",
    )

    allowed_treatments: list[DataPreparationTreatment] = Field(
        min_length=1,
        description="Treatments permitted for this issue.",
    )

    requires_explicit_human_decision: bool = Field(
        default=False,
        description=(
            "Whether this issue must be explicitly resolved by a human "
            "before Modeling may proceed."
        ),
    )
