from typing import Literal

from pydantic import BaseModel, Field, model_validator


DataValidationRuleType = Literal[
    "numeric_range",
    "allowed_values",
    "column_comparison",
]

DataValidationIssueType = Literal[
    "invalid_value",
    "business_rule_violation",
]


class DataValidationRule(BaseModel):
    """A declared validation constraint; approval is managed by the workflow."""

    rule_id: str = Field(min_length=1)
    rule_type: DataValidationRuleType
    issue_type: DataValidationIssueType
    column: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source: str = Field(min_length=1)

    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    comparison_column: str | None = None
    comparison_operator: Literal["<", "<=", ">", ">=", "==", "!="] | None = None

    @model_validator(mode="after")
    def validate_rule_parameters(self):
        if self.rule_type == "numeric_range":
            if self.min_value is None and self.max_value is None:
                raise ValueError("numeric_range requires at least one bound")
            if (
                self.min_value is not None
                and self.max_value is not None
                and self.min_value > self.max_value
            ):
                raise ValueError("min_value cannot exceed max_value")
            if self.allowed_values is not None or self.comparison_column is not None or self.comparison_operator is not None:
                raise ValueError("numeric_range has incompatible parameters")

        elif self.rule_type == "allowed_values":
            if not self.allowed_values:
                raise ValueError("allowed_values requires a non-empty list")
            if any(value == "" for value in self.allowed_values):
                raise ValueError("allowed_values cannot contain empty strings")
            if any(value is not None for value in (self.min_value, self.max_value, self.comparison_column, self.comparison_operator)):
                raise ValueError("allowed_values has incompatible parameters")

        elif self.rule_type == "column_comparison":
            if not self.comparison_column or not self.comparison_operator:
                raise ValueError("column_comparison requires a column and operator")
            if self.comparison_column == self.column:
                raise ValueError("column_comparison requires two distinct columns")
            if any(value is not None for value in (self.min_value, self.max_value, self.allowed_values)):
                raise ValueError("column_comparison has incompatible parameters")

        return self
