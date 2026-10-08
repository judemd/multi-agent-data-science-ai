import operator

import pandas as pd

from domain.data_validation_rule import DataValidationRule
from tools.data_quality import NULL_MARKERS


COMPARISON_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _missing_mask(series: pd.Series) -> pd.Series:
    """Identify pandas nulls and the project's canonical textual null markers."""

    mask = series.isna()

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        normalized = series.astype("string").str.strip().str.lower()
        mask = mask | normalized.isin(NULL_MARKERS)

    return mask.fillna(False)


def evaluate_data_validation_rule(
    dataframe: pd.DataFrame,
    rule: DataValidationRule,
) -> dict[str, int]:
    """Evaluate a declared rule without coercing or modifying source data."""

    if rule.column not in dataframe.columns:
        raise KeyError(f"Column '{rule.column}' does not exist.")

    series = dataframe[rule.column]
    eligible = ~_missing_mask(series)

    if rule.rule_type == "numeric_range":
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise TypeError(
                f"Rule '{rule.rule_id}' requires a numeric column: '{rule.column}'."
            )

        checked = series.loc[eligible]
        violations = pd.Series(False, index=checked.index)

        if rule.min_value is not None:
            violations = violations | (checked < rule.min_value)

        if rule.max_value is not None:
            violations = violations | (checked > rule.max_value)

    elif rule.rule_type == "allowed_values":
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            raise TypeError(
                f"Rule '{rule.rule_id}' requires a categorical/text column: "
                f"'{rule.column}'."
            )

        checked = series.loc[eligible]
        violations = ~checked.isin(rule.allowed_values)

    else:
        other_column = rule.comparison_column

        if other_column not in dataframe.columns:
            raise KeyError(f"Column '{other_column}' does not exist.")

        other = dataframe[other_column]

        both_numeric = (
            pd.api.types.is_numeric_dtype(series)
            and pd.api.types.is_numeric_dtype(other)
            and not pd.api.types.is_bool_dtype(series)
            and not pd.api.types.is_bool_dtype(other)
        )

        both_datetime = (
            pd.api.types.is_datetime64_any_dtype(series)
            and pd.api.types.is_datetime64_any_dtype(other)
        )

        if not (both_numeric or both_datetime):
            raise TypeError(
                f"Rule '{rule.rule_id}' requires compatible numeric or "
                "datetime columns."
            )

        eligible = eligible & ~_missing_mask(other)
        left = series.loc[eligible]
        right = other.loc[eligible]

        comparison = COMPARISON_OPERATORS[rule.comparison_operator]
        violations = ~comparison(left, right)
        checked = left

    return {
        "checked_count": int(len(checked)),
        "violation_count": int(violations.sum()),
    }
