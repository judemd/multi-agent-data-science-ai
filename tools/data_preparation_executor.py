from __future__ import annotations

import pandas as pd

from domain.data_preparation_action import DataPreparationAction
from tools.data_quality import NULL_MARKERS


def execute_data_preparation_actions(
    dataframe: pd.DataFrame,
    actions: list[DataPreparationAction],
) -> pd.DataFrame:
    """Apply approved deterministic data-preparation actions."""

    prepared = dataframe.copy()

    for action in actions:
        if action.operation == "remove_duplicates":
            prepared = prepared.drop_duplicates().reset_index(drop=True)

        elif action.operation == "drop_rows":
            if not action.column:
                raise ValueError(
                    "drop_rows requires a column."
                )

            if action.column not in prepared.columns:
                raise ValueError(
                    f"Cannot drop rows for unknown column "
                    f"'{action.column}'."
                )

            prepared = prepared.loc[
                prepared[action.column].notna()
            ].reset_index(drop=True)

        elif action.operation == "exclude_feature":
            if not action.column:
                raise ValueError(
                    "exclude_feature requires a column."
                )

            if action.column not in prepared.columns:
                raise ValueError(
                    f"Cannot exclude missing column '{action.column}'."
                )

            prepared = prepared.drop(columns=[action.column])

        elif action.operation == "trim_whitespace":
            if not action.column:
                raise ValueError(
                    "trim_whitespace requires a column."
                )

            if action.column not in prepared.columns:
                raise ValueError(
                    f"Cannot trim whitespace in unknown column "
                    f"'{action.column}'."
                )

            prepared[action.column] = prepared[action.column].map(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )

        elif action.operation == "normalize_nulls":
            if not action.column:
                raise ValueError(
                    "normalize_nulls requires a column."
                )

            if action.column not in prepared.columns:
                raise ValueError(
                    f"Cannot normalize nulls in unknown column "
                    f"'{action.column}'."
                )

            series = prepared[action.column]

            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                normalized = series.astype("string").str.strip().str.lower()
                fake_null_mask = (
                    series.notna() & normalized.isin(NULL_MARKERS)
                ).fillna(False)

                prepared.loc[fake_null_mask, action.column] = pd.NA

        elif action.operation == "normalize_categories":
            if not action.column:
                raise ValueError(
                    "normalize_categories requires a column."
                )

            if action.column not in prepared.columns:
                raise ValueError(
                    f"Cannot normalize categories in unknown column "
                    f"'{action.column}'."
                )

            series = prepared[action.column]

            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                raise ValueError(
                    "normalize_categories requires a text column."
                )

            groups: dict[str, dict[str, int]] = {}

            for value in series:
                if not isinstance(value, str):
                    continue

                normalized = value.strip().lower()

                if normalized in NULL_MARKERS:
                    continue

                frequencies = groups.setdefault(normalized, {})
                frequencies[value] = frequencies.get(value, 0) + 1

            canonical_values: dict[str, str] = {}

            for normalized, frequencies in groups.items():
                if len(frequencies) < 2:
                    continue

                # Python dictionaries preserve insertion order.
                # max therefore selects the first observed spelling
                # when multiple spellings have equal frequencies.
                representative = max(
                    frequencies,
                    key=frequencies.get,
                )
                canonical_values[normalized] = representative.strip()

            prepared[action.column] = series.map(
                lambda value: canonical_values.get(
                    value.strip().lower(),
                    value,
                )
                if isinstance(value, str)
                and value.strip().lower() not in NULL_MARKERS
                else value
            )

        elif action.operation == "impute_missing":
            if not action.column:
                raise ValueError(
                    "impute_missing requires a column."
                )

            if action.column not in prepared.columns:
                raise ValueError(
                    f"Cannot impute missing values in "
                    f"unknown column '{action.column}'."
                )

            if action.strategy in ("median", "mean"):
                if not pd.api.types.is_numeric_dtype(
                    prepared[action.column]
                ):
                    raise ValueError(
                        f"{action.strategy.capitalize()} imputation "
                        f"requires a numeric column: "
                        f"'{action.column}'."
                    )

                statistic = (
                    prepared[action.column].median()
                    if action.strategy == "median"
                    else prepared[action.column].mean()
                )

                if pd.isna(statistic):
                    raise ValueError(
                        f"Cannot calculate {action.strategy} for column "
                        f"'{action.column}'."
                    )

                prepared[action.column] = prepared[
                    action.column
                ].fillna(statistic)

            elif action.strategy == "mode":
                non_missing = prepared[action.column].dropna()

                if non_missing.empty:
                    raise ValueError(
                        f"Cannot calculate mode for column "
                        f"'{action.column}'."
                    )

                frequencies = non_missing.value_counts(
                    dropna=True,
                    sort=False,
                )
                highest_frequency = frequencies.max()

                for value in non_missing:
                    if frequencies.loc[value] == highest_frequency:
                        selected_mode = value
                        break

                prepared[action.column] = prepared[
                    action.column
                ].fillna(selected_mode)

            else:
                raise ValueError(
                    f"Unsupported imputation strategy: "
                    f"{action.strategy!r}"
                )

        else:
            raise ValueError(
                f"Unsupported data preparation operation: "
                f"{action.operation!r}"
            )

    return prepared