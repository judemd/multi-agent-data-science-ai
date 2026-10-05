from __future__ import annotations

import pandas as pd

from domain.data_preparation_action import DataPreparationAction


def execute_data_preparation_actions(
    dataframe: pd.DataFrame,
    actions: list[DataPreparationAction],
) -> pd.DataFrame:
    """Apply approved deterministic data-preparation actions."""

    prepared = dataframe.copy()

    for action in actions:
        if action.operation == "remove_duplicates":
            prepared = prepared.drop_duplicates().reset_index(drop=True)

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

            if action.strategy == "median":
                if not pd.api.types.is_numeric_dtype(
                    prepared[action.column]
                ):
                    raise ValueError(
                        f"Median imputation requires a numeric column: "
                        f"'{action.column}'."
                    )

                median = prepared[action.column].median()

                if pd.isna(median):
                    raise ValueError(
                        f"Cannot calculate median for column "
                        f"'{action.column}'."
                    )

                prepared[action.column] = prepared[
                    action.column
                ].fillna(median)

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