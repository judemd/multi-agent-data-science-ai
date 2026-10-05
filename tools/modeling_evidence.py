import pandas as pd

from domain.modeling import ModelingArtifact

MAX_CATEGORICAL_LEVELS_FOR_TARGET_RELATIONSHIP = 50
MAX_TARGET_VALUES_IN_CATEGORY_SUMMARY = 20


def build_modeling_evidence(
    dataframe: pd.DataFrame,
    target_column: str,
    file_name: str,
) -> ModelingArtifact:
    """Build deterministic modeling evidence without modifying the DataFrame.

    Evidence sent to downstream agents is deliberately bounded. In particular,
    category-level target relationships are only calculated for categorical
    features with a manageable number of distinct values. This prevents
    high-cardinality columns such as identifiers from creating enormous
    downstream prompts.
    """

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    target = dataframe[target_column]

    target_missing_count = int(target.isna().sum())
    target_unique_count = int(target.nunique(dropna=True))

    target_distribution = {
        str(value): int(count)
        for value, count in target.dropna().value_counts().items()
    }

    feature_columns = [
        column
        for column in dataframe.columns
        if column != target_column
    ]

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    categorical_features = [
        column
        for column in feature_columns
        if (
            pd.api.types.is_string_dtype(dataframe[column])
            or pd.api.types.is_categorical_dtype(dataframe[column])
            or pd.api.types.is_object_dtype(dataframe[column])
        )
    ]

    constant_features = [
        column
        for column in feature_columns
        if dataframe[column].nunique(dropna=True) <= 1
    ]

    potential_identifier_features: list[str] = []

    for column in feature_columns:
        series = dataframe[column]
        non_null_count = int(series.notna().sum())

        if non_null_count == 0:
            continue

        unique_count = int(series.nunique(dropna=True))

        if unique_count / non_null_count >= 0.95:
            potential_identifier_features.append(column)

    missing_value_summary = {
        column: int(dataframe[column].isna().sum())
        for column in feature_columns
        if int(dataframe[column].isna().sum()) > 0
    }

    candidate_date_columns: list[str] = []

    for column in feature_columns:
        series = dataframe[column]

        if not pd.api.types.is_string_dtype(series):
            continue

        non_null = series.dropna().astype(str)

        if non_null.empty:
            continue

        date_ratio = float(
            pd.to_datetime(
                non_null,
                errors="coerce",
                format="mixed",
            ).notna().mean()
        )

        if date_ratio >= 0.8:
            candidate_date_columns.append(column)

    numeric_target_relationships: dict[str, float] = {}

    if (
        pd.api.types.is_numeric_dtype(target)
        and target_unique_count > 1
    ):
        for column in numeric_features:
            relationship = dataframe[column].corr(target)

            if pd.notna(relationship):
                numeric_target_relationships[column] = float(
                    relationship
                )

    categorical_target_relationships: dict[str, dict[str, object]] = {}

    if target_unique_count > 1:
        for column in categorical_features:
            category_count = int(
                dataframe[column].nunique(dropna=True)
            )

            if category_count > MAX_CATEGORICAL_LEVELS_FOR_TARGET_RELATIONSHIP:
                continue

            grouped = dataframe.groupby(
                column,
                dropna=False,
            )[target_column]

            category_summary: dict[str, object] = {}

            for value, values in grouped:
                key = str(value)

                if pd.api.types.is_numeric_dtype(target):
                    category_summary[key] = {
                        "count": int(values.count()),
                        "target_mean": (
                            float(values.mean())
                            if values.notna().any()
                            else None
                        ),
                    }
                else:
                    counts = values.value_counts(
                        dropna=True,
                    ).head(
                        MAX_TARGET_VALUES_IN_CATEGORY_SUMMARY
                    )

                    category_summary[key] = {
                        "count": int(values.count()),
                        "target_distribution": {
                            str(target_value): int(count)
                            for target_value, count in counts.items()
                        },
                    }

            categorical_target_relationships[column] = category_summary

    potential_data_leakage_columns: list[str] = []

    target_name_normalized = target_column.strip().casefold()

    for column in feature_columns:
        normalized = column.strip().casefold()

        if normalized == target_name_normalized:
            continue

        leakage_terms = (
            "target",
            "outcome",
            "label",
            "response",
            "prediction",
            "predicted",
            "actual",
            "result",
            "future",
        )

        if any(term in normalized for term in leakage_terms):
            potential_data_leakage_columns.append(column)

    modeling_questions: list[str] = []

    if target_missing_count:
        modeling_questions.append(
            "How should observations with missing target values be handled?"
        )

    if target_unique_count <= 1:
        modeling_questions.append(
            "Does the selected target contain enough distinct values "
            "to support a useful predictive model?"
        )

    if missing_value_summary:
        modeling_questions.append(
            "How should missing feature values be handled before modeling?"
        )

    if constant_features:
        modeling_questions.append(
            "Should constant features be excluded from modeling?"
        )

    if potential_identifier_features:
        modeling_questions.append(
            "Which high-cardinality features are genuine business identifiers "
            "and should therefore be excluded?"
        )

    if candidate_date_columns:
        modeling_questions.append(
            "How should candidate date features be represented for modeling?"
        )

    if potential_data_leakage_columns:
        modeling_questions.append(
            "Could any flagged features contain information unavailable "
            "at prediction time?"
        )

    return ModelingArtifact(
        file_name=file_name,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        target_column=target_column,
        target_dtype=str(target.dtype),
        target_missing_count=target_missing_count,
        target_unique_count=target_unique_count,
        target_distribution=target_distribution,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        constant_features=constant_features,
        potential_identifier_features=potential_identifier_features,
        numeric_target_relationships=numeric_target_relationships,
        categorical_target_relationships=categorical_target_relationships,
        missing_value_summary=missing_value_summary,
        candidate_date_columns=candidate_date_columns,
        potential_data_leakage_columns=potential_data_leakage_columns,
        modeling_questions=modeling_questions,
    )
