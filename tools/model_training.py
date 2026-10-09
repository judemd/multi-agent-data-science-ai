"""Deterministic, human-controlled model training utilities."""

from dataclasses import dataclass
from sklearn.base import clone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tools.data_quality import NULL_MARKERS
from tools.modeling_sanity_check import check_modeling_dataset


SUPPORTED_TRAINING_MODELS = frozenset({
    "Logistic Regression",
    "Random Forest",
})


def validate_training_features(
    dataframe: pd.DataFrame,
    target_column: str,
    selected_model: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Validate the approved feature set without modifying the source data."""

    if selected_model not in SUPPORTED_TRAINING_MODELS:
        raise ValueError(f"Unsupported training model: {selected_model}")

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    if not feature_columns:
        raise ValueError("At least one approved feature column is required.")

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Approved feature columns contain duplicates.")

    if target_column in feature_columns:
        raise ValueError("Target column cannot be included as a feature.")

    unknown_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if unknown_columns:
        raise ValueError(
            f"Unknown approved feature columns: {unknown_columns}"
        )

    selected_data = dataframe.loc[
        :,
        [*feature_columns, target_column],
    ].copy()

    sanity = check_modeling_dataset(
        selected_data,
        target_column,
    )

    if sanity.status == "BLOCK":
        raise ValueError(
            "Modeling sanity check failed: "
            + "; ".join(sanity.errors)
        )

    if set(selected_data[target_column].unique()) != {0, 1}:
        raise ValueError(
            "Training requires binary target labels 0 and 1."
        )

    return selected_data


@dataclass
class TrainingResult:
    """Fitted pipelines and held-out data for deterministic evaluation."""

    selected_model: str
    model_pipeline: Pipeline
    baseline_pipeline: Pipeline
    x_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    train_row_count: int
    test_row_count: int
    threshold_selection: "ThresholdSelectionResult"
    validation_row_count: int
    validation_indices: list[int]


def train_selected_model(
    dataframe: pd.DataFrame,
    target_column: str,
    selected_model: str,
    feature_columns: list[str],
) -> TrainingResult:
    """Fit the approved classifier and baseline on the same data split."""

    selected_data = validate_training_features(
        dataframe,
        target_column,
        selected_model,
        feature_columns,
    )

    x = selected_data[feature_columns].copy()
    y = selected_data[target_column].copy()

    numeric_columns = x.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in feature_columns
        if column not in numeric_columns
    ]

    # Interpret known textual null markers as missing only in model inputs.
    for column in categorical_columns:
        x[column] = x[column].map(
            lambda value: (
                np.nan
                if isinstance(value, str)
                and value.strip().lower() in NULL_MARKERS
                else value
            )
        ).astype(object)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    if selected_model == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    baseline_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DummyClassifier(
            strategy="most_frequent",
        )),
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    x_inner, x_validation, y_inner, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=42,
    )

    validation_pipeline = clone(model_pipeline)
    validation_pipeline.fit(x_inner, y_inner)

    validation_classes = list(
        validation_pipeline.named_steps["classifier"].classes_
    )
    positive_index = validation_classes.index(1)
    validation_probabilities = validation_pipeline.predict_proba(
        x_validation
    )[:, positive_index]

    threshold_selection = select_classification_threshold(
        y_validation,
        validation_probabilities,
    )

    model_pipeline.fit(x_train, y_train)
    baseline_pipeline.fit(x_train, y_train)

    return TrainingResult(
        selected_model=selected_model,
        model_pipeline=model_pipeline,
        baseline_pipeline=baseline_pipeline,
        x_test=x_test,
        y_test=y_test,
        feature_columns=list(feature_columns),
        train_row_count=len(x_train),
        test_row_count=len(x_test),
        threshold_selection=threshold_selection,
        validation_row_count=len(x_validation),
        validation_indices=x_validation.index.tolist(),
    )


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Threshold chosen using validation data only."""

    threshold: float
    validation_f1: float


def select_classification_threshold(
    labels: pd.Series,
    probabilities: list[float] | np.ndarray,
) -> ThresholdSelectionResult:
    """Maximize validation F1, breaking ties toward higher thresholds."""

    y_true = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("Labels and probabilities must have equal nonzero length.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Threshold selection requires binary labels 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")

    best_threshold = None
    best_f1 = -1.0

    for threshold in np.unique(scores):
        predictions = (scores >= threshold).astype(int)
        score = float(f1_score(y_true, predictions, zero_division=0))

        if score > best_f1 or (
            score == best_f1
            and (best_threshold is None or threshold > best_threshold)
        ):
            best_threshold = float(threshold)
            best_f1 = score

    return ThresholdSelectionResult(
        threshold=best_threshold,
        validation_f1=best_f1,
    )


@dataclass
class ClassificationMetrics:
    """Held-out binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


@dataclass
class ModelEvaluationResult:
    """Comparison of the approved model against the dummy baseline."""

    selected_model: str
    model_metrics: ClassificationMetrics
    baseline_metrics: ClassificationMetrics
    train_row_count: int
    test_row_count: int
    feature_columns: list[str]
    selected_threshold: float
    validation_f1: float
    threshold_metrics: ClassificationMetrics


def evaluate_training_result(
    training_result: TrainingResult,
) -> ModelEvaluationResult:
    """Evaluate fitted pipelines on their shared held-out test set."""

    y_test = training_result.y_test

    if y_test.nunique() != 2:
        raise ValueError(
            "Held-out evaluation requires exactly two target classes."
        )

    # The POC uses a binary target encoded as 0 and 1.
    if set(y_test.unique()) != {0, 1}:
        raise ValueError(
            "Evaluation requires binary target labels 0 and 1."
        )

    def calculate_metrics(
        pipeline: Pipeline,
        threshold: float | None = None,
    ) -> ClassificationMetrics:
        probabilities = pipeline.predict_proba(
            training_result.x_test
        )

        classes = list(
            pipeline.named_steps["classifier"].classes_
        )
        positive_index = classes.index(1)
        positive_scores = probabilities[:, positive_index]
        predictions = (
            pipeline.predict(training_result.x_test)
            if threshold is None
            else (positive_scores >= threshold).astype(int)
        )

        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(precision_score(
                y_test,
                predictions,
                zero_division=0,
            )),
            recall=float(recall_score(
                y_test,
                predictions,
                zero_division=0,
            )),
            f1=float(f1_score(
                y_test,
                predictions,
                zero_division=0,
            )),
            roc_auc=float(roc_auc_score(
                y_test,
                positive_scores,
            )),
        )

    return ModelEvaluationResult(
        selected_model=training_result.selected_model,
        model_metrics=calculate_metrics(
            training_result.model_pipeline
        ),
        baseline_metrics=calculate_metrics(
            training_result.baseline_pipeline
        ),
        selected_threshold=training_result.threshold_selection.threshold,
        validation_f1=training_result.threshold_selection.validation_f1,
        threshold_metrics=calculate_metrics(
            training_result.model_pipeline,
            threshold=training_result.threshold_selection.threshold,
        ),
        train_row_count=training_result.train_row_count,
        test_row_count=training_result.test_row_count,
        feature_columns=list(training_result.feature_columns),
    )