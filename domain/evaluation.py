"""Structured evidence produced by deterministic model evaluation."""

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Held-out binary classification performance."""

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)


class EvaluationArtifact(BaseModel):
    """Auditable evidence for the human GO / ITERATE / NO-GO review."""

    selected_model: str = Field(min_length=1)
    target_column: str = Field(min_length=1)

    feature_columns: list[str] = Field(min_length=1)
    excluded_columns: list[str] = Field(default_factory=list)

    train_row_count: int = Field(gt=0)
    test_row_count: int = Field(gt=0)

    model_metrics: EvaluationMetrics
    baseline_metrics: EvaluationMetrics

    # Optional for compatibility with evaluation records from earlier revisions.
    selected_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    validation_f1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    threshold_metrics: EvaluationMetrics | None = None

    limitations: list[str] = Field(default_factory=list)