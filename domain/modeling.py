from pydantic import BaseModel, Field


class ModelingArtifact(BaseModel):
    """Structured deterministic evidence for model-selection planning."""

    file_name: str = Field(
        min_length=1,
        description="Name of the dataset being considered for modeling.",
    )

    row_count: int = Field(
        ge=0,
        description="Number of rows in the dataset.",
    )

    column_count: int = Field(
        ge=0,
        description="Number of columns in the dataset.",
    )

    target_column: str = Field(
        min_length=1,
        description="Human-selected target column for modeling.",
    )

    target_dtype: str = Field(
        min_length=1,
        description="Observed datatype of the selected target column.",
    )

    target_missing_count: int = Field(
        ge=0,
        description="Number of missing values in the target column.",
    )

    target_unique_count: int = Field(
        ge=0,
        description="Number of distinct non-null target values.",
    )

    target_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Observed frequency of each target value.",
    )

    feature_columns: list[str] = Field(
        default_factory=list,
        description="Columns available as candidate modeling features.",
    )

    numeric_features: list[str] = Field(
        default_factory=list,
        description="Candidate numeric feature columns.",
    )

    categorical_features: list[str] = Field(
        default_factory=list,
        description="Candidate categorical or string feature columns.",
    )

    constant_features: list[str] = Field(
        default_factory=list,
        description="Candidate features observed to contain a single value.",
    )

    potential_identifier_features: list[str] = Field(
        default_factory=list,
        description="Candidate feature columns that may represent identifiers.",
    )

    numeric_target_relationships: dict[str, float] = Field(
        default_factory=dict,
        description="Observed numeric feature relationships with the target.",
    )

    categorical_target_relationships: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description="Observed categorical feature relationships with the target.",
    )

    missing_value_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Observed missing-value counts for candidate features.",
    )

    candidate_date_columns: list[str] = Field(
        default_factory=list,
        description="Candidate feature columns that may represent dates.",
    )

    potential_data_leakage_columns: list[str] = Field(
        default_factory=list,
        description="Columns requiring investigation for potential target leakage.",
    )

    modeling_questions: list[str] = Field(
        default_factory=list,
        description="Questions requiring human or business clarification before model selection.",
    )
