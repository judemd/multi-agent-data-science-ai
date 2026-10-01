from pydantic import BaseModel, Field


class DataUnderstandingArtifact(BaseModel):
    """Structured evidence produced by the data understanding stage.

    The artifact combines deterministic data-quality and exploratory-analysis
    results so downstream agents can interpret evidence without calculating
    statistics themselves.
    """

    file_name: str = Field(
        min_length=1,
        description="Name of the dataset file that was analyzed.",
    )

    row_count: int = Field(
        ge=0,
        description="Number of rows in the analyzed dataset.",
    )

    column_count: int = Field(
        ge=0,
        description="Number of columns in the analyzed dataset.",
    )

    columns: list[str] = Field(
        default_factory=list,
        description="Column names present in the dataset.",
    )

    duplicate_row_count: int = Field(
        ge=0,
        description="Number of duplicate rows identified in the dataset.",
    )

    duplicate_value_summary: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Repeated non-null values identified in candidate columns.",
    )

    missing_value_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Columns containing missing or null-like values and their counts.",
    )

    datatype_summary: dict[str, str] = Field(
        default_factory=dict,
        description="Detected pandas datatype for each column.",
    )

    numeric_columns: list[str] = Field(
        default_factory=list,
        description="Columns identified as numeric by pandas.",
    )

    categorical_columns: list[str] = Field(
        default_factory=list,
        description="Columns identified as categorical or string-based by pandas.",
    )

    numeric_like_columns: dict[str, dict[str, float | int]] = Field(
        default_factory=dict,
        description="String columns containing predominantly numeric values.",
    )

    categorical_inconsistencies: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description="Case or whitespace variants found within categorical columns.",
    )

    formatting_issues: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="String formatting issues identified by column.",
    )

    outlier_summary: dict[str, dict[str, float | int]] = Field(
        default_factory=dict,
        description="IQR-based outlier statistics for numeric columns.",
    )

    potential_issues: list[str] = Field(
        default_factory=list,
        description="Human-readable summaries of material data-quality findings.",
    )

    analyst_questions: list[str] = Field(
        default_factory=list,
        description="Questions requiring human or business clarification.",
    )

    numeric_summary: dict[str, dict[str, float | int]] = Field(
        default_factory=dict,
        description="Descriptive statistics for numeric columns.",
    )

    categorical_summary: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description="Frequency summaries for categorical columns.",
    )

    numeric_correlations: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Pearson correlations between numeric columns.",
    )

    numeric_target_relationships: dict[str, float] = Field(
        default_factory=dict,
        description="Correlations between numeric features and the target.",
    )

    categorical_target_relationships: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description="Target rates across categorical feature values.",
    )
