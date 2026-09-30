from pydantic import BaseModel, Field


class DataUnderstandingArtifact(BaseModel):
    """Structured output from the data understanding stage."""

    file_name: str = Field(min_length=1)

    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)

    columns: list[str] = Field(default_factory=list)

    duplicate_row_count: int = Field(ge=0)

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
    )

    categorical_columns: list[str] = Field(
        default_factory=list,
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
