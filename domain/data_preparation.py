from pydantic import BaseModel, Field


class DataPreparationArtifact(BaseModel):
    """Structured evidence describing preparation considerations.

    This artifact contains deterministic evidence only. It does not
    modify, clean, delete, impute, encode, or otherwise transform data.
    """

    file_name: str = Field(
        min_length=1,
        description="Name of the dataset being considered for preparation.",
    )

    row_count: int = Field(
        ge=0,
        description="Number of rows in the dataset.",
    )

    column_count: int = Field(
        ge=0,
        description="Number of columns in the dataset.",
    )

    columns: list[str] = Field(
        default_factory=list,
        description="Column names present in the dataset.",
    )

    datatype_summary: dict[str, str] = Field(
        default_factory=dict,
        description="Observed datatype for each column.",
    )

    missing_value_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Observed missing or null-like values by column.",
    )

    numeric_like_columns: dict[str, dict[str, float | int]] = Field(
        default_factory=dict,
        description="Text columns containing predominantly numeric values.",
    )

    categorical_inconsistencies: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description="Observed categorical case or whitespace inconsistencies.",
    )

    formatting_issues: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Observed string formatting issues by column.",
    )

    outlier_summary: dict[str, dict[str, float | int]] = Field(
        default_factory=dict,
        description="Observed IQR-based outlier evidence by numeric column.",
    )

    constant_columns: list[str] = Field(
        default_factory=list,
        description="Columns observed to contain a single distinct non-null value.",
    )

    candidate_date_columns: list[str] = Field(
        default_factory=list,
        description="Columns whose observed values appear suitable for date-type review.",
    )

    potential_identifier_columns: list[str] = Field(
        default_factory=list,
        description="Columns that may represent identifiers and require human validation.",
    )

    duplicate_row_count: int = Field(
        default=0,
        ge=0,
        description="Observed exact duplicate row count.",
    )

    preparation_questions: list[str] = Field(
        default_factory=list,
        description="Questions requiring human or business clarification before preparation.",
    )
