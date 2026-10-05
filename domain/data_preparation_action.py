from typing import Literal

from pydantic import BaseModel, Field

DataPreparationOperation = Literal[
    "impute_missing",
    "remove_duplicates",
    "exclude_feature",
]


class DataPreparationAction(BaseModel):
    """A structured, human-approved data preparation operation."""

    operation: DataPreparationOperation = Field(
        description="The deterministic preparation operation to perform."
    )

    column: str | None = Field(
        default=None,
        description="Column affected by the operation, when applicable.",
    )

    strategy: str | None = Field(
        default=None,
        description="Strategy used by the operation, when applicable.",
    )

    reason: str = Field(
        description="Evidence-based reason for proposing this operation."
    )
