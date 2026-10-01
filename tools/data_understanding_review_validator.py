from domain.data_understanding_review import DataUnderstandingReview


def validate_data_understanding_review(
    data: dict,
) -> DataUnderstandingReview:
    """Validate an agent's data-understanding review."""

    return DataUnderstandingReview.model_validate(data)
