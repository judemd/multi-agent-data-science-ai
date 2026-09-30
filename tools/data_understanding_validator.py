from domain.data_understanding import DataUnderstandingArtifact


def validate_data_understanding(
    data: dict,
) -> DataUnderstandingArtifact:
    """Validate deterministic profiling output against its contract."""

    return DataUnderstandingArtifact.model_validate(data)
