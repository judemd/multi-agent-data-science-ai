import pytest

from agents.data_understanding_runner import (
    _extract_text,
    _parse_review,
)


class FakePart:
    """Represent a minimal ADK response part for testing."""

    def __init__(self, text: str | None = None):
        self.text = text


class FakeContent:
    """Represent a minimal ADK response content object."""

    def __init__(self, parts):
        self.parts = parts


class FakeEvent:
    """Represent a minimal ADK event for testing."""

    def __init__(self, content):
        self.content = content


def valid_review_json() -> str:
    """Return a valid DataUnderstandingReview payload."""

    return """
    {
        "observed_evidence": [
            "The dataset contains 100 rows."
        ],
        "interpretation": [
            "The dataset is sufficiently populated for initial review."
        ],
        "risks_and_limitations": [
            "Business meaning of the target requires confirmation."
        ],
        "human_review_questions": [
            "Should duplicate records be retained?"
        ],
        "recommended_next_investigation": [
            "Confirm the business definition of the target."
        ],
        "requires_human_review": true
    }
    """


def test_parse_review_accepts_valid_json():
    result = _parse_review(valid_review_json())

    assert result.observed_evidence == [
        "The dataset contains 100 rows."
    ]
    assert result.requires_human_review is True


def test_parse_review_accepts_markdown_code_fence():
    response = f"```json\n{valid_review_json()}\n```"

    result = _parse_review(response)

    assert result.requires_human_review is True


def test_parse_review_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        _parse_review("not valid json")


def test_parse_review_rejects_invalid_schema():
    response = """
    {
        "observed_evidence": "this should be a list"
    }
    """

    with pytest.raises(ValueError):
        _parse_review(response)


def test_extract_text_returns_text_from_event():
    events = [
        FakeEvent(
            FakeContent(
                [
                    FakePart(text="First response"),
                ]
            )
        ),
        FakeEvent(
            FakeContent(
                [
                    FakePart(text="Final response"),
                ]
            )
        ),
    ]

    assert _extract_text(events) == "Final response"


def test_extract_text_skips_events_without_text():
    events = [
        FakeEvent(None),
        FakeEvent(
            FakeContent(
                [
                    FakePart(text=None),
                ]
            )
        ),
        FakeEvent(
            FakeContent(
                [
                    FakePart(text="Useful response"),
                ]
            )
        ),
    ]

    assert _extract_text(events) == "Useful response"


def test_extract_text_rejects_events_without_text():
    events = [
        FakeEvent(None),
        FakeEvent(
            FakeContent(
                [
                    FakePart(text=None),
                ]
            )
        ),
    ]

    with pytest.raises(
        ValueError,
        match="no textual response",
    ):
        _extract_text(events)
