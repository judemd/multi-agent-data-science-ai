import json
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.data_understanding import data_understanding_agent
from domain.data_understanding_review import DataUnderstandingReview

APP_NAME = "multi_agent_data_science_ai"
DEFAULT_USER_ID = "data_science_user"


def _build_runner(
    session_service: InMemorySessionService,
) -> Runner:
    """Create an ADK runner for the Data Understanding Agent."""

    return Runner(
        app_name=APP_NAME,
        agent=data_understanding_agent,
        session_service=session_service,
    )


def _extract_text(events: list[Any]) -> str:
    """Extract the final textual response from ADK events."""

    for event in reversed(events):
        content = getattr(event, "content", None)

        if content is None:
            continue

        parts = getattr(content, "parts", None)

        if not parts:
            continue

        for part in reversed(parts):
            text = getattr(part, "text", None)

            if text:
                return text

    raise ValueError(
        "The Data Understanding Agent returned no textual response."
    )


def _parse_review(response_text: str) -> DataUnderstandingReview:
    """Parse and validate the agent's structured review."""

    cleaned_response = response_text.strip()

    if cleaned_response.startswith("```"):
        lines = cleaned_response.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_response = "\n".join(lines).strip()

    try:
        payload = json.loads(cleaned_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The Data Understanding Agent did not return valid JSON."
        ) from exc

    return DataUnderstandingReview.model_validate(payload)


def run_data_understanding_agent(
    prompt: str,
    *,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = "data_understanding",
    session_service: InMemorySessionService | None = None,
) -> DataUnderstandingReview:
    """Execute the Data Understanding Agent and validate its review."""

    service = session_service or InMemorySessionService()

    service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    runner = _build_runner(service)

    message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=prompt,
            )
        ],
    )

    events = list(
        runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        )
    )

    response_text = _extract_text(events)

    return _parse_review(response_text)
