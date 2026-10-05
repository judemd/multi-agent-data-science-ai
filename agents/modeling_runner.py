import asyncio
import json
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.modeling import modeling_agent
from domain.modeling_review import ModelingReview

APP_NAME = "multi_agent_data_science_ai"
DEFAULT_USER_ID = "data_science_user"


def _build_runner(
    session_service: InMemorySessionService,
) -> Runner:
    """Create an ADK runner for the Modeling Agent."""

    return Runner(
        app_name=APP_NAME,
        agent=modeling_agent,
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
        "The Modeling Agent returned no textual response."
    )


def _parse_review(response_text: str) -> ModelingReview:
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
            "The Modeling Agent did not return valid JSON."
        ) from exc

    return ModelingReview.model_validate(payload)


async def _run_modeling_agent(
    prompt: str,
    *,
    user_id: str,
    session_id: str,
    session_service: InMemorySessionService,
) -> ModelingReview:
    """Create the session and execute the agent."""

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    runner = _build_runner(session_service)

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


def run_modeling_agent(
    prompt: str,
    *,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = "modeling",
    session_service: InMemorySessionService | None = None,
) -> ModelingReview:
    """Execute the Modeling Agent and validate its review."""

    service = session_service or InMemorySessionService()

    return asyncio.run(
        _run_modeling_agent(
            prompt,
            user_id=user_id,
            session_id=session_id,
            session_service=service,
        )
    )
