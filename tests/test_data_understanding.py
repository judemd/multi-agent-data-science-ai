from agents.data_understanding import (
    DATA_UNDERSTANDING_INSTRUCTION,
    data_understanding_agent,
)


def test_data_understanding_agent_exists():
    assert data_understanding_agent is not None


def test_data_understanding_agent_has_expected_name():
    assert data_understanding_agent.name == "data_understanding_agent"


def test_data_understanding_instruction_is_present():
    assert DATA_UNDERSTANDING_INSTRUCTION.strip()


def test_data_understanding_instruction_prohibits_inventing_statistics():
    assert "Do not invent dataset statistics." in DATA_UNDERSTANDING_INSTRUCTION


def test_data_understanding_instruction_requires_evidence_separation():
    assert "observed evidence" in DATA_UNDERSTANDING_INSTRUCTION
    assert "interpretation" in DATA_UNDERSTANDING_INSTRUCTION
    assert "assumptions" in DATA_UNDERSTANDING_INSTRUCTION
    assert "risks" in DATA_UNDERSTANDING_INSTRUCTION
