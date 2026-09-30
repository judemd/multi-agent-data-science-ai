from enum import StrEnum


class HumanDecision(StrEnum):
    """Valid human decisions at workflow approval gates."""

    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    SELECT_MODEL = "SELECT_MODEL"
    GO = "GO"
    ITERATE = "ITERATE"
    NO_GO = "NO_GO"
    APPROVE_HANDOFF = "APPROVE_HANDOFF"
    REJECT = "REJECT"
