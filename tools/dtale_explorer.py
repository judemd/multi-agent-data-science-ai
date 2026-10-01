from typing import Any

import pandas as pd


def create_dtale_session(
    dataframe: pd.DataFrame,
    *,
    name: str | None = None,
) -> Any:
    """Create an interactive D-Tale session for a DataFrame.

    D-Tale is an optional human-exploration surface. It does not produce
    governed workflow evidence and must not replace the deterministic
    DataUnderstandingArtifact.
    """

    import dtale

    return dtale.show(
        dataframe,
        name=name,
    )


def get_dtale_url(session: Any) -> str:
    """Return the browser URL associated with a D-Tale session."""

    return session._main_url
