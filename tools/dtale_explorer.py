import os
from typing import Any
from urllib.parse import quote

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

    options = {}
    if os.getenv("DTALE_APP_ROOT"):
        options = {"app_root": os.environ["DTALE_APP_ROOT"], "port": 40000}

    return dtale.show(
        dataframe.copy(deep=True),
        name=name,
        host="127.0.0.1",
        open_browser=False,
        subprocess=True,
        **options,
    )


def get_dtale_url(session: Any) -> str:
    """Return the browser URL associated with a D-Tale session."""

    if root := os.getenv("DTALE_APP_ROOT"):
        return f"{root.rstrip('/')}/dtale/main/{quote(str(session._data_id), safe='')}"
    return session._main_url
