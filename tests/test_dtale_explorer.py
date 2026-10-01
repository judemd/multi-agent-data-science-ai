from unittest.mock import Mock

import pandas as pd

from tools.dtale_explorer import (
    create_dtale_session,
    get_dtale_url,
)


def test_create_dtale_session_passes_dataframe_and_name(monkeypatch):
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    session = Mock()
    fake_dtale = Mock()
    fake_dtale.show.return_value = session

    monkeypatch.setitem(
        __import__("sys").modules,
        "dtale",
        fake_dtale,
    )

    result = create_dtale_session(
        dataframe,
        name="customers",
    )

    assert result is session
    fake_dtale.show.assert_called_once_with(
        dataframe,
        name="customers",
    )


def test_get_dtale_url_returns_session_url():
    session = Mock()
    session._main_url = "http://localhost:40000/dtale/main/1"

    assert get_dtale_url(session) == (
        "http://localhost:40000/dtale/main/1"
    )
