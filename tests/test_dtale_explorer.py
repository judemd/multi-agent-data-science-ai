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
    fake_dtale.show.assert_called_once()
    explored_data = fake_dtale.show.call_args.args[0]
    assert explored_data is not dataframe
    pd.testing.assert_frame_equal(explored_data, dataframe)
    assert fake_dtale.show.call_args.kwargs == {
        "name": "customers", "host": "127.0.0.1",
        "open_browser": False, "subprocess": True,
    }
    explored_data.loc[0, "revenue"] = 999
    assert dataframe.loc[0, "revenue"] == 100


def test_get_dtale_url_returns_session_url():
    session = Mock()
    session._main_url = "http://localhost:40000/dtale/main/1"

    assert get_dtale_url(session) == (
        "http://localhost:40000/dtale/main/1"
    )


def test_cloud_explorer_uses_same_origin_path(monkeypatch):
    monkeypatch.setenv("DTALE_APP_ROOT", "/explorer")
    session = Mock()
    session._main_url = "http://127.0.0.1:40000/explorer/dtale/main/1"
    session._data_id = "1"
    assert get_dtale_url(session) == "/explorer/dtale/main/1"


def test_cloud_explorer_uses_proxy_root_and_fixed_internal_port(monkeypatch):
    monkeypatch.setenv("DTALE_APP_ROOT", "/explorer")
    fake_dtale = Mock()
    monkeypatch.setitem(__import__("sys").modules, "dtale", fake_dtale)
    create_dtale_session(pd.DataFrame({"feature": [1]}))
    assert fake_dtale.show.call_args.kwargs["app_root"] == "/explorer"
    assert fake_dtale.show.call_args.kwargs["port"] == 40000
