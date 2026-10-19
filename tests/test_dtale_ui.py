from concurrent.futures import Future
from unittest.mock import Mock

import pandas as pd
import pytest

import ui_dtale


def test_reference_starts_once_per_dataset_without_waiting(monkeypatch):
    session = {}
    future = Future()
    executor = Mock()
    executor.submit.return_value = future
    monkeypatch.setattr(ui_dtale, "_explorer_executor", lambda: executor)
    monkeypatch.setattr(ui_dtale, "_render_explorer_status", Mock())
    monkeypatch.setattr(ui_dtale.st, "session_state", session)
    monkeypatch.setattr(ui_dtale.st, "caption", Mock())
    monkeypatch.setattr(ui_dtale.st, "write", Mock())
    dataframe = pd.DataFrame({"feature": [1, 2]})

    ui_dtale.render_dtale_reference(dataframe, "project-one", "churn.csv")
    ui_dtale.render_dtale_reference(dataframe, "project-one", "churn.csv")

    executor.submit.assert_called_once()
    snapshot = executor.submit.call_args.args[1]
    assert snapshot is not dataframe
    snapshot.loc[0, "feature"] = 10
    assert dataframe.loc[0, "feature"] == 1
    assert not future.done()

    ui_dtale.render_dtale_reference(dataframe, "project-two", "other.csv")
    assert executor.submit.call_count == 2


def test_start_explorer_returns_url(monkeypatch):
    explorer = Mock()
    explorer._main_url = "http://127.0.0.1:40000/dtale/main/1"
    monkeypatch.setattr(ui_dtale, "create_dtale_session", Mock(return_value=explorer))
    assert ui_dtale._start_explorer(pd.DataFrame()) == explorer._main_url


def test_start_explorer_reports_missing_session(monkeypatch):
    monkeypatch.setattr(ui_dtale, "create_dtale_session", Mock(return_value=None))
    with pytest.raises(RuntimeError, match="could not create"):
        ui_dtale._start_explorer(pd.DataFrame())
