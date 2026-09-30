from pathlib import Path

import pandas as pd
import pytest

from tools.data_loader import (
    UnsupportedFileTypeError,
    load_dataset,
)


def test_loads_csv_dataset():
    dataset = Path("datasets/customer_churn_poc.csv")

    dataframe = load_dataset(str(dataset))

    assert isinstance(dataframe, pd.DataFrame)
    assert not dataframe.empty
    assert len(dataframe.columns) > 0


def test_rejects_unsupported_file_type(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"customer_id": "C001"}', encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        load_dataset(str(dataset))


def test_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("datasets/does_not_exist.csv")
