from pathlib import Path

import pandas as pd
import pytest

from tools.data_loader import (
    UnsupportedFileTypeError,
    load_dataset,
    load_uploaded_dataset,
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

def test_load_uploaded_csv():
    content = (
        b"customer_id,revenue\n"
        b"C001,100\n"
        b"C002,200\n"
    )

    result = load_uploaded_dataset(
        "customers.csv",
        content,
    )

    assert list(result.columns) == [
        "customer_id",
        "revenue",
    ]
    assert len(result) == 2
    assert result["revenue"].tolist() == [100, 200]
