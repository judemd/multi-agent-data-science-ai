from hashlib import sha256

import pytest

from tools.dataset_fingerprint import fingerprint_dataset_file


def test_fingerprint_matches_sha256_of_original_bytes(tmp_path):
    content = b"customer_id,revenue\nC001,100\nC002,200\n"
    path = tmp_path / "customers.csv"
    path.write_bytes(content)

    assert fingerprint_dataset_file(path) == sha256(content).hexdigest()


def test_fingerprint_is_stable_for_unchanged_file(tmp_path):
    path = tmp_path / "customers.csv"
    path.write_bytes(b"customer_id\nC001\n")

    first = fingerprint_dataset_file(path)
    second = fingerprint_dataset_file(path)

    assert first == second


def test_fingerprint_changes_when_file_is_overwritten(tmp_path):
    path = tmp_path / "active_dataset.csv"
    path.write_bytes(b"customer_id\nC001\n")
    original = fingerprint_dataset_file(path)

    path.write_bytes(b"customer_id\nC002\n")

    assert fingerprint_dataset_file(path) != original


def test_fingerprint_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint_dataset_file(tmp_path / "missing.csv")
