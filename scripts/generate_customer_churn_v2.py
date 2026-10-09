"""Create a reproducible v2 synthetic churn dataset without altering v1."""

from pathlib import Path

import pandas as pd


SOURCE = Path("datasets/customer_churn_poc.csv")
DESTINATION = Path("datasets/customer_churn_poc_v2.csv")

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%Y/%m/%d",
)


def transform(source: pd.DataFrame) -> pd.DataFrame:
    result = source.copy(deep=True)

    charges = (
        result["monthly_charge"]
        .astype(str)
        .str.strip()
        .str.replace("\u00a3", "", regex=False)
    )
    result["monthly_charge"] = pd.to_numeric(charges, errors="raise")

    dates = result["signup_date"].astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")

    for date_format in DATE_FORMATS:
        candidate = pd.to_datetime(
            dates, format=date_format, errors="coerce"
        )
        parsed = parsed.fillna(candidate)

    if parsed.isna().any():
        raise ValueError("Some signup dates could not be parsed")

    result["signup_date"] = parsed.dt.strftime("%Y-%m-%d")

    # Synthetic age repair: original invalid values are not recoverable.
    ages = pd.to_numeric(result["age"], errors="coerce")
    invalid_age = ages.isna() | (ages < 18) | (ages > 110)
    valid_age = ages.loc[~invalid_age]

    if valid_age.empty:
        raise ValueError("No valid ages available for median repair")

    age_median = valid_age.median()
    result["age"] = ages.mask(invalid_age, age_median)

    # Synthetic total-charge repair: change only injected extremes.
    extreme_charge = result["total_charge"] > 20000
    estimated_charge = (
        result["monthly_charge"] * result["tenure_months"]
    )
    result.loc[extreme_charge, "total_charge"] = (
        estimated_charge.loc[extreme_charge]
    )

    # Remove outcome-leakage columns from the v2 dataset.
    result = result.drop(
        columns=["account_closed_date", "retention_offer_accepted"]
    )

    print(f"Invalid ages repaired: {invalid_age.sum()}")
    print(f"Age replacement median: {age_median:g}")
    print(f"Extreme total charges repaired: {extreme_charge.sum()}")
    print("Leakage columns removed: 2")

    return result


def validate(source: pd.DataFrame, result: pd.DataFrame) -> None:
    """Check that v2 preserves source integrity and agreed scenarios."""
    removed = {"account_closed_date", "retention_offer_accepted"}
    expected_columns = [c for c in source.columns if c not in removed]

    if len(result) != len(source):
        raise ValueError("Row count changed")

    if result.columns.tolist() != expected_columns:
        raise ValueError("Unexpected column changes")

    for column in ("customer_id", "account_number", "churned"):
        if not source[column].equals(result[column]):
            raise ValueError(f"{column} changed")

    if source.duplicated().sum() != result.duplicated().sum():
        raise ValueError("Exact duplicate count changed")

    changed_columns = {
        "age", "monthly_charge", "signup_date", "total_charge"
    }
    unchanged_columns = [
        c for c in expected_columns if c not in changed_columns
    ]

    for column in unchanged_columns:
        if not source[column].equals(result[column]):
            raise ValueError(f"Unintended changes to {column}")

    ages = pd.to_numeric(result["age"], errors="coerce")
    if ages.isna().any() or ((ages < 18) | (ages > 110)).any():
        raise ValueError("Invalid ages remain")

    charges = pd.to_numeric(result["monthly_charge"], errors="coerce")
    if charges.isna().any():
        raise ValueError("Unparseable monthly charges remain")

    dates = pd.to_datetime(
        result["signup_date"], format="%Y-%m-%d", errors="coerce"
    )
    if dates.isna().any():
        raise ValueError("Invalid signup dates remain")

    if (result["total_charge"] > 20000).any():
        raise ValueError("Extreme total charges remain")

    print("Validation passed: row count, columns, identifiers, target")
    print("Validation passed: duplicates and unchanged fields")
    print("Validation passed: ages, charges, signup dates")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    if DESTINATION.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {DESTINATION}"
        )

    source = pd.read_csv(SOURCE)
    result = transform(source)
    validate(source, result)

    print(f"Source records: {len(source):,}")
    print(f"Transformed records: {len(result):,}")
    print("Monthly charge dtype:", result["monthly_charge"].dtype)
    print("Signup date format: YYYY-MM-DD")
    # Exclusive creation prevents accidental replacement of an existing v2.
    with DESTINATION.open("x", encoding="utf-8", newline="") as output:
        result.to_csv(output, index=False)

    print(f"Created: {DESTINATION}")


if __name__ == "__main__":
    main()
