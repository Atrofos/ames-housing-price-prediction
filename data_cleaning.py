"""
data_cleaning.py
----------------
Loads the raw Ames Housing dataset, handles missing values, corrects data
types, drops low-signal columns, and saves a cleaned CSV ready for feature
engineering.

Input:  data/AmesHousing.csv
Output: data/processed/ames_cleaned.csv
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_PATH = os.path.join("data","raw", "AmesHousing.csv")
OUTPUT_PATH = os.path.join("data", "processed", "ames_cleaned.csv")


# ---------------------------------------------------------------------------
# Columns to drop entirely
# Low variance, identifiers, or too sparse to be useful
# ---------------------------------------------------------------------------

COLS_TO_DROP = [
    "Order",          # Row identifier, no predictive value
    "PID",            # Parcel ID, no predictive value
    "Utilities",      # Almost all rows are 'AllPub' — zero variance
    "Street",         # 99%+ are 'Pave'
    "Alley",          # ~93% missing — not meaningful to impute
    "Condition 2",    # Very rare second condition, mostly 'Norm'
    "Roof Matl",      # Nearly all 'CompShg', other values too rare
    "Pool QC",        # 99%+ missing — almost no houses have pools
    "Pool Area",      # Same reason as Pool QC
    "Misc Feature",   # 96% missing, low signal
    "Misc Val",       # Tied to Misc Feature — same issue
    "3Ssn Porch",     # Extremely low variance, most values are 0
    "Low Qual Fin SF" # Very low variance, rarely non-zero
]


# ---------------------------------------------------------------------------
# Null handling maps
# ---------------------------------------------------------------------------

# These nulls mean the feature genuinely doesn't exist on the property.
# Fill with 'None' (string) for categoricals, 0 for numerics.
CATEGORICAL_NONE_FILLS = [
    "Mas Vnr Type",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Fence",
]

NUMERIC_ZERO_FILLS = [
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Garage Yr Blt",   # Will fill with 0; can be handled further in features.py
    "Garage Area",
    "Garage Cars",
]


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]    Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[1]
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    after = df.shape[1]
    print(f"[drop]    Removed {before - after} low-signal columns. {after} remaining.")
    return df


def fill_meaningful_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills nulls that represent absence of a feature rather than missing data.
    e.g. no garage → Garage Type is null → fill with 'None'
    """
    for col in CATEGORICAL_NONE_FILLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in NUMERIC_ZERO_FILLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"[nulls]   Filled meaningful nulls in {len(CATEGORICAL_NONE_FILLS)} categorical "
          f"and {len(NUMERIC_ZERO_FILLS)} numeric columns.")
    return df


def impute_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the small number of genuinely missing values.
    - Lot Frontage: median imputation grouped by Neighborhood (better than global median)
    - Electrical: single missing row — fill with mode
    - Any remaining numeric nulls: global median fallback
    """
    # Lot Frontage — neighbourhood-grouped median
    if "Lot Frontage" in df.columns:
        df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
            lambda x: x.fillna(x.median())
        )
        print(f"[impute]  'Lot Frontage' imputed using neighbourhood-grouped median.")

    # Electrical — single missing row
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
        print(f"[impute]  'Electrical' imputed using mode.")

    # Fallback: any remaining numeric nulls
    numeric_cols = df.select_dtypes(include="number").columns
    remaining = df[numeric_cols].isnull().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        for col in remaining.index:
            df[col] = df[col].fillna(df[col].median())
            print(f"[impute]  '{col}' imputed using global median (fallback).")

    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    A small number of columns are stored as integers but are actually
    ordinal/categorical codes. Convert them so downstream encoding handles
    them correctly.
    """
    categorical_ints = [
        "MS SubClass",   # Building class code — not a continuous number
        "Mo Sold",       # Month sold — cyclical, not linear
        "Yr Sold",       # Year sold — treat as category for encoding
    ]
    for col in categorical_ints:
        if col in df.columns:
            df[col] = df[col].astype(str)

    print(f"[dtypes]  Converted {len(categorical_ints)} integer columns to categorical strings.")
    return df

def wait_until_closed(path: str, timeout: int = 30, interval: int = 3) -> None:
    """
    If the file is currently open by another process (e.g. Excel, a notebook),
    waits until it is closed before proceeding. Retries every `interval` seconds
    up to `timeout` seconds total, then raises a TimeoutError if still locked.
    """
    import time
    elapsed = 0
    while True:
        try:
            os.rename(path, path)  # Raises PermissionError on Windows if file is open
            break
        except PermissionError:
            if elapsed == 0:
                print(f"[cleanup] '{path}' is open in another process. Waiting for it to close...")
            print(f"[cleanup] Retrying in {interval}s... ({elapsed + interval}/{timeout}s elapsed)")
            time.sleep(interval)
            elapsed += interval
            if elapsed >= timeout:
                raise TimeoutError(
                    f"[cleanup] Timed out after {timeout}s. "
                    f"Please close '{path}' and re-run the script."
                )
 
 
def remove_if_exists(path: str) -> None:
    """
    Checks if the output file already exists and removes it before saving.
    If the file is open in another process, waits for it to be closed first.
    """
    if os.path.exists(path):
        wait_until_closed(path)
        os.remove(path)
        print(f"[cleanup] Existing file removed: {path}")
    else:
        print(f"[cleanup] No existing file found at: {path}")
 
 

def validate(df: pd.DataFrame) -> None:
    """
    Quick sanity checks after cleaning.
    """
    remaining_nulls = df.isnull().sum().sum()
    print(f"\n[validate] Remaining nulls: {remaining_nulls}")
    print(f"[validate] Final shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    if remaining_nulls > 0:
        print("[validate] WARNING — nulls still present. Review impute_remaining_nulls().")
    else:
        print("[validate] Clean. Ready for feature engineering.")


def save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n[save]    Saved cleaned data to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("Running Data Cleaning")
    df = load_data(RAW_PATH)
    df_dropped = drop_columns(df)
    df_filled = fill_meaningful_nulls(df_dropped)
    df_imputed = impute_remaining_nulls(df_filled)
    df_clean = fix_dtypes(df_imputed)
    validate(df_clean)
    remove_if_exists(OUTPUT_PATH)
    save(df_clean, OUTPUT_PATH)


if __name__ == "__main__":
    run()
