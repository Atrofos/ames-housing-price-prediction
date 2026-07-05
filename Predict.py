"""
predict.py
----------
Loads the trained model and scores new houses, taking raw data in the same
format as AmesHousing.csv and producing predicted sale prices.

The central concern of any inference script is TRAINING-SERVING SKEW: if
new data is transformed even slightly differently from how the training
data was transformed, the model receives inputs that don't mean what it
learned, and predictions silently degrade — no error, just wrong numbers.

Two design choices defend against that:

  1. REUSE, don't reimplement. This script imports the exact cleaning and
     feature-engineering functions from data_cleaning.py and features.py
     and applies them in the same order. Re-writing the logic here would
     invite drift the moment either file changed; calling the same
     functions guarantees new data walks the identical path.

  2. ALIGN to the model, don't hardcode columns. One-hot encoding a small
     batch of new houses will NOT reproduce the training columns — a batch
     with no houses in 'Veenker' won't generate a Neighborhood_Veenker
     column, and a brand-new category would generate one the model has
     never seen. So after transforming, we reindex the columns to exactly
     what the model was trained on (model.feature_names_in_): missing
     columns are filled with 0, unexpected ones are dropped. Because we
     read the expected columns FROM the model itself, this script needs no
     changes if the feature set is ever revised and the model retrained.

The model predicts in log space (see model.py); predictions are converted
back to dollars with np.expm1 before output. The bundle's metadata records
the transform so this script applies the correct inverse.

Usage:
    python predict.py                  # demo on the raw dataset
    python predict.py path/to/new.csv  # score a batch of new houses

Input:  a CSV in AmesHousing.csv format (SalePrice optional)
Output: data/processed/predictions.csv
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

# Reuse the EXACT transformation logic from the pipeline (see design note
# above). Import names must match the filenames on disk — if you've named
# the files DataCleaning.py / FeatureSelection.py, adjust these two lines.
import data_cleaning as dc
import feature_selection as ft


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("models", "best_model.pkl")
DEFAULT_INPUT = os.path.join("data", "raw", "AmesHousing.csv")
OUTPUT_PATH = os.path.join("data", "processed", "predictions.csv")

TARGET = "SalePrice"
# Identifier columns kept aside for a readable output (dropped during
# cleaning, so we grab them before transforming).
ID_COLS = ["Order", "PID"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path: str):
    """
    Returns the full saved bundle: {"model", "metadata", and optionally
    "lower_model"/"upper_model" for prediction intervals}. We return the
    whole bundle (not just model+metadata) so predict() can reach the
    quantile models when they're present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model at {path}. Run model.py first to train and save one."
        )
    bundle = joblib.load(path)
    metadata = bundle["metadata"]
    interval_note = (" with 90% intervals" if metadata.get("has_intervals")
                     else " (point estimates only)")
    print(f"[load]     Model: {metadata['model_name']}{interval_note} "
          f"(trained test RMSE ${metadata['test_rmse_dollars']:,.0f})")
    return bundle


def load_new_data(path: str) -> pd.DataFrame:
    """
    Reads raw input the SAME way data_cleaning.py reads the raw dataset:
    with pandas' default NA (none) handling. This matters and is the opposite of
    what features.py/model.py do, because those read already-CLEANED files.

    Raw Ames data marks genuine missing values as the text 'NA' (e.g. a
    missing Lot Frontage). Pandas' default reader correctly turns 'NA'
    into NaN, keeping numeric columns numeric so imputation works. If we
    instead used keep_default_na=False here (as the cleaned-file readers
    do, to protect their literal 'None' strings), those 'NA's would stay
    as text, poison the column to string dtype, and break the median
    imputation. Different file conventions -> different read settings.
    """
    df = pd.read_csv(path)
    print(f"[load]     Input: {df.shape[0]} houses from {path}")
    return df


# ---------------------------------------------------------------------------
# Transformation — mirrors the run() order of each pipeline module
# ---------------------------------------------------------------------------

def clean_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same steps as data_cleaning.run(), minus load/validate/save."""
    df = dc.drop_columns(df)
    df = dc.fill_meaningful_nulls(df)
    df = dc.impute_remaining_nulls(df)
    df = dc.fix_dtypes(df)
    return df


def engineer_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same steps as features.run(), minus load/validate/save."""
    df = ft.drop_low_signal(df)
    df = ft.drop_redundant(df)
    df = ft.engineer_space_features(df)
    df = ft.engineer_bath_features(df)
    df = ft.engineer_porch_features(df)
    df = ft.engineer_age_features(df)
    df = ft.engineer_condition_flags(df)
    df = ft.engineer_functional_flag(df)
    df = ft.engineer_binary_conversions(df)
    df = ft.encode_ordinals(df)
    df = ft.encode_one_hot(df)
    return df


def align_to_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Force the feature matrix to match exactly what the model was trained
    on — same columns, same order. This is what makes batch prediction
    safe (see design note). Reports anything filled or dropped so silent
    mismatches become visible.
    """
    expected = list(model.feature_names_in_)

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]

    if missing:
        print(f"[align]    {len(missing)} expected column(s) absent in this "
              f"batch — filled with 0 (e.g. {missing[:3]}).")
    if extra:
        print(f"[align]    {len(extra)} column(s) not seen in training — "
              f"dropped (e.g. {extra[:3]}).")

    # reindex does both jobs at once: adds missing (fill 0), drops extra,
    # and enforces the training column order.
    return df.reindex(columns=expected, fill_value=0)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(bundle, X: pd.DataFrame):
    """
    Predict the point estimate and, if available, the bounds of a 90%
    prediction interval. Handles TWO interval mechanisms via the bundle's
    "interval_kind" flag — this is what keeps the XGBoost and TabPFN
    prediction outputs symmetric despite very different model internals:

      "quantile_models" (XGBoost): two SEPARATE models were trained on the
          5th/95th percentile quantile loss. We call each one. Three models
          total (point + lower + upper).

      "tabpfn_native" (TabPFN): a SINGLE fitted model returns the whole
          predictive distribution; quantiles are decoded from it in one
          forward pass via output_type="quantiles". No extra models.

      None / absent: no interval support -> point estimate only.

    Returns {"point": array, "lower": array|None, "upper": array|None}.
    Everything inverted from log space to dollars when the target was logged.
    Quantiles are preserved under the monotonic expm1, so a percentile in
    log space stays that percentile in dollars.
    """
    model = bundle["model"]
    metadata = bundle["metadata"]
    is_log = "log1p" in metadata.get("target_transform", "")
    kind = metadata.get("interval_kind")

    def to_dollars(arr):
        return np.expm1(arr) if is_log else arr

    out = {"point": None, "lower": None, "upper": None}

    if kind == "tabpfn_native":
        # One forward pass returns mean + the requested quantiles together.
        # We ask for 5th / 50th / 95th. Median (50th) is a more robust point
        # estimate than the mean for a skewed target, so we use it here.
        q = model.predict(X, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
        out["lower"] = to_dollars(np.asarray(q[0]))
        out["point"] = to_dollars(np.asarray(q[1]))
        out["upper"] = to_dollars(np.asarray(q[2]))

    elif kind == "quantile_models" and "lower_model" in bundle:
        out["point"] = to_dollars(model.predict(X))
        out["lower"] = to_dollars(bundle["lower_model"].predict(X))
        out["upper"] = to_dollars(bundle["upper_model"].predict(X))

    else:
        # No interval support — point estimate only.
        out["point"] = to_dollars(model.predict(X))

    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_output(original: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """
    Assemble a readable results frame. Transforms only ever drop columns,
    never rows, so predictions align to the original rows by position.

    Always includes PredictedPrice. If interval models were available,
    adds PriceLower / PriceUpper / IntervalWidth — the width column makes
    it easy to sort by "how uncertain is the model about this house".

    If the input carried a real SalePrice, include it plus error columns —
    turning the run into a sanity check (does predicted track actual?).
    Otherwise output predictions alone (true inference).
    """
    out = pd.DataFrame(index=original.index)

    for col in ID_COLS:
        if col in original.columns:
            out[col] = original[col]

    out["PredictedPrice"] = preds["point"].round(0).astype(int)

    has_intervals = preds["lower"] is not None
    if has_intervals:
        # Guard against quantile crossing (rare, but lower can edge above
        # upper on odd inputs) by sorting the two bounds per row.
        lower = np.minimum(preds["lower"], preds["upper"])
        upper = np.maximum(preds["lower"], preds["upper"])
        out["PriceLower"] = lower.round(0).astype(int)
        out["PriceUpper"] = upper.round(0).astype(int)
        out["IntervalWidth"] = (upper - lower).round(0).astype(int)

    if TARGET in original.columns:
        out["ActualPrice"] = original[TARGET].values
        out["Error"] = out["PredictedPrice"] - out["ActualPrice"]
        out["ErrorPct"] = (out["Error"].abs() / out["ActualPrice"] * 100).round(2)

        # If we have intervals, report how often the true price actually
        # falls inside the predicted range — the real test of interval quality.
        if has_intervals:
            inside = ((out["ActualPrice"] >= out["PriceLower"]) &
                      (out["ActualPrice"] <= out["PriceUpper"]))
            coverage = inside.mean() * 100
            print(f"\n[check]    Interval coverage: {coverage:.0f}% of actual "
                  f"prices fell within the 90% interval.")
            print(f"[check]    (A well-calibrated 90% interval should capture "
                  f"~90%. On synthetic test prices this is only indicative.)")

        mae = out["Error"].abs().mean()
        mape = out["ErrorPct"].mean()
        print(f"\n[check]    Ground truth present — predicted vs actual:")
        print(f"[check]      MAE  ${mae:,.0f}")
        print(f"[check]      MAPE {mape:.2f}%")
        print(f"[check]    (Note: if scoring data the model trained on, "
              f"this flatters the model — real performance is the held-out\n"
              f"[check]     test metrics in model.py, not these.)")

    return out


def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def save_output(out: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    remove_if_exists(path)
    out.to_csv(path, index=False)
    print(f"\n[save]     {len(out)} predictions -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(input_path: str = DEFAULT_INPUT, output_path: str = OUTPUT_PATH):
    bundle = load_model(MODEL_PATH)
    model = bundle["model"]

    original = load_new_data(input_path)

    # Keep ground truth (if any) aside so transforms never touch it.
    df = original.drop(columns=[TARGET], errors="ignore")

    df = clean_new_data(df)
    df = engineer_new_data(df)
    X = align_to_model(df, model)

    preds = predict(bundle, X)
    out = build_output(original, preds)
    save_output(out, output_path)


if __name__ == "__main__":
    # Optional CLI path; defaults to the raw dataset for a one-click demo.
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    run(path)