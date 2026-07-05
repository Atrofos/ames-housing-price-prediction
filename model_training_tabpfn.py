"""
model_training_tabpfn.py
----------------
Scores the Ames dataset with TabPFN — a tabular FOUNDATION model — as a
"state of the art" counterpoint to the four traditional models in model.py.

What makes TabPFN different: it's a transformer pre-trained on millions of
synthetic tabular tasks, so it makes predictions by IN-CONTEXT LEARNING
rather than per-dataset training. In practice that means NO hyperparameter
tuning at all — the entire GridSearch/RandomizedSearch machinery in
model.py collapses to a single .fit()/.predict(). This file is the test of
whether all that tuning effort actually buys a win over a zero-tuning
foundation model on a small dataset like ours.

This file runs TabPFN TWO ways, which gives two distinct comparisons:

  1. LOG target — identical harness to model.py (same split, same seed,
     same log1p transform, same dollar metrics). The FAIR FIGHT:
     whoever wins under identical conditions wins cleanly.

  2. RAW target — no transform, predicting dollars directly. Probes the
     claim that a foundation model handles skewed targets well on its own,
     without the manual log crutch the linear models needed. Asking a
     linear model to do this would be unfair; asking TabPFN is a
     legitimate strength test.

The DELTA between the two is itself the finding:
  - raw ~= log  -> TabPFN doesn't need the manual preprocessing crutch
  - raw worse   -> even foundation models benefit from target engineering
  - raw better  -> the log transform was slightly holding TabPFN back

Kept deliberately parallel to model.py (same split, RANDOM_STATE, metrics,
output style) and quarantined from the main pipeline so the core project
still runs for anyone without TabPFN installed.

-----------------------------------------------------------------------------
ONE-TIME SETUP (learned the hard way — documented so it never bites again)
-----------------------------------------------------------------------------
Requirements: pip install tabpfn   (downloads a ~233MB checkpoint on first
                                     fit; needs internet that one time).

1) LICENCE / TOKEN. Recent TabPFN (v2.5+/v3, the current default) gates the
   model weights behind a one-time licence acceptance. On first fit it tries
   to open a browser to log in — but that flow crashes on Windows with
   `OSError [WinError 10038] ... not a socket`, because it calls
   select.select([sys.stdin]) which only works on sockets on Windows, not
   on stdin. It's a library bug, not your setup.

   The robust fix is to cache an API key directly, bypassing the browser:
     - Get a key from https://ux.priorlabs.ai/account (accept the licence
       at https://ux.priorlabs.ai/account/licenses).
     - Save it ONCE — see ensure_authenticated() below. It writes the token
       to a file on disk (under your user profile), which persists across
       terminal sessions. Future runs just read that file via
       get_cached_token(), so no browser, no env var, no prompt is needed
       after the first save. (To clear it: tabpfn.browser_auth
       .delete_cached_token().)

   To set the key for the first save, put it in the PRIORLABS_API_KEY
   environment variable before running, e.g. on Windows:
       set PRIORLABS_API_KEY=tabpfn_sk_xxxxx        (NO spaces around =)
   ensure_authenticated() reads it and caches it. After that the env var is
   irrelevant — the on-disk cache is the source of truth.
   Security note: the cached token is PLAINTEXT in your profile; treat it
   like a password and rotate it if exposed.

2) DEVICE. device="auto" selects CUDA if available, else CPU. NOTE: a
   default `pip install torch` is often the CPU-ONLY build, in which case
   "cuda" raises "Torch not compiled with CUDA enabled" even with a good
   GPU. CPU is fine for this dataset (~2.3k rows runs in seconds-to-a-
   minute). For real GPU use, install a CUDA build of torch, e.g.:
       pip install torch --index-url https://download.pytorch.org/whl/cu124
-----------------------------------------------------------------------------

Input:  data/processed/ames_featured.csv
Output: models/best_model_tabpfn.pkl, models/model_results_tabpfn.csv
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# TabPFN is an optional, quarantined dependency — import lazily with a
# clear message so this file fails helpfully rather than cryptically if
# the package isn't installed.
try:
    from tabpfn import TabPFNRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TabPFN is not installed. Run `pip install tabpfn` (a GPU is "
        "recommended; a checkpoint downloads automatically on first fit)."
    ) from exc


# ---------------------------------------------------------------------------
# Config — kept identical to model.py so the comparison is apples-to-apples
# ---------------------------------------------------------------------------

INPUT_PATH = os.path.join("data", "processed", "ames_featured.csv")
MODEL_PATH = os.path.join("models", "best_model_tabpfn.pkl")
RESULTS_PATH = os.path.join("models", "model_results_tabpfn.csv")

RANDOM_STATE = 42       # same seed as model.py -> same train/test split
TEST_SIZE = 0.2

DEVICE = "auto"         # selects CUDA GPU if available, else CPU


# ---------------------------------------------------------------------------
# Authentication (see the ONE-TIME SETUP note in the module docstring)
# ---------------------------------------------------------------------------

def ensure_authenticated() -> None:
    """
    Make sure a TabPFN licence token is cached on disk before fitting, so the
    fit never falls into the browser-login flow that crashes on Windows.

    Logic:
      - If a token is already cached (from a previous run), do nothing — the
        on-disk cache persists across sessions and is the source of truth.
      - Otherwise, if PRIORLABS_API_KEY is set in the environment, save it to
        the cache once via save_token(). After this first save the env var is
        no longer needed on future runs.
      - If neither is available, print a clear instruction rather than letting
        the fit crash cryptically later.

    This is best-effort and never raises: if the auth module's internals
    differ across TabPFN versions, we warn and let .fit() handle auth itself.
    """
    try:
        import tabpfn.browser_auth as ba

        if ba.get_cached_token():
            print("[auth]     Cached TabPFN token found — good to go.")
            return

        key = os.environ.get("PRIORLABS_API_KEY")
        if key:
            ba.save_token(key)
            print("[auth]     Saved PRIORLABS_API_KEY to TabPFN's on-disk "
                  "cache (persists for future runs).")
        else:
            print("[auth]     No cached token and PRIORLABS_API_KEY not set. "
                  "Get a key from https://ux.priorlabs.ai/account, then run "
                  "with it set:  set PRIORLABS_API_KEY=tabpfn_sk_...  (no "
                  "spaces around '='). See the module docstring for details.")
    except Exception as e:  # pragma: no cover - version-dependent internals
        print(f"[auth]     Could not pre-cache token ({e}); letting .fit() "
              "handle authentication directly.")


# ---------------------------------------------------------------------------
# Data prep — mirrors model.py exactly
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    print(f"[load]     Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def split_data(df: pd.DataFrame):
    """
    Same split, same seed as model.py. Because RANDOM_STATE matches, the
    exact same rows land in train and test as in the traditional run —
    which is what makes the cross-file comparison valid rather than
    approximate.
    """
    X = df.drop(columns=["SalePrice"])
    y_raw = df["SalePrice"]

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[split]    Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows "
          f"(same split as model.py).")
    return X_train, X_test, y_train_raw, y_test_raw


# ---------------------------------------------------------------------------
# Metrics — identical to model.py's dollar_metrics
# ---------------------------------------------------------------------------

def dollar_metrics(y_true_raw, y_pred_dollars) -> dict:
    return {
        "RMSE ($)": np.sqrt(mean_squared_error(y_true_raw, y_pred_dollars)),
        "MAE ($)": mean_absolute_error(y_true_raw, y_pred_dollars),
        "MAPE (%)": mean_absolute_percentage_error(y_true_raw, y_pred_dollars) * 100,
        "R2": r2_score(y_true_raw, y_pred_dollars),
    }


# ---------------------------------------------------------------------------
# The experiment — one function, run twice (log vs raw) via a single flag
# ---------------------------------------------------------------------------

def evaluate_tabpfn(X_train, X_test, y_train_raw, y_test_raw, log_target: bool):
    """
    Fits TabPFN and evaluates on the held-out test set. The ONLY difference
    between the two runs is the target transform, so the logic lives in one
    place and the boolean expresses exactly what varies (same DRY reasoning
    as model.py's shared maps).

    Returns (fitted_model, metrics_dict). Metrics are always in dollars:
    when log_target is True we predict in log space and expm1 back, so both
    variants are measured on the same real-money footing.
    """
    label = "log target" if log_target else "raw target"
    print(f"[tabpfn]   Fitting ({label}) ...")

    # Foundation model: no tuning, no CV, no parameter grid. Just fit.
    model = TabPFNRegressor(device=DEVICE, random_state=RANDOM_STATE)

    if log_target:
        model.fit(X_train, np.log1p(y_train_raw))
        pred_dollars = np.expm1(model.predict(X_test))
    else:
        model.fit(X_train, y_train_raw)
        pred_dollars = model.predict(X_test)

    metrics = dollar_metrics(y_test_raw, pred_dollars)
    print(f"[tabpfn]     RMSE ${metrics['RMSE ($)']:,.0f} | "
          f"MAE ${metrics['MAE ($)']:,.0f} | "
          f"MAPE {metrics['MAPE (%)']:.2f}% | R2 {metrics['R2']:.3f}")
    return model, metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"[cleanup]  Existing file removed: {path}")


def save_outputs(results: pd.DataFrame, models: dict) -> None:
    """
    Saves the better of the two TabPFN variants (lower test RMSE) plus the
    two-row results table.

    Caveat noted honestly: TabPFN model objects are large and bundle a
    transformer checkpoint, so the .pkl is heavy and its loadability can be
    sensitive to the tabpfn/torch versions installed at load time. For a
    portfolio the saved file is fine as an artifact; in a real deployment
    you'd typically re-fit (it's fast — there's no training to redo) or use
    the cloud client rather than ship the pickle.
    """
    os.makedirs("models", exist_ok=True)

    winner_row = results.loc[results["RMSE ($)"].idxmin()]
    winner_label = winner_row["Model"]
    winner_key = "log" if "log" in winner_label else "raw"

    metadata = {
        "model_name": winner_label,
        "target_transform": ("log1p (invert predictions with np.expm1)"
                             if winner_key == "log" else "none (raw dollars)"),
        "test_rmse_dollars": float(winner_row["RMSE ($)"]),
        "test_mae_dollars": float(winner_row["MAE ($)"]),
        "test_mape_pct": float(winner_row["MAPE (%)"]),
        "test_r2": float(winner_row["R2"]),
        "random_state": RANDOM_STATE,
        # TabPFN produces intervals NATIVELY: a single fitted model returns
        # the full predictive distribution, and quantiles are decoded from it
        # at predict-time via output_type="quantiles" — no extra models. This
        # flag tells predict.py to use that native path (vs XGBoost's separate
        # lower_model/upper_model). The contrast is the whole point of the
        # symmetric comparison.
        "has_intervals": True,
        "interval_kind": "tabpfn_native",   # vs "quantile_models" for XGBoost
        "interval_level": "90% (5th-95th percentile)",
    }

    remove_if_exists(MODEL_PATH)
    joblib.dump({"model": models[winner_key], "metadata": metadata}, MODEL_PATH)
    print(f"\n[save]     Better variant: {winner_label} "
          f"(test RMSE ${winner_row['RMSE ($)']:,.0f}) -> {MODEL_PATH}")

    remove_if_exists(RESULTS_PATH)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"[save]     Results table -> {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    ensure_authenticated()

    df = load_data(INPUT_PATH)
    X_train, X_test, y_train_raw, y_test_raw = split_data(df)

    rows = []
    models = {}

    log_model, log_metrics = evaluate_tabpfn(
        X_train, X_test, y_train_raw, y_test_raw, log_target=True
    )
    models["log"] = log_model
    rows.append({"Model": "TabPFN (log target)", **log_metrics})

    raw_model, raw_metrics = evaluate_tabpfn(
        X_train, X_test, y_train_raw, y_test_raw, log_target=False
    )
    models["raw"] = raw_model
    rows.append({"Model": "TabPFN (raw target)", **raw_metrics})

    results = pd.DataFrame(rows)
    print("\n[results]  TabPFN — held-out test set:\n")
    print(results.round(3).to_string(index=False))

    # The delta IS the finding — surface it explicitly.
    delta = raw_metrics["RMSE ($)"] - log_metrics["RMSE ($)"]
    print(f"\n[delta]    Raw RMSE - Log RMSE = ${delta:,.0f} "
          f"({'log helps' if delta > 0 else 'raw is as good or better'}).")

    save_outputs(results, models)


if __name__ == "__main__":
    run()