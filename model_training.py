"""
model_training.py
--------
Trains, tunes, and compares four regression models on the featured Ames
dataset, then saves the winner and a results table.

The experiment is a deliberate PROGRESSION, each model adds one concept,
and each must justify its added complexity by beating the previous:

    1. Linear Regression  — the baseline. No knobs, fully interpretable.
                            Its job is to set the bar, not to win.
                            (Run twice: raw target vs log target, as an
                            explicit demonstration of why we log.)
    2. Ridge & Lasso      — linear + a penalty on coefficient size
                            (regularisation), tuned via GridSearchCV.
                            Lasso can shrink coefficients to exactly zero:
                            automatic feature selection, a second opinion
                            on the manual selection done in features.py.
    3. Random Forest      — first non-linear model: hundreds of decision
                            trees on random data/feature subsets, averaged.
                            Captures interactions linear models can't
                            (e.g. quality matters MORE in big houses).
    4. XGBoost            — boosted trees built sequentially, each tree
                            correcting the previous ones' errors. The
                            usual winner on tabular data.

Methodology decisions (and why):

    TARGET = log1p(SalePrice).
        Prices are right-skewed (skew 1.74 -> -0.01 after log) and price
        effects are multiplicative (a garage adds ~X%, not a flat $X).
        log converts multiplicative structure into the additive form
        linear models can express, equalises error penalties across the
        price range (a $30k miss on a $100k house is a disaster; on a
        $700k house it's 4%), and stabilises residual variance.
        Predictions are expm1()'d back to dollars BEFORE computing
        metrics, so all reported numbers are in real money.

    HOLD-OUT TEST SET (20%), touched exactly once.
        All tuning happens via 5-fold cross-validation INSIDE the
        training 80%. If the test set influences any decision — even
        indirectly, by peeking at scores and re-tuning — it stops
        measuring generalisation and starts measuring memorisation
        ("leakage through the researcher").

    5-FOLD CV during tuning.
        One train/val split makes the score hostage to which rows landed
        in the split; K-fold averages over K rotations so every row is
        validated exactly once. K=5 on ~2,344 training rows gives ~469
        rows per validation fold — plenty — at half the cost of K=10.

    GRID search for Ridge/Lasso, RANDOMIZED search for RF/XGBoost.
        Ridge/Lasso have ONE knob (alpha): an exhaustive 1-D grid is
        cheap and complete. RF/XGBoost have 4-6 interacting knobs whose
        full grid would be thousands of fits; sampling N random
        combinations gets near-optimal results at a fraction of the cost.

    StandardScaler INSIDE a Pipeline, for linear models only.
        Ridge/Lasso penalise coefficient SIZE, so features on wild scales
        (Lot Area in tens of thousands vs TotalBaths at 1-4.5) would be
        penalised unfairly without scaling. Trees split on thresholds and
        are scale-invariant — they get raw features. The Pipeline matters:
        it re-fits the scaler on each CV training fold only, so no
        validation-fold statistics ever leak into training.

    Known caveat — encoding before splitting:
        features.py one-hot encoded on the full dataset before this
        script's train/test split. For dummy variables this is benign
        (column structure only; no statistics are learned from the data).
        It would NOT be benign for target encoding — which learns mean
        prices — and that is exactly why Neighborhood was one-hotted
        rather than target-encoded. Knowing where leakage can and cannot
        hide matters more than ritual.

Input:  data/processed/ames_featured.csv
Output: models/best_model.pkl, models/model_results.csv
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Config — paths and reproducibility
# ---------------------------------------------------------------------------

INPUT_PATH = os.path.join("data", "processed", "ames_featured.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")
RESULTS_PATH = os.path.join("models", "model_results.csv")

# One seed used EVERYWHERE (split, CV shuffling, RF, XGBoost, randomized
# search) so every run of this script reproduces identical numbers.
RANDOM_STATE = 42

TEST_SIZE = 0.2
K_FOLDS = 5

# Randomized search budget: 30 sampled combinations x 5 folds = 150 fits
# per tree model. Thorough enough to find a strong region of the space,
# cheap enough to run on a laptop.
N_RANDOM_CANDIDATES = 30


# ---------------------------------------------------------------------------
# Config — hyperparameter spaces
# ---------------------------------------------------------------------------

# alpha = regularisation strength (the ONLY knob for Ridge/Lasso).
# Log-spaced because alpha's effect is multiplicative: the interesting
# difference is between 0.01 and 0.1, not between 10.01 and 10.1.
# 'model__' prefix is sklearn's syntax for reaching through a Pipeline
# to the step named 'model'.
RIDGE_GRID = {"model__alpha": np.logspace(-3, 2, 30)}

# Lasso gets a slightly higher floor: with very small alpha it converges
# slowly and behaves like plain LinearRegression anyway.
LASSO_GRID = {"model__alpha": np.logspace(-3, 1, 30)}

# Random Forest knobs:
#   n_estimators     — number of trees. More = better but diminishing
#                      returns; mainly costs time.
#   max_depth        — how deep each tree may grow. Deeper = more complex
#                      patterns but more overfitting risk.
#   min_samples_leaf — minimum rows per leaf. Higher = smoother, more
#                      regularised trees.
#   max_features     — fraction of features each split may consider.
#                      Lower = more decorrelated trees (the "random" in
#                      Random Forest).
RF_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.3, 0.5, 0.8],
}

# XGBoost knobs:
#   n_estimators / learning_rate — a coupled pair: more trees with a
#                      smaller learning rate = slower, steadier learning.
#   max_depth        — per-tree complexity (boosted trees are kept shallow;
#                      3-8 is the usual range, unlike RF's deep trees).
#   subsample        — fraction of ROWS each tree sees.
#   colsample_bytree — fraction of FEATURES each tree sees. Both inject
#                      randomness that fights overfitting.
#   reg_alpha/lambda — L1/L2 penalties on leaf weights: XGBoost's own
#                      version of the Ridge/Lasso idea.
XGB_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 800, 1200],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6, 8],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0],
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    keep_default_na=False for the same reason as features.py: pandas'
    default reader treats the literal string "None" as NaN, silently
    undoing earlier cleaning on the CSV round trip. The featured file is
    fully numeric so the guard is belt-and-braces here, but consistency
    across the pipeline beats per-file cleverness.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    print(f"[load]     Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def split_data(df: pd.DataFrame):
    """
    Builds the one and only train/test wall.

    y is returned in BOTH forms: log (what models train on) and raw
    dollars (what metrics are computed against). The test set is not
    touched again until final_evaluation().
    """
    X = df.drop(columns=["SalePrice"])
    y_raw = df["SalePrice"]
    y_log = np.log1p(y_raw)

    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = (
        train_test_split(
            X, y_log, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    )

    print(f"[split]    Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows "
          f"(held out, touched once at the end).")
    return X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_searches() -> dict:
    """
    Returns {name: unfitted search object (or plain model for the
    baseline)}. Linear models are wrapped in Pipelines with a scaler;
    tree models are not (scale-invariant — see module docstring).
    """
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=RANDOM_STATE)),
    ])
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(random_state=RANDOM_STATE, max_iter=50000)),
    ])

    return {
        "Linear Regression": LinearRegression(),

        "Ridge": GridSearchCV(
            ridge_pipe, RIDGE_GRID,
            cv=K_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1,
        ),
        "Lasso": GridSearchCV(
            lasso_pipe, LASSO_GRID,
            cv=K_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1,
        ),
        "Random Forest": RandomizedSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            RF_DISTRIBUTIONS, n_iter=N_RANDOM_CANDIDATES,
            cv=K_FOLDS, scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": RandomizedSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                         objective="reg:squarederror"),
            XGB_DISTRIBUTIONS, n_iter=N_RANDOM_CANDIDATES,
            cv=K_FOLDS, scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_models(searches: dict, X_train, y_train_log) -> dict:
    """
    Fits every search on TRAINING data only. All cross-validation happens
    inside this call, inside the training 80%. Returns the fitted best
    estimator per model name.
    """
    fitted = {}
    for name, search in searches.items():
        print(f"[tune]     {name} ...")
        search.fit(X_train, y_train_log)

        if hasattr(search, "best_params_"):
            # CV score is in log space (and negated by sklearn convention)
            print(f"[tune]       best CV RMSE (log space): {-search.best_score_:.4f}")
            print(f"[tune]       best params: {search.best_params_}")
            fitted[name] = search.best_estimator_
        else:
            fitted[name] = search  # plain baseline, no search wrapper

    return fitted


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def dollar_metrics(y_true_raw, y_pred_dollars) -> dict:
    """All four reported metrics, computed in real dollars."""
    return {
        "RMSE ($)": np.sqrt(mean_squared_error(y_true_raw, y_pred_dollars)),
        "MAE ($)": mean_absolute_error(y_true_raw, y_pred_dollars),
        "MAPE (%)": mean_absolute_percentage_error(y_true_raw, y_pred_dollars) * 100,
        "R2": r2_score(y_true_raw, y_pred_dollars),
    }


def final_evaluation(fitted: dict, X_train, X_test,
                     y_train_raw, y_test_raw, y_test_log) -> pd.DataFrame:
    """
    The single visit to the held-out test set.

    Includes one extra row — 'Linear Regression (raw target)' — trained
    here on raw dollars, as the explicit log-vs-raw experiment: same
    model, same features, same split; the only difference is the target
    transformation. The delta in the table IS the argument for logging.
    """
    rows = []

    # The explicit experiment: baseline trained on RAW dollars
    raw_lr = LinearRegression().fit(X_train, y_train_raw)
    raw_pred = raw_lr.predict(X_test)
    rows.append({"Model": "Linear Regression (raw target)",
                 **dollar_metrics(y_test_raw, raw_pred)})

    # Everything else predicts in log space -> expm1 back to dollars
    for name, model in fitted.items():
        pred_log = model.predict(X_test)
        pred_dollars = np.expm1(pred_log)
        rows.append({"Model": f"{name} (log target)",
                     **dollar_metrics(y_test_raw, pred_dollars)})

    results = pd.DataFrame(rows)
    print("\n[results]  Held-out test set comparison:\n")
    print(results.round(3).to_string(index=False))
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def train_quantile_models(winner_name: str, winner_model, X_train, y_train_log):
    """
    For adaptive PREDICTION INTERVALS, train two extra XGBoost models that
    predict the 5th and 95th percentiles instead of the mean — using
    XGBoost's quantile loss (objective="reg:quantileerror"). The gap between
    them is a 90% interval that NATURALLY WIDENS for unusual houses and
    tightens for typical ones, because each is a full model learning from
    features (unlike a flat RMSE band, which would be the same width for
    every house).

    Only applies when the winner is XGBoost — quantile regression is an
    XGBoost capability here, and the other model types don't provide it the
    same way. Returns (lower_model, upper_model) or (None, None) if the
    winner isn't XGBoost, so the caller can degrade gracefully.

    Note on the log target: quantiles are preserved under monotonic
    transforms, so the 5th/95th percentile learned in log space maps
    correctly to the 5th/95th percentile in dollars via expm1 — no
    distortion. The quantile models train on the SAME log target and reuse
    the winner's tuned hyperparameters, changing only the objective.
    """
    if winner_name != "XGBoost":
        print(f"[interval] Winner is {winner_name}, not XGBoost — "
              f"intervals unavailable, saving point model only.")
        return None, None

    # Reuse the winner's tuned params, swap in the quantile objective.
    base_params = winner_model.get_params()
    base_params.pop("objective", None)  # we override this per quantile

    def make_q(alpha):
        m = XGBRegressor(**base_params,
                         objective="reg:quantileerror",
                         quantile_alpha=alpha)
        m.fit(X_train, y_train_log)
        return m

    print("[interval] Training 5th/95th percentile XGBoost models "
          "for adaptive 90% intervals ...")
    lower = make_q(0.05)
    upper = make_q(0.95)
    return lower, upper


def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"[cleanup]  Existing file removed: {path}")


def save_outputs(results: pd.DataFrame, fitted: dict,
                 X_train=None, y_train_log=None) -> None:
    """
    Winner = lowest test RMSE among the log-target models (the raw-target
    row is an experiment exhibit, not a contender). Saved via joblib —
    the standard for sklearn objects (handles numpy arrays better than
    raw pickle) — alongside a metadata dict so the .pkl is never a
    mystery file.

    If the winner is XGBoost, two extra quantile models are trained and
    bundled so predict.py can emit adaptive prediction intervals. If not,
    only the point model is saved and predict.py falls back to point
    estimates — the bundle's "has_intervals" flag tells it which.
    """
    os.makedirs("models", exist_ok=True)

    contenders = results[results["Model"].str.contains("log target")]
    winner_row = contenders.loc[contenders["RMSE ($)"].idxmin()]
    winner_name = winner_row["Model"].replace(" (log target)", "")
    winner_model = fitted[winner_name]

    # Train interval models (None, None if winner isn't XGBoost).
    lower_model, upper_model = train_quantile_models(
        winner_name, winner_model, X_train, y_train_log
    )
    has_intervals = lower_model is not None

    metadata = {
        "model_name": winner_name,
        "target_transform": "log1p (invert predictions with np.expm1)",
        "test_rmse_dollars": float(winner_row["RMSE ($)"]),
        "test_mae_dollars": float(winner_row["MAE ($)"]),
        "test_mape_pct": float(winner_row["MAPE (%)"]),
        "test_r2": float(winner_row["R2"]),
        "random_state": RANDOM_STATE,
        "has_intervals": has_intervals,
        "interval_kind": "quantile_models" if has_intervals else None,
        "interval_level": "90% (5th-95th percentile)" if has_intervals else None,
        "params": (winner_model.get_params() if not hasattr(winner_model, "steps")
                   else winner_model.named_steps["model"].get_params()),
    }

    bundle = {"model": winner_model, "metadata": metadata}
    if has_intervals:
        bundle["lower_model"] = lower_model
        bundle["upper_model"] = upper_model

    remove_if_exists(MODEL_PATH)
    joblib.dump(bundle, MODEL_PATH)
    interval_note = " + 90% intervals" if has_intervals else ""
    print(f"\n[save]     Winner: {winner_name}{interval_note} "
          f"(test RMSE ${winner_row['RMSE ($)']:,.0f}) -> {MODEL_PATH}")

    remove_if_exists(RESULTS_PATH)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"[save]     Results table -> {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    df = load_data(INPUT_PATH)
    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = (
        split_data(df)
    )

    searches = build_searches()
    fitted = tune_models(searches, X_train, y_train_log)

    results = final_evaluation(
        fitted, X_train, X_test, y_train_raw, y_test_raw, y_test_log
    )
    save_outputs(results, fitted, X_train=X_train, y_train_log=y_train_log)


if __name__ == "__main__":
    run()