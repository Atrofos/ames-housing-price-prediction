"""
explain_model.py
----------------
Model interpretability with SHAP (SHapley Additive exPlanations).

The trained models predict a price, but on their own they don't tell you WHY.
SHAP opens the box: for any prediction it attributes the result across the
features, so you can see that (for example) a prediction started near the
dataset average and was then pushed up by high quality, up again by large
living area, down a little by age, and so on.

This script produces three kinds of output:

  GLOBAL  — which features drive the model overall, across all houses.
            The beeswarm summary plot and a simpler mean-importance bar plot.

  LOCAL   — why the model made one specific prediction. Waterfall plots for a
            few hand-picked houses (a typical one and some deliberate
            outliers), showing how each feature moved that single prediction.

  DEPENDENCE — how the single most important feature's effect varies across
            its range (and where interactions show up).

It explains the saved XGBoost model (`models/best_model.pkl`), so no retraining
is needed. SHAP's TreeExplainer is exact and fast for tree models.

IMPORTANT — units. The model is trained on log1p(price), so SHAP values here are
in LOG space, not dollars. They still rank and compare features correctly (a
larger SHAP value means a larger push on the prediction), but read them as
"relative pushes", not dollar amounts. This is standard and fine for
interpretability; it's just worth stating so the numbers aren't misread.

Usage:
    python explain_model.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("models", "best_model.pkl")
FEATURED_PATH = os.path.join("data", "processed", "ames_featured.csv")
OUT_DIR = "analysis"
TARGET = "SalePrice"
RANDOM_STATE = 42

# How many houses to explain individually, and a background sample size for
# the beeswarm (a few hundred points is plenty and keeps the plot readable).
N_SUMMARY_SAMPLE = 500


def _save(fig, name, explanation=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart]  saved {path}")
    if explanation:
        for line in explanation.strip("\n").split("\n"):
            print(f"           {line}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run model_training.py first.")
    if not os.path.exists(FEATURED_PATH):
        raise FileNotFoundError(f"No featured data at {FEATURED_PATH}. Run feature_selection.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    if type(model).__name__ != "XGBRegressor":
        print(f"  [warn] Saved model is {type(model).__name__}, not XGBoost. "
              f"SHAP's TreeExplainer expects a tree model; results may vary.")

    df = pd.read_csv(FEATURED_PATH)
    X = df.drop(columns=[TARGET])
    # Align to exactly the columns the model was trained on, in order.
    X = X[list(model.feature_names_in_)]
    y = df[TARGET]
    return model, X, y


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(model, X):
    print("  Computing SHAP values (TreeExplainer, exact for tree models) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    print(f"  Done. SHAP values shape: {shap_values.values.shape}")
    return explainer, shap_values


def global_plots(shap_values, X):
    print("\n  GLOBAL — what drives the model overall:")

    # Beeswarm summary: each dot is a house; colour is the feature value.
    # Sample for readability if the dataset is large.
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), size=min(N_SUMMARY_SAMPLE, len(X)), replace=False)
    sv_sample = shap_values[idx]

    fig = plt.figure()
    shap.plots.beeswarm(sv_sample, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary: feature impact on predictions (log space)",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "shap_summary_beeswarm.png", explanation="""
WHAT: each dot is a house. Features are ranked top-to-bottom by overall
      importance. A dot's horizontal position is how much that feature pushed
      that house's prediction; colour is the feature's value (red high, blue
      low).
LOOK: for a feature like Overall Qual, red dots on the right mean "high quality
      pushes the price up". The spread of each row shows how much that feature
      swings predictions across houses.""")

    # Mean absolute SHAP: a clean importance ranking (good for the README).
    fig = plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("Mean feature importance (average absolute SHAP)",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "shap_importance_bar.png", explanation="""
WHAT: features ranked by their average absolute SHAP value, i.e. how much each
      one moves the prediction on average across all houses.
LOOK: this is the clean "which features matter most" ranking. Compare it to the
      correlations from the EDA: SHAP accounts for the model's actual use of
      each feature, including interactions, so the ranking can differ.""")

    # Print the top-10 ranking as text too.
    importance = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    print("\n  Top 10 features by mean absolute SHAP value:")
    for i in order[:10]:
        print(f"    {X.columns[i]:<22} {importance[i]:.4f}")


def local_plots(explainer, shap_values, X, y, model):
    print("\n  LOCAL — why individual predictions came out as they did:")

    preds_log = model.predict(X)
    preds = np.expm1(preds_log)

    # Pick interesting houses to explain:
    #   - a typical, mid-priced, well-predicted house
    #   - the most expensive house (a high-end extrapolation case)
    #   - the house the model most UNDER-predicted (biggest miss low)
    err = preds - y.values
    picks = {
        "typical": int(np.argmin(np.abs(preds - np.median(preds)))),
        "most_expensive": int(y.values.argmax()),
        "biggest_underprediction": int(err.argmin()),
    }

    for label, i in picks.items():
        fig = plt.figure()
        shap.plots.waterfall(shap_values[i], max_display=12, show=False)
        fig = plt.gcf()
        actual = y.values[i]
        pred = preds[i]
        fig.suptitle(f"Why this prediction? ({label.replace('_',' ')})\n"
                     f"predicted ${pred:,.0f} vs actual ${actual:,.0f}",
                     fontsize=11, fontweight="bold", y=1.02)
        _save(fig, f"shap_local_{label}.png", explanation=f"""
WHAT: the {label.replace('_',' ')} house. Starts from the dataset's average
      prediction (bottom) and each feature pushes it up (red) or down (blue) to
      the final value at the top.
LOOK: the longest bars are the features that mattered most for THIS house. This
      is how you'd explain a single prediction to someone who asks "why that
      price?".""")


def dependence_plot(shap_values, X):
    print("\n  DEPENDENCE — how the top feature's effect varies:")

    importance = np.abs(shap_values.values).mean(axis=0)
    top_feature = X.columns[int(np.argmax(importance))]

    fig = plt.figure()
    shap.plots.scatter(shap_values[:, top_feature], show=False)
    fig = plt.gcf()
    fig.suptitle(f"Dependence: how {top_feature} affects predictions",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "shap_dependence_top.png", explanation=f"""
WHAT: for the single most important feature ({top_feature}), each dot is a
      house: its {top_feature} value (x) vs how much that feature pushed the
      prediction (y).
LOOK: the trend shows the feature's effect across its range. A rising trend
      means higher values push the price up more. Vertical spread at a given x
      hints at interactions with other features.""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("=" * 64)
    print("  SHAP MODEL EXPLANATION")
    print("=" * 64)
    print("\n  Note: the model predicts log1p(price), so SHAP values are in LOG")
    print("  space. They rank and compare features correctly, but are relative")
    print("  pushes, not dollar amounts.\n")

    model, X, y = load()
    explainer, shap_values = compute_shap(model, X)

    global_plots(shap_values, X)
    local_plots(explainer, shap_values, X, y, model)
    dependence_plot(shap_values, X)

    print("\n" + "=" * 64)
    print(f"  Done. SHAP charts saved in ./{OUT_DIR}/")
    print("=" * 64)


if __name__ == "__main__":
    run()
