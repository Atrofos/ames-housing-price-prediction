"""
compare_predictions.py
----------------------
Head-to-head comparison of the two model families' predictions: the
traditional gradient booster (XGBoost) versus the tabular foundation model
(TabPFN). Reads the prediction CSVs produced by the predict scripts and
produces console analysis plus saved charts.

The file has TWO sections, because the two comparisons answer different
questions:

  SECTION A — DEMO (real data, ~2,930 houses): about ACCURACY.
      These rows have true SalePrice, so we can ask "which model predicts
      real prices better?" — metrics, predicted-vs-actual scatters, and
      where the two models disagree most.

  SECTION B — TEST (synthetic ~50 houses): about BEHAVIOUR.
      These rows are hand-crafted, so their "actual" prices are a rough
      hand-rule, NOT ground truth — accuracy here is meaningless. Instead
      we probe how the two models BEHAVE: how closely they agree, how they
      handle deliberate outliers, what each thinks a single feature is
      worth (the pair experiments), and — if interval columns are present —
      whether they flag the same houses as uncertain.

Inputs (in data/processed/, produced by the predict scripts):
    predictions_demo.csv, predictions_demo_tabpfn.csv
    predictions_test.csv, predictions_test_tabpfn.csv
Outputs:
    analysis/*.png  (charts)  + console summary

Usage:
    python compare_predictions.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # file output, no interactive display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROC = os.path.join("data", "processed")
OUT_DIR = "analysis"

DEMO_X = os.path.join(PROC, "predictions_demo.csv")
DEMO_T = os.path.join(PROC, "predictions_demo_tabpfn.csv")
TEST_X = os.path.join(PROC, "predictions_test.csv")
TEST_T = os.path.join(PROC, "predictions_test_tabpfn.csv")

# Segment boundaries in the synthetic test set (by Order). Mirrors
# generate_test_data.py: spread first, then 8 outliers, then 12 pair-houses.
# Resolved dynamically below from the actual row ordering, so this is only
# a fallback description.
N_OUTLIERS = 8
N_PAIRS = 12
PAIR_NAMES = ["+ Fireplace", "- Central Air", "+ Finished Basement",
              "+ Extra Bathroom", "Nicer Neighbourhood", "+1 Overall Quality"]

# Consistent colours for the two models across all charts.
C_XGB = "#D85A30"   # warm orange
C_TAB = "#1D9E75"   # green

# What each synthetic segment IS — printed in Section B so the comparison is
# self-documenting (mirrors generate_test_data.py's three segments).
SEGMENT_EXPLAINER = """\
  The synthetic test set is built in THREE purpose-designed segments:

  SPREAD  — an even sweep of ordinary houses from budget to luxury, with
            quality/size/age/finish scaled together. Purpose: see how each
            model behaves across the WHOLE price range (not just on average).

  OUTLIER — 8 deliberately unusual houses, each probing a different EDGE
            CASE the training data barely contains, to test extrapolation:
              1. Pool + everything maxed (very rare)
              2. Tiny, ancient, damaged, by a railway, no central air
              3. Huge lot, modest house (land-heavy)
              4. Brand new, sold same year (age 0), partial sale
              5. Old but immaculately kept (age/condition mismatch)
              6. Massive house, mediocre quality (size/quality mismatch)
              7. Luxury but tiny (quality without space)
              8. Commercial-zoned oddity

  PAIR    — 6 controlled experiments = 12 houses. Each pair is identical
            except for ONE feature, so the difference in predicted price is
            what that model thinks the feature is WORTH. The six features:
            fireplace, central air, finished basement, extra bathroom,
            nicer neighbourhood, and +1 overall quality.
"""


def _load(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _metrics(df):
    """MAE, MAPE, and (vs actual) a simple R² — for the demo accuracy view."""
    err = df["PredictedPrice"] - df["ActualPrice"]
    mae = err.abs().mean()
    mape = (err.abs() / df["ActualPrice"]).mean() * 100
    ss_res = (err ** 2).sum()
    ss_tot = ((df["ActualPrice"] - df["ActualPrice"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return mae, mape, r2


# ---------------------------------------------------------------------------
# SECTION A — DEMO: accuracy comparison on real data
# ---------------------------------------------------------------------------

def section_demo(demo_x, demo_t):
    print("\n" + "=" * 64)
    print("  SECTION A — DEMO (real data): ACCURACY")
    print("=" * 64)

    if demo_x is None or demo_t is None:
        print("  Demo prediction files missing — skipping.")
        return

    print("""
  DEMO = the real AmesHousing dataset scored through both models. Because
  these rows have TRUE sale prices, we can measure accuracy and compare the
  two models head-to-head on identical, real-world input.
""")

    mae_x, mape_x, r2_x = _metrics(demo_x)
    mae_t, mape_t, r2_t = _metrics(demo_t)

    print(f"\n  {'Metric':<14}{'XGBoost':>14}{'TabPFN':>14}{'Winner':>12}")
    print("  " + "-" * 52)
    for name, vx, vt, lower_better in [
        ("MAE ($)", mae_x, mae_t, True),
        ("MAPE (%)", mape_x, mape_t, True),
        ("R2", r2_x, r2_t, False),
    ]:
        win = ("TabPFN" if (vt < vx) == lower_better else "XGBoost")
        if name == "R2":
            print(f"  {name:<14}{vx:>14.3f}{vt:>14.3f}{win:>12}")
        else:
            print(f"  {name:<14}{vx:>14,.0f}{vt:>14,.0f}{win:>12}")

    print("\n  Note: these are scored on TRAINING data, so both look better")
    print("  than reality — the honest numbers are the held-out test metrics")
    print("  in model.py. This section is for COMPARING the two models on")
    print("  identical input, which remains valid.")

    # --- Chart 1: predicted vs actual, side by side ---
    merged = demo_x[["Order", "PredictedPrice", "ActualPrice"]].merge(
        demo_t[["Order", "PredictedPrice"]], on="Order", suffixes=("_xgb", "_tab")
    )
    lim = [0, max(merged["ActualPrice"].max(),
                  merged["PredictedPrice_xgb"].max()) * 1.05]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, col, colour, title in [
        (axes[0], "PredictedPrice_xgb", C_XGB, "XGBoost"),
        (axes[1], "PredictedPrice_tab", C_TAB, "TabPFN"),
    ]:
        ax.scatter(merged["ActualPrice"], merged[col], s=6, alpha=0.3,
                   color=colour, edgecolors="none")
        ax.plot(lim, lim, "--", color="#444", lw=1)  # perfect-prediction line
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Actual price ($)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.ticklabel_format(style="plain")
    axes[0].set_ylabel("Predicted price ($)")
    fig.suptitle("Predicted vs Actual — tighter to the dashed line is better",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "demo_predicted_vs_actual.png", explanation="""
WHAT: each dot is a house — its true price (x) vs the model's prediction (y).
LOOK: dots hugging the dashed line = accurate. Both models track the line
      tightly; notice the scatter widens at the high-price end (right), where
      expensive houses are rarer and harder to predict for both models.""")

    # --- Chart 2: where the two models disagree most ---
    merged["diff"] = merged["PredictedPrice_xgb"] - merged["PredictedPrice_tab"]
    corr = merged["PredictedPrice_xgb"].corr(merged["PredictedPrice_tab"])
    print(f"\n  Prediction agreement (correlation): {corr:.3f}")
    print(f"  Mean absolute gap between the models: "
          f"${merged['diff'].abs().mean():,.0f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(merged["diff"], bins=60, color="#5B8DB8", edgecolor="white", lw=0.3)
    ax.axvline(0, color="#444", ls="--", lw=1)
    ax.set_title("Where the models disagree\n(XGBoost minus TabPFN, per house)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Prediction difference ($)")
    ax.set_ylabel("Number of houses")
    fig.tight_layout()
    _save(fig, "demo_disagreement.png", explanation="""
WHAT: for every house, XGBoost's prediction minus TabPFN's. Centred at 0
      (dashed line) = the two models mostly agree.
LOOK: a tall narrow peak at 0 means strong agreement; the few houses in the
      tails are where the models price most differently.""")


# ---------------------------------------------------------------------------
# SECTION B — TEST: behaviour comparison on synthetic data
# ---------------------------------------------------------------------------

def section_test(test_x, test_t):
    print("\n" + "=" * 64)
    print("  SECTION B — TEST (synthetic data): BEHAVIOUR")
    print("=" * 64)

    if test_x is None or test_t is None:
        print("  Test prediction files missing — skipping.")
        return

    print()
    print(SEGMENT_EXPLAINER)

    m = test_x.merge(test_t[["Order", "PredictedPrice"]],
                     on="Order", suffixes=("_xgb", "_tab"))
    m = m.sort_values("Order").reset_index(drop=True)

    # Resolve segments by position: last N_PAIRS are pairs, the N_OUTLIERS
    # before them are outliers, the rest are the price spread.
    n = len(m)
    seg = np.array(["spread"] * n, dtype=object)
    seg[n - N_PAIRS:] = "pair"
    seg[n - N_PAIRS - N_OUTLIERS: n - N_PAIRS] = "outlier"
    m["segment"] = seg

    corr = m["PredictedPrice_xgb"].corr(m["PredictedPrice_tab"])
    print(f"\n  Overall prediction agreement (correlation): {corr:.3f}")
    print(f"  Mean absolute gap: ${(m['PredictedPrice_xgb'] - m['PredictedPrice_tab']).abs().mean():,.0f}")
    print("  (Synthetic 'actual' prices are a hand-rule, so we compare the")
    print("   models to EACH OTHER and study behaviour, not accuracy.)")

    # --- Outlier divergence ---
    print("\n  --- OUTLIERS: where the families diverge (extrapolation) ---")
    out = m[m["segment"] == "outlier"]
    print(f"  {'Order':>6}{'XGBoost':>12}{'TabPFN':>12}{'Gap':>12}")
    for _, r in out.iterrows():
        gap = r["PredictedPrice_xgb"] - r["PredictedPrice_tab"]
        print(f"  {int(r['Order']):>6}{r['PredictedPrice_xgb']:>12,.0f}"
              f"{r['PredictedPrice_tab']:>12,.0f}{gap:>+12,.0f}")

    # --- Pair experiments ---
    print("\n  --- PAIR EXPERIMENTS: implied value of each single feature ---")
    pairs = m[m["segment"] == "pair"].reset_index(drop=True)
    print(f"  {'Feature change':<22}{'XGBoost':>12}{'TabPFN':>12}")
    pair_rows = []
    for i, name in enumerate(PAIR_NAMES):
        if 2 * i + 1 >= len(pairs):
            break
        a, b = pairs.iloc[2 * i], pairs.iloc[2 * i + 1]
        dx = b["PredictedPrice_xgb"] - a["PredictedPrice_xgb"]
        dt = b["PredictedPrice_tab"] - a["PredictedPrice_tab"]
        pair_rows.append((name, dx, dt))
        print(f"  {name:<22}{dx:>+12,.0f}{dt:>+12,.0f}")

    # --- Chart 3: pair-experiment feature values, grouped bars ---
    if pair_rows:
        labels = [r[0] for r in pair_rows]
        xgb_vals = [r[1] for r in pair_rows]
        tab_vals = [r[2] for r in pair_rows]
        y = np.arange(len(labels))
        h = 0.38
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(y + h / 2, xgb_vals, height=h, color=C_XGB, label="XGBoost")
        ax.barh(y - h / 2, tab_vals, height=h, color=C_TAB, label="TabPFN")
        ax.axvline(0, color="#444", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("Implied change in predicted price ($)")
        ax.set_title("What is each feature worth?\nPair experiments: one feature changed, "
                     "rest held constant", fontsize=13, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        _save(fig, "test_feature_values.png", explanation="""
WHAT: each bar pair = how much each model's prediction changed when ONE
      feature was added/removed (rest held identical). The bar length is the
      feature's implied dollar value to that model.
LOOK: bars on the same side = models agree on direction. Different lengths =
      they weight the feature differently (e.g. XGBoost values a quality bump
      more; TabPFN values a finished basement and nicer area more).""")

    # --- Chart 4: agreement scatter across all test houses ---
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colours = {"spread": "#5B8DB8", "outlier": "#D85A30", "pair": "#9B7BB8"}
    for s, c in colours.items():
        sub = m[m["segment"] == s]
        ax.scatter(sub["PredictedPrice_xgb"], sub["PredictedPrice_tab"],
                   s=40, alpha=0.8, color=c, label=s, edgecolors="white", lw=0.5)
    lim = [0, m[["PredictedPrice_xgb", "PredictedPrice_tab"]].max().max() * 1.05]
    ax.plot(lim, lim, "--", color="#444", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("XGBoost predicted ($)")
    ax.set_ylabel("TabPFN predicted ($)")
    ax.set_title("Do the two models agree?\nPoints near the line = agreement",
                 fontsize=13, fontweight="bold")
    ax.legend(title="segment")
    fig.tight_layout()
    _save(fig, "test_model_agreement.png", explanation="""
WHAT: each synthetic house plotted as XGBoost's price (x) vs TabPFN's (y),
      coloured by segment.
LOOK: points on the diagonal = the two models agree. Points far off the line
      are where they disagree — typically the orange OUTLIER points (e.g. the
      pool house), showing the two model families extrapolate differently.""")

    # --- Uncertainty agreement, only if BOTH have interval columns ---
    if "IntervalWidth" in test_x.columns and "IntervalWidth" in test_t.columns:
        print("\n  --- UNCERTAINTY AGREEMENT: do both widen on the same houses? ---")
        mw = test_x[["Order", "IntervalWidth"]].merge(
            test_t[["Order", "IntervalWidth"]], on="Order", suffixes=("_xgb", "_tab"))
        wcorr = mw["IntervalWidth_xgb"].corr(mw["IntervalWidth_tab"])
        print(f"  Interval-width correlation: {wcorr:.3f}")
        print("  (High = the two DIFFERENT uncertainty mechanisms independently")
        print("   flag the same houses as hard — strong cross-validation.)")

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(mw["IntervalWidth_xgb"], mw["IntervalWidth_tab"],
                   s=40, alpha=0.75, color="#9B7BB8", edgecolors="white", lw=0.5)
        ax.set_xlabel("XGBoost interval width ($)")
        ax.set_ylabel("TabPFN interval width ($)")
        ax.set_title("Do the models agree on WHICH houses are uncertain?",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        _save(fig, "test_uncertainty_agreement.png", explanation="""
WHAT: each house plotted by how WIDE its prediction interval is under each
      model — XGBoost width (x) vs TabPFN width (y).
LOOK: an upward trend means both models — using completely DIFFERENT
      uncertainty mechanisms — flag the same houses as hard to price. That
      independent agreement is strong evidence the uncertainty is real, not
      an artefact of either method.""")
    else:
        print("\n  --- UNCERTAINTY AGREEMENT: skipped ---")
        print("  TabPFN predictions have no interval columns. Re-run")
        print("  model_tabpfn.py with the interval-flagged version so its")
        print("  saved bundle carries interval_kind='tabpfn_native', then")
        print("  re-score — and this comparison will appear.")


# ---------------------------------------------------------------------------
# Helpers + entry point
# ---------------------------------------------------------------------------

def _save(fig, name, explanation=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [chart]  saved {path}")
    if explanation:
        for line in explanation.strip("\n").split("\n"):
            print(f"           {line}")


def run():
    section_demo(_load(DEMO_X), _load(DEMO_T))
    section_test(_load(TEST_X), _load(TEST_T))
    print("\n" + "=" * 64)
    print(f"  Done. Charts saved in ./{OUT_DIR}/")
    print("=" * 64)


if __name__ == "__main__":
    run()