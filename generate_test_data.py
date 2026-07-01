"""
generate_test_data.py
----------------------
Generates a synthetic test set of houses in the AmesHousing.csv schema, for
probing the trained models (predict.py / predict_tabpfn.py).

Why generate rather than sample real rows? Sampling from AmesHousing.csv
would re-use TRAINING data, so predictions would look flatteringly good and
wouldn't be a real test. These houses are novel — built from realistic value
ranges — so scoring them genuinely exercises the models on unseen inputs.

The set is deliberately structured into THREE segments, because a good test
set is about COVERAGE and EXPERIMENT DESIGN, not volume:

  A) SPREAD  (~30 houses) — an even sweep across price tiers (budget ->
     luxury), built by scaling quality, size, age and finish together. Lets
     you ask "does prediction error grow with price?" and see each model's
     accuracy across the market, not just on average.

  B) OUTLIERS (~8 houses) — deliberately unusual: pools, huge lots, 4-car
     garages, functional damage, railway-adjacent, very old. Probes
     EXTRAPOLATION — how each model behaves outside the dense middle of the
     training distribution (where models are known to struggle).

  C) PAIRS (~12 houses = 6 pairs) — two otherwise-identical houses differing
     in exactly ONE feature (a fireplace, a garage, central air, a finished
     basement, an extra bathroom, a better neighbourhood). This turns the
     test set into a little EXPERIMENT: compare the two predictions and you
     can read off how much each model thinks that single feature is worth.
     The most writeup-worthy segment — it reveals model *behaviour*, not just
     accuracy.

Reproducible: a fixed SEED means the same file every run. Change N_SPREAD /
the seed / the segments to make your own variants.

Output: AmesHousing_test.csv  (in AmesHousing.csv schema, with SalePrice so
        predict.py reports predicted-vs-actual; SalePrice here is a rough
        hand-rule "plausible" price, NOT ground truth — see note below.)

Usage:
    python generate_test_data.py
"""

import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REFERENCE_CSV = os.path.join("data", "raw", "AmesHousing.csv")  # for schema + column order
OUTPUT_CSV = "AmesHousing_test.csv"

SEED = 42

# ---------------------------------------------------------------------------
# How many test houses to generate. The set has three segments:
#   - SPREAD   : scalable sweep across price tiers (the bulk)
#   - OUTLIERS : 8 hand-designed edge cases (fixed — each probes a specific
#                failure mode, so they don't meaningfully "scale")
#   - PAIRS    : 6 hand-designed one-feature-difference experiments = 12
#                houses (fixed — each isolates a specific feature's value)
#
# TOTAL_HOUSES is an APPROXIMATE target. The outliers and pairs are fixed
# sets, so we scale the SPREAD segment to make up the rest. If the requested
# total is smaller than the fixed segments, spread gets a sensible minimum.
# The actual count produced is printed at the end (it may differ by a couple
# from the request because pairs come in 2s and outliers are a fixed 8).
# ---------------------------------------------------------------------------

TOTAL_HOUSES = 50            # <-- change this to make a bigger/smaller set

N_OUTLIERS = 8               # fixed: the 8 hand-designed edge cases
N_PAIRS = 12                 # fixed: 6 experiments x 2 houses
# Spread fills the remainder, with a floor so the price sweep stays useful.
N_SPREAD = max(10, TOTAL_HOUSES - N_OUTLIERS - N_PAIRS)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# A "base" house — a complete, valid Ames row we mutate to build everything.
# Every field is a realistic value in the dataset's vocabulary. Segments A/B/C
# all start from a copy of this and change only what they need to.
# ---------------------------------------------------------------------------

BASE_HOUSE = {
    "Order": 0, "PID": 900000000, "MS SubClass": 20, "MS Zoning": "RL",
    "Lot Frontage": 70, "Lot Area": 9000, "Street": "Pave", "Alley": "NA",
    "Lot Shape": "Reg", "Land Contour": "Lvl", "Utilities": "AllPub",
    "Lot Config": "Inside", "Land Slope": "Gtl", "Neighborhood": "NAmes",
    "Condition 1": "Norm", "Condition 2": "Norm", "Bldg Type": "1Fam",
    "House Style": "1Story", "Overall Qual": 5, "Overall Cond": 5,
    "Year Built": 1975, "Year Remod/Add": 1975, "Roof Style": "Gable",
    "Roof Matl": "CompShg", "Exterior 1st": "VinylSd", "Exterior 2nd": "VinylSd",
    "Mas Vnr Type": "None", "Mas Vnr Area": 0, "Exter Qual": "TA",
    "Exter Cond": "TA", "Foundation": "CBlock", "Bsmt Qual": "TA",
    "Bsmt Cond": "TA", "Bsmt Exposure": "No", "BsmtFin Type 1": "Unf",
    "BsmtFin SF 1": 0, "BsmtFin Type 2": "Unf", "BsmtFin SF 2": 0,
    "Bsmt Unf SF": 800, "Total Bsmt SF": 800, "Heating": "GasA",
    "Heating QC": "TA", "Central Air": "Y", "Electrical": "SBrkr",
    "1st Flr SF": 1100, "2nd Flr SF": 0, "Low Qual Fin SF": 0,
    "Gr Liv Area": 1100, "Bsmt Full Bath": 0, "Bsmt Half Bath": 0,
    "Full Bath": 1, "Half Bath": 0, "Bedroom AbvGr": 3, "Kitchen AbvGr": 1,
    "Kitchen Qual": "TA", "TotRms AbvGrd": 6, "Functional": "Typ",
    "Fireplaces": 0, "Fireplace Qu": "NA", "Garage Type": "Attchd",
    "Garage Yr Blt": 1975, "Garage Finish": "Unf", "Garage Cars": 1,
    "Garage Area": 300, "Garage Qual": "TA", "Garage Cond": "TA",
    "Paved Drive": "Y", "Wood Deck SF": 0, "Open Porch SF": 0,
    "Enclosed Porch": 0, "3Ssn Porch": 0, "Screen Porch": 0, "Pool Area": 0,
    "Pool QC": "NA", "Fence": "NA", "Misc Feature": "NA", "Misc Val": 0,
    "Mo Sold": 6, "Yr Sold": 2009, "Sale Type": "WD ",
    "Sale Condition": "Normal", "SalePrice": 150000,
}


def new_house(order: int, **overrides) -> dict:
    """Copy the base house, apply overrides, set its Order/PID."""
    h = dict(BASE_HOUSE)
    h.update(overrides)
    h["Order"] = 4000 + order
    h["PID"] = 900000000 + order
    return h


def estimate_price(h: dict) -> int:
    """
    A transparent hand-rule to attach a *plausible* SalePrice to each
    synthetic house, so predict.py can show predicted-vs-actual.

    IMPORTANT: this is NOT ground truth — it's a rough heuristic so the
    output has something to compare against. The models were trained on the
    real market, not this rule, so treat the "Error" columns as indicative,
    not authoritative. The rule is intentionally simple and multiplicative
    (matching how housing value really works).
    """
    price = 30000  # base
    price += h["Gr Liv Area"] * 70          # ~$70 per above-grade sq ft
    price += h["Total Bsmt SF"] * 25         # basement worth less per sq ft
    price += (h["Overall Qual"] - 5) * 18000  # quality swing around the mean
    price += h["Fireplaces"] * 4000
    price += h["Garage Cars"] * 6000
    price += (2010 - h["Year Built"]) * -250  # older = cheaper
    price += h["Pool Area"] * 30
    if h["Central Air"] == "N":
        price -= 8000
    if h["Functional"] != "Typ":
        price -= 25000
    if h["Neighborhood"] in ("NoRidge", "StoneBr", "NridgHt"):
        price *= 1.25
    if h["Condition 1"] in ("Artery", "Feedr", "RRAn", "RRNn"):
        price *= 0.92
    return int(max(price, 40000))


# ---------------------------------------------------------------------------
# Segment A — spread across price tiers
# ---------------------------------------------------------------------------

def build_spread(n: int) -> list:
    """
    Sweep a 'tier' parameter from 0 (budget) to 1 (luxury), scaling the
    features that drive price together so we get an even, realistic spread
    rather than random noise.
    """
    houses = []
    neighbourhoods_by_tier = ["OldTown", "NAmes", "Gilbert", "Somerst", "NoRidge"]

    for i in range(n):
        t = i / (n - 1)  # 0..1

        qual = int(round(3 + t * 6))                 # 3 -> 9
        gr_liv = int(round(900 + t * 2600))          # 900 -> 3500
        bsmt = int(round(500 + t * 1500))            # 500 -> 2000
        year = int(round(1940 + t * 68))             # 1940 -> 2008
        cars = int(round(1 + t * 2))                 # 1 -> 3
        fireplaces = int(round(t * 2))               # 0 -> 2
        full_bath = int(round(1 + t * 2))            # 1 -> 3
        two_story = t > 0.5
        nbhd = neighbourhoods_by_tier[min(int(t * len(neighbourhoods_by_tier)),
                                          len(neighbourhoods_by_tier) - 1)]

        first_flr = gr_liv if not two_story else int(gr_liv * 0.55)
        second_flr = 0 if not two_story else gr_liv - first_flr

        h = new_house(i)
        # assign properly (keys with spaces can't be kwargs)
        h["Overall Qual"] = qual
        h["Gr Liv Area"] = gr_liv
        h["1st Flr SF"] = first_flr
        h["2nd Flr SF"] = second_flr
        h["House Style"] = "2Story" if two_story else "1Story"
        h["Total Bsmt SF"] = bsmt
        h["Bsmt Unf SF"] = bsmt
        h["Year Built"] = year
        h["Year Remod/Add"] = year
        h["Garage Yr Blt"] = year
        h["Garage Cars"] = cars
        h["Garage Area"] = 250 + cars * 130
        h["Fireplaces"] = fireplaces
        h["Fireplace Qu"] = "Gd" if fireplaces > 0 else "NA"
        h["Full Bath"] = full_bath
        h["TotRms AbvGrd"] = 5 + int(t * 5)
        h["Bedroom AbvGr"] = 2 + int(t * 2)
        h["Neighborhood"] = nbhd
        h["Kitchen Qual"] = "Ex" if qual >= 8 else ("Gd" if qual >= 6 else "TA")
        h["Exter Qual"] = "Gd" if qual >= 7 else "TA"
        h["Bsmt Qual"] = "Gd" if qual >= 7 else "TA"
        h["SalePrice"] = estimate_price(h)
        houses.append(h)

    return houses


# ---------------------------------------------------------------------------
# Segment B — deliberate outliers (extrapolation probes)
# ---------------------------------------------------------------------------

def build_outliers(start_order: int) -> list:
    houses = []
    o = start_order

    # 1. Pool + everything maxed (very rare in training)
    h = new_house(o, **{}); o += 1
    h.update({"Overall Qual": 10, "Overall Cond": 9, "Gr Liv Area": 3800,
              "1st Flr SF": 3800, "Total Bsmt SF": 2400, "Bsmt Unf SF": 200,
              "BsmtFin Type 1": "GLQ", "BsmtFin SF 1": 2200, "Year Built": 2010,
              "Year Remod/Add": 2010, "Garage Yr Blt": 2010, "Garage Cars": 4,
              "Garage Area": 1200, "Garage Finish": "Fin", "Fireplaces": 2,
              "Fireplace Qu": "Ex", "Pool Area": 600, "Pool QC": "Ex",
              "Neighborhood": "NoRidge", "Kitchen Qual": "Ex", "Exter Qual": "Ex",
              "Bsmt Qual": "Ex", "Full Bath": 3, "Lot Area": 35000,
              "Mas Vnr Type": "Stone", "Mas Vnr Area": 600})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    # 2. Tiny, ancient, damaged, by a railway, no central air
    h = new_house(o); o += 1
    h.update({"Overall Qual": 3, "Gr Liv Area": 750, "1st Flr SF": 750,
              "Total Bsmt SF": 400, "Bsmt Unf SF": 400, "Year Built": 1900,
              "Year Remod/Add": 1900, "Garage Type": "NA", "Garage Yr Blt": "NA",
              "Garage Finish": "NA", "Garage Cars": 0, "Garage Area": 0,
              "Garage Qual": "NA", "Garage Cond": "NA", "Central Air": "N",
              "Functional": "Maj1", "Condition 1": "RRAn", "Neighborhood": "IDOTRR",
              "Electrical": "FuseA", "Heating QC": "Fa"})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    # 3. Huge lot, modest house (land-heavy)
    h = new_house(o); o += 1
    h.update({"Lot Area": 50000, "Lot Frontage": 200, "Overall Qual": 5,
              "Gr Liv Area": 1200, "1st Flr SF": 1200, "Neighborhood": "ClearCr"})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    # 4. Brand new, sold same year (HouseAge 0), partial sale
    h = new_house(o); o += 1
    h.update({"Year Built": 2010, "Year Remod/Add": 2010, "Yr Sold": 2010,
              "Garage Yr Blt": 2010, "Overall Qual": 8, "Gr Liv Area": 2200,
              "1st Flr SF": 2200, "Sale Type": "New", "Sale Condition": "Partial",
              "Kitchen Qual": "Gd", "Neighborhood": "NridgHt"})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    # 5. Old but immaculately kept (high cond, low age signal mismatch)
    h = new_house(o); o += 1
    h.update({"Year Built": 1925, "Year Remod/Add": 2008, "Overall Qual": 7,
              "Overall Cond": 9, "Gr Liv Area": 1800, "1st Flr SF": 900,
              "2nd Flr SF": 900, "House Style": "2Story", "Kitchen Qual": "Gd"})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    # 6. Massive house, mediocre quality (size/quality mismatch)
    h = new_house(o); o += 1
    h.update({"Gr Liv Area": 3500, "1st Flr SF": 1800, "2nd Flr SF": 1700,
              "House Style": "2Story", "Overall Qual": 4, "Total Bsmt SF": 1800,
              "Bsmt Unf SF": 1800})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    # 7. Luxury but tiny (quality without space)
    h = new_house(o); o += 1
    h.update({"Overall Qual": 9, "Gr Liv Area": 1000, "1st Flr SF": 1000,
              "Kitchen Qual": "Ex", "Exter Qual": "Ex", "Bsmt Qual": "Ex",
              "Neighborhood": "StoneBr"})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    # 8. Commercial-zoned oddity
    h = new_house(o); o += 1
    h.update({"MS Zoning": "C (all)", "Overall Qual": 4, "Gr Liv Area": 1300,
              "1st Flr SF": 1300, "Condition 1": "Feedr", "Neighborhood": "IDOTRR"})
    h["SalePrice"] = estimate_price(h); houses.append(h)

    return houses


# ---------------------------------------------------------------------------
# Segment C — controlled pairs (one-feature-difference experiments)
# ---------------------------------------------------------------------------

def build_pairs(start_order: int) -> list:
    """
    Each pair shares a baseline; the second house changes exactly ONE thing.
    Comparing the two predictions isolates how much each model values that
    feature. The shared baseline is a mid-market house so effects are
    measured in the dense, reliable part of the model's range.
    """
    houses = []
    o = start_order

    def pair(baseline_changes: dict, feature_change: dict):
        nonlocal o
        a = new_house(o); a.update(baseline_changes); o += 1
        a["SalePrice"] = estimate_price(a)
        b = new_house(o); b.update(baseline_changes); b.update(feature_change); o += 1
        b["SalePrice"] = estimate_price(b)
        houses.extend([a, b])

    mid = {"Overall Qual": 6, "Gr Liv Area": 1600, "1st Flr SF": 1600,
           "Total Bsmt SF": 1000, "Bsmt Unf SF": 1000, "Year Built": 2000,
           "Year Remod/Add": 2000, "Garage Yr Blt": 2000, "Garage Cars": 2,
           "Garage Area": 480, "Neighborhood": "Gilbert"}

    # Pair 1 — +/- a fireplace
    pair(mid, {"Fireplaces": 1, "Fireplace Qu": "Gd"})
    # Pair 2 — +/- central air
    pair(mid, {"Central Air": "N"})
    # Pair 3 — +/- a finished basement (same total SF, finished vs unfinished)
    pair(mid, {"BsmtFin Type 1": "GLQ", "BsmtFin SF 1": 1000, "Bsmt Unf SF": 0})
    # Pair 4 — +/- an extra full bathroom
    pair(mid, {"Full Bath": 3})
    # Pair 5 — neighbourhood swap (Gilbert -> NoRidge, a premium area)
    pair(mid, {"Neighborhood": "NoRidge"})
    # Pair 6 — +1 to Overall Qual (the single strongest price driver)
    pair(mid, {"Overall Qual": 7, "Kitchen Qual": "Gd", "Exter Qual": "Gd"})

    return houses


# ---------------------------------------------------------------------------
# Assemble + save
# ---------------------------------------------------------------------------

def run():
    if not os.path.exists(REFERENCE_CSV):
        raise FileNotFoundError(
            f"Need {REFERENCE_CSV} to copy the exact column schema/order."
        )
    ref_cols = list(pd.read_csv(REFERENCE_CSV, nrows=0).columns)

    spread = build_spread(N_SPREAD)
    outliers = build_outliers(start_order=N_SPREAD)
    pairs = build_pairs(start_order=N_SPREAD + len(outliers))

    all_houses = spread + outliers + pairs
    df = pd.DataFrame(all_houses)[ref_cols]  # enforce exact schema + order

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[generate] {len(spread)} spread + {len(outliers)} outliers + "
          f"{len(pairs)} pair-houses = {len(df)} total.")
    print(f"[generate] Columns match reference: {list(df.columns) == ref_cols}")
    print(f"[save]     Wrote {OUTPUT_CSV} ({df.shape[0]} rows, {df.shape[1]} cols).")


if __name__ == "__main__":
    run()