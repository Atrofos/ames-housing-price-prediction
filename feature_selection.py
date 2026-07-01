"""
features_selection.py
-----------
Takes the cleaned Ames Housing dataset and produces a fully numeric,
model-ready dataset through feature selection, feature engineering,
and encoding.

Every drop / engineer / encode decision is documented inline with the
reasoning behind it. The guiding framework used for feature selection:

    1. VARIANCE  — does the column actually vary in THIS dataset?
                   (a feature that is 95%+ one value teaches the model nothing)
    2. IMPACT    — when it does vary, does it move SalePrice meaningfully?
                   (low variance + HIGH impact can still earn a place, usually
                    collapsed into a binary flag, e.g. Functional)
    3. REDUNDANCY — is its information already carried by another column?
                   (keeping both adds multicollinearity noise, destabilises
                    linear model coefficients and muddies SHAP later)

Encoding rules:
    - ORDERED categories (quality grades) -> ordinal integer mapping,
      preserving the inherent ranking (Ex > Gd > TA > Fa > Po).
    - UNORDERED categories (e.g. Garage Type, Neighborhood) -> one-hot,
      because inventing a numeric order would feed the model false info.

Input:  data/processed/ames_cleaned.csv
Output: data/processed/ames_featured.csv
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Config — paths
# ---------------------------------------------------------------------------

INPUT_PATH = os.path.join("data", "processed", "ames_cleaned.csv")
OUTPUT_PATH = os.path.join("data", "processed", "ames_featured.csv")


# ---------------------------------------------------------------------------
# Config — columns dropped for LOW VARIANCE and/or LOW IMPACT
# ---------------------------------------------------------------------------
# Sale Type      — describes the transaction mechanism, not the property.
#                  Weak, indirect signal at best.
# Mo Sold        — mild seasonality exists in housing but the effect is
#                  marginal, and months are cyclical (Dec/Jan adjacent),
#                  which raw integers misrepresent. Defensible either way;
#                  dropped for simplicity.
# Lot Frontage   — 17% of values are our own imputed guesses (490 nulls
#                  filled in cleaning), and its genuine signal is largely
#                  duplicated by Lot Area (bigger plots = wider frontage).
#                  Imputation uncertainty + redundancy = drop.
# Lot Shape      — some variance, but mildly irregular plots in Ames mostly
#                  proxy for bigger plots (already in Lot Area). Borderline.
# Land Contour   — Ames is in the Midwest: overwhelmingly 'Lvl'. Hillside
#                  location would matter in San Francisco; in this dataset
#                  it barely exists. Near-zero variance.
# Land Slope     — same reasoning as Land Contour: almost all 'Gtl'.
#                  Low variance AND low impact when it does vary.
# Roof Style     — ~80% gable, and the price gap between roof shapes is
#                  negligible. Any quality signal lives in Overall Qual.
# Heating        — ~98% 'GasA'. The remaining ~50 rows are scattered across
#                  four rare categories: not enough data points for the
#                  model to learn anything from them.
# Electrical     — ~92% 'SBrkr' (modern circuit breaker). Low variance.
# Fence          — ~80% of houses have no fence, and fence quality when
#                  present is a weak, ambiguous price signal (privacy
#                  fence vs chain link cuts both ways). Caught by the
#                  validation layer after being missed in initial review —
#                  kept honest by our own sanity checks.
# ---------------------------------------------------------------------------

LOW_SIGNAL_DROPS = [
    "Sale Type",
    "Mo Sold",
    "Lot Frontage",
    "Lot Shape",
    "Land Contour",
    "Land Slope",
    "Roof Style",
    "Heating",
    "Electrical",
    "Fence",
]


# ---------------------------------------------------------------------------
# Config — columns dropped for REDUNDANCY
# ---------------------------------------------------------------------------
# Exterior 2nd   — equals Exterior 1st in ~85% of rows; only differs for the
#                  minority of houses with two cladding materials.
# Mas Vnr Type   — the premium signal of masonry veneer is mostly in HAVING
#                  it, which Mas Vnr Area already captures (0 = none).
# BsmtFin Type 2 — grades a 'second finished area' that most basements don't
#                  have; overwhelmingly 'Unf'/'None'. Same situation as
#                  Exterior 2nd.
# Garage Cars    — correlates ~0.89 with Garage Area. Cars is a coarse
#                  bucket; Area is the precise continuous measurement.
#                  Keep the richer feature, drop the count.
# Garage Yr Blt  — correlates ~0.85 with Year Built (garages are mostly
#                  built with the house). Additionally, our cleaning step
#                  filled 'no garage' with 0, which would make age math on
#                  this column treat those garages as 2,000+ years old —
#                  a landmine we created ourselves. Redundant AND poisoned.
# Garage Qual /  — grade the same object as each other and overlap heavily
# Garage Cond      with Overall Qual / garage presence. Weak individual
#                  signal in EDA; dropped to reduce quality-column clutter.
# Fireplace Qu   — adds little beyond fireplace PRESENCE, which the
#                  Fireplaces count column already carries.
# Bsmt Cond /    — weaker siblings of Bsmt Qual / Exter Qual. Condition
# Exter Cond /     columns are consistently weaker predictors than quality
# Overall Cond     columns in Ames (quality = craftsmanship, expensive to
#                  change; condition = state of repair, fixable cheaply —
#                  the market prices quality far more heavily).
# MS SubClass    — a categorical BUILDING-CLASS code (20 = 1-storey 1946+,
#                  60 = 2-storey 1946+, 120 = 1-storey PUD, etc.). Its
#                  numeric values carry no ordinal meaning — 120 is not
#                  "more" than 20 — and what it encodes is already split
#                  across House Style + Bldg Type + age. Subtle trap:
#                  data_cleaning casts it to string, but the CSV round-trip
#                  silently re-infers it as int (a column of "20","60"
#                  looks numeric to pandas), so it slips past the
#                  all-numeric validation check and would otherwise be fed
#                  to the model as a meaningless ordinal. Dropped here.
# ---------------------------------------------------------------------------

REDUNDANCY_DROPS = [
    "Exterior 2nd",
    "Mas Vnr Type",
    "BsmtFin Type 2",
    "Garage Cars",
    "Garage Yr Blt",
    "Garage Qual",
    "Garage Cond",
    "Fireplace Qu",
    "Bsmt Cond",
    "Exter Cond",
    "Overall Cond",
    "MS SubClass",
]


# ---------------------------------------------------------------------------
# Config — ordinal encoding maps
# ---------------------------------------------------------------------------
# These columns are ORDERED categories: Ex(cellent) really is better than
# Po(or). One-hot encoding would discard that ranking and force the model
# to relearn it from scratch across dozens of sparse dummy columns.
# Ordinal integers preserve the order in a single column.
#
# 'None' -> 0 slots in naturally: data_cleaning.py filled "feature doesn't
# exist" nulls (no basement / no fireplace / no garage) with the string
# 'None', and zero quality is exactly the right representation of a thing
# that isn't there.
#
# The nine QUALITY_COLS share one map because the dataset's assessors used
# one grading rubric across them — the code mirrors the structure of the
# data (single source of truth; no copy-paste drift between nine identical
# dictionaries).
# ---------------------------------------------------------------------------

QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

QUALITY_COLS = [
    "Exter Qual",
    "Bsmt Qual",
    "Heating QC",
    "Kitchen Qual",
]

# BsmtFin Type 1 — 6-point scale of how the finished basement area is
# finished: good living quarters down to unfinished. Ordered, own vocabulary.
BSMT_FIN_MAP = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

# Bsmt Exposure — walkout / garden-level exposure. A walkout basement is
# dramatically more usable living space than a fully buried one. Ordered.
EXPOSURE_MAP = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}

# Garage Finish — finished interior > rough finished > unfinished. Ordered.
GARAGE_FIN_MAP = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}

# Paved Drive — paved > partial > dirt/gravel. A genuine ordering (partial
# pavement sits between the two), so ordinal rather than one-hot.
PAVED_DRIVE_MAP = {"N": 0, "P": 1, "Y": 2}


# ---------------------------------------------------------------------------
# Config — one-hot encoding columns
# ---------------------------------------------------------------------------
# These are UNORDERED (nominal) categories. There is no defensible ranking:
# is 'Attchd' garage > 'Detchd'? Is NAmes > OldTown? The question is
# meaningless — forcing ordinal numbers onto these would invent a ranking
# that doesn't exist and feed the model false information.
#
# drop_first=True avoids the dummy-variable trap (perfect multicollinearity
# between dummies) for linear models.
# ---------------------------------------------------------------------------

ONE_HOT_COLS = [
    "MS Zoning",
    "Neighborhood",
    "Bldg Type",
    "House Style",
    "Foundation",      # kept despite partial overlap with house age —
                       # carries some independent signal (a renovated old
                       # house keeps its old foundation). Flagged as a
                       # redundancy candidate to re-test in EDA.
    "Exterior 1st",
    "Garage Type",
    "Sale Condition",  # kept because non-Normal sales (foreclosures,
                       # family sales) systematically distort price —
                       # ignoring this would corrupt the model's view
                       # of true market value.
    "Lot Config",
]


# ---------------------------------------------------------------------------
# Config — Condition 1 grouping
# ---------------------------------------------------------------------------
# Condition 1 has 9 sparse categories (~85% 'Norm'). One-hot encoding it
# would create 9 nearly-all-zero columns. But the information isn't
# worthless — it's just spread too thin. Solution: collapse into two
# hypothesis-driven binary flags.
#
# CRITICAL: two flags, not one. Railways/busy roads push price DOWN;
# parks/greenbelts push price UP. Lumping them into a single "affected"
# flag would blend two opposite-direction signals into noise — when
# grouping categories, only group ones that move the target the SAME way.
# The four railroad codes (RRNn/RRAn/RRNe/RRAe) are individually useless
# (20-50 rows each) but combined form one meaningful concept: "near a
# railway".
# ---------------------------------------------------------------------------

NEGATIVE_CONDITIONS = ["Artery", "Feedr", "RRNn", "RRAn", "RRNe", "RRAe"]
POSITIVE_CONDITIONS = ["PosN", "PosA"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    keep_default_na=False is CRITICAL here: data_cleaning.py fills
    "feature doesn't exist" nulls with the string 'None', but pandas'
    default CSV reader treats the literal text "None" as a missing value
    and silently converts it back to NaN — undoing our cleaning on the
    round trip. Disabling the default NaN list preserves our 'None'
    strings. Safe because the cleaned file is guaranteed null-free.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    print(f"[load]     Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# Each function derives new columns, then DROPS the source columns it
# replaces ("derive then drop") — keeping both the derived feature and its
# raw ingredients would reintroduce the redundancy we're trying to avoid.
# ---------------------------------------------------------------------------

def engineer_space_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalSF    — Total Bsmt SF + 1st Flr SF + 2nd Flr SF.
                 Total space across the WHOLE house, which no single raw
                 column captures. Routinely one of the top-3 correlates
                 with SalePrice: buyers fundamentally pay for space.
    BsmtFinSF  — BsmtFin SF 1 + BsmtFin SF 2.
                 Total FINISHED basement space. Total Bsmt SF alone is
                 blind to finish state: a carpeted rec room and a bare
                 concrete cave can have identical square footage but very
                 different value.

    Kept alongside: Gr Liv Area — above-ground space is valued at a
    premium per square foot vs basement space; keeping it next to TotalSF
    lets the model learn that premium.

    Dropped: 1st/2nd Flr SF (sum lives in TotalSF and Gr Liv Area; the
    floor split carries little signal beyond House Style), BsmtFin SF 1/2
    (now in BsmtFinSF), Bsmt Unf SF (pure double-count: it equals
    Total Bsmt SF minus finished space, so it's fully implied).
    """
    df["TotalSF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
    df["BsmtFinSF"] = df["BsmtFin SF 1"] + df["BsmtFin SF 2"]

    df = df.drop(columns=[
        "1st Flr SF", "2nd Flr SF",
        "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    ])
    print("[engineer] Space: built TotalSF, BsmtFinSF. Dropped 5 source columns.")
    return df


def engineer_bath_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalBaths — Full baths + 0.5 x half baths, across all floors.

    The 0.5 weighting encodes real-world value: a half bath (toilet +
    sink, no shower/tub) is not worth a full bathroom, and a raw count
    would falsely equate them. Fractional results like 3.5 are not a bug —
    it's literally how listings are written ("3.5 bath"), and the model
    only needs a number where bigger = more bathroom value.

    The four raw columns are individually sparse (basement baths mostly 0,
    half baths mostly 0/1) and weak; combined they form one dense, strong
    feature. The above/below-ground bath split is not separately encoded —
    the above-ground premium is already captured by keeping Gr Liv Area
    alongside TotalSF.
    """
    df["TotalBaths"] = (
        df["Full Bath"] + df["Bsmt Full Bath"]
        + 0.5 * (df["Half Bath"] + df["Bsmt Half Bath"])
    )

    df = df.drop(columns=[
        "Full Bath", "Half Bath", "Bsmt Full Bath", "Bsmt Half Bath",
    ])
    print("[engineer] Baths: built TotalBaths (0.5-weighted). Dropped 4 source columns.")
    return df


def engineer_porch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalPorchSF — sum of all deck/porch square footage.

    Four sparse, mostly-zero columns describing one concept for price
    purposes: usable outdoor living space. Merged into one dense feature
    (same medicine as Condition 1's sparse categories). No fractional
    weighting needed — unlike baths, there's no market convention that a
    deck is worth half a porch.
    """
    df["TotalPorchSF"] = (
        df["Wood Deck SF"] + df["Open Porch SF"]
        + df["Enclosed Porch"] + df["Screen Porch"]
    )

    df = df.drop(columns=[
        "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "Screen Porch",
    ])
    print("[engineer] Porch: built TotalPorchSF. Dropped 4 source columns.")
    return df


def engineer_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HouseAge   — Yr Sold - Year Built. What matters to price isn't WHEN a
                 house was built but HOW OLD it was at the moment of sale:
                 a 2005 build was nearly new if sold in 2006, a different
                 proposition by 2010. Raw years are also awkward on a
                 numeric scale (huge values, meaningless zero point);
                 ages starting from 0 are far better behaved.
    RemodAge   — Yr Sold - Year Remod/Add. Years since last work; a recent
                 renovation is a price boost that fades over time.
    Remodelled — binary flag: was the house EVER remodelled? (In the raw
                 data, Year Remod/Add simply equals Year Built when no
                 work was ever recorded.) Note this captures action taken,
                 not need for action — an untouched 1960s kitchen shows up
                 instead via the quality grades (Kitchen Qual etc.), which
                 capture current state. Division of labour: age features
                 carry the TIME dimension, quality columns carry the
                 CURRENT STATE dimension.

    Yr Sold is KEPT (cast back to int — data_cleaning stored it as a
    string): this dataset spans 2006-2010, i.e. straight through the
    financial crash, so the sale year captures genuine market conditions
    independent of the house itself.

    Dropped: Year Built, Year Remod/Add (their information now lives in
    the engineered ages).
    """
    df["Yr Sold"] = df["Yr Sold"].astype(int)

    df["HouseAge"] = df["Yr Sold"] - df["Year Built"]
    df["RemodAge"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["Remodelled"] = (df["Year Remod/Add"] != df["Year Built"]).astype(int)

    df = df.drop(columns=["Year Built", "Year Remod/Add"])
    print("[engineer] Age: built HouseAge, RemodAge, Remodelled. Dropped 2 source columns.")
    return df


def engineer_condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    NearNegative — adjacent/near busy roads or railways (price-suppressing).
    NearPositive — adjacent/near parks or greenbelts (price-lifting).

    See NEGATIVE_CONDITIONS / POSITIVE_CONDITIONS config comments for the
    full reasoning on why two flags rather than one.
    """
    df["NearNegative"] = df["Condition 1"].isin(NEGATIVE_CONDITIONS).astype(int)
    df["NearPositive"] = df["Condition 1"].isin(POSITIVE_CONDITIONS).astype(int)

    df = df.drop(columns=["Condition 1"])
    print("[engineer] Condition: built NearNegative, NearPositive flags. Dropped Condition 1.")
    return df


def engineer_functional_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    HasDeductions — 1 if the home has ANY functional deduction (damage /
    deficiency), 0 if typical.

    Functional is ~93% 'Typ' — a variance-check failure on its face. But
    it's the low-variance + HIGH-impact case: when a house IS rated Maj2
    or Sev, its price sits substantially below what every other feature
    suggests. Rare, but fires hard. Collapsing to a binary flag keeps the
    "something is wrong with this house" signal at the cost of one column.
    (Contrast with Land Slope: low variance + LOW impact = plain drop.)
    """
    df["HasDeductions"] = (df["Functional"] != "Typ").astype(int)

    df = df.drop(columns=["Functional"])
    print("[engineer] Functional: built HasDeductions flag. Dropped Functional.")
    return df


def engineer_binary_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Central Air — Y/N to 1/0. Kept despite lowish variance (~7% without):
    it's a clean binary needing zero encoding work, and no-AC houses are
    systematically older and cheaper — when it fires, it points one
    direction. (Same frequency-x-impact logic as HasDeductions.)

    Mas Vnr Area is kept as-is: masonry veneer (decorative brick/stone
    facing) is a genuine premium marker, it's already numeric, and 0
    already means "none".
    """
    df["Central Air"] = (df["Central Air"] == "Y").astype(int)
    print("[engineer] Binary: converted Central Air to 1/0.")
    return df


# ---------------------------------------------------------------------------
# Dropping
# ---------------------------------------------------------------------------

def drop_low_signal(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[1]
    df = df.drop(columns=LOW_SIGNAL_DROPS, errors="ignore")
    print(f"[drop]     Low variance/impact: removed {before - df.shape[1]} columns.")
    return df


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[1]
    df = df.drop(columns=REDUNDANCY_DROPS, errors="ignore")
    print(f"[drop]     Redundancy: removed {before - df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the shared QUALITY_MAP across all standard-scale grade columns,
    then the three special-scale maps. See config comments for reasoning.
    """
    for col in QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP)

    if "BsmtFin Type 1" in df.columns:
        df["BsmtFin Type 1"] = df["BsmtFin Type 1"].map(BSMT_FIN_MAP)

    if "Bsmt Exposure" in df.columns:
        df["Bsmt Exposure"] = df["Bsmt Exposure"].map(EXPOSURE_MAP)

    if "Garage Finish" in df.columns:
        df["Garage Finish"] = df["Garage Finish"].map(GARAGE_FIN_MAP)

    if "Paved Drive" in df.columns:
        df["Paved Drive"] = df["Paved Drive"].map(PAVED_DRIVE_MAP)

    print(f"[encode]   Ordinal: mapped {len(QUALITY_COLS)} shared-scale + 4 special-scale columns.")
    return df


def encode_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[1]
    present = [c for c in ONE_HOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present, drop_first=True, dtype=int)
    print(f"[encode]   One-hot: {len(present)} nominal columns -> "
          f"{df.shape[1] - before + len(present)} dummy columns.")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    """
    Sanity checks before saving:
      - no nulls (an unmapped ordinal value would surface here as NaN)
      - no remaining text columns (everything must be numeric for modelling)
      - SalePrice still present and untouched
      - row count unchanged (feature work should never gain/lose rows)
    """
    issues = 0

    nulls = df.isnull().sum().sum()
    if nulls > 0:
        null_cols = df.columns[df.isnull().any()].tolist()
        print(f"[validate] WARNING — {nulls} nulls present in: {null_cols}")
        print("[validate]           (likely an ordinal map missing a value)")
        issues += 1

    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        print(f"[validate] WARNING — non-numeric columns remain: {object_cols}")
        issues += 1

    if "SalePrice" not in df.columns:
        print("[validate] WARNING — SalePrice missing!")
        issues += 1

    if df.shape[0] != 2930:
        print(f"[validate] WARNING — row count changed: {df.shape[0]} (expected 2930)")
        issues += 1

    print(f"\n[validate] Final shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    if issues == 0:
        print("[validate] Fully numeric, no nulls. Ready for modelling.")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"[cleanup]  Existing file removed: {path}")


def save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save]     Saved featured data to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    df = load_data(INPUT_PATH)

    # Selection: remove what doesn't earn its place
    df = drop_low_signal(df)
    df = drop_redundant(df)

    # Engineering: derive, then drop sources
    df = engineer_space_features(df)
    df = engineer_bath_features(df)
    df = engineer_porch_features(df)
    df = engineer_age_features(df)
    df = engineer_condition_flags(df)
    df = engineer_functional_flag(df)
    df = engineer_binary_conversions(df)

    # Encoding: ordered -> ordinal, unordered -> one-hot
    df = encode_ordinals(df)
    df = encode_one_hot(df)

    validate(df)
    remove_if_exists(OUTPUT_PATH)
    save(df, OUTPUT_PATH)


if __name__ == "__main__":
    run()