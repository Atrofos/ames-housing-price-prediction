"""
predict_tabpfn.py
-----------------
Inference for the TabPFN (foundation model) side of the project — the
parallel of predict.py, for the parallel of model.py.

Design: this file deliberately owns almost NO logic of its own. Every
transformation step — cleaning, feature engineering, column alignment to
the model, target inversion, output building — is imported directly from
predict.py and reused unchanged. The only things that genuinely differ
between predicting with the traditional model and the TabPFN model are:

    1. which saved bundle to load   (best_model_tabpfn.pkl)
    2. where to write the output    (predictions_tabpfn.csv)

...so those are the only things this file specifies. Everything else is the
exact same code path, which means:

  - no duplicated transformation logic to drift out of sync (the same
    anti-skew principle predict.py is built on, applied one level up:
    the two predict files can't diverge because they share one body of
    functions);
  - the TabPFN dependency stays quarantined here — predict.py and the
    whole core pipeline remain runnable for anyone without TabPFN/GPU;
  - structural symmetry: model.py -> predict.py mirrors
    model_tabpfn.py -> predict_tabpfn.py.

Note on target inversion: predict.py's predict() already chooses the
inverse transform from the bundle metadata ("log1p" -> expm1, else raw).
model_tabpfn.py records whichever variant won (log or raw) in exactly that
metadata field, so the SAME predict() function handles both automatically
with no special-casing here.

Usage:
    python predict_tabpfn.py                  # demo on the raw dataset
    python predict_tabpfn.py path/to/new.csv  # score a batch of new houses

Input:  a CSV in AmesHousing.csv format (SalePrice optional)
Output: data/processed/predictions_tabpfn.csv
"""

import os
import sys

# Reuse the entire inference toolkit from predict.py. Importing these by
# name (rather than copy-pasting) is what guarantees the traditional and
# foundation-model predict paths stay identical wherever they should be.
from predict import (
    align_to_model,
    build_output,
    clean_new_data,
    engineer_new_data,
    load_model,
    load_new_data,
    predict,
    save_output,
)


# ---------------------------------------------------------------------------
# Config — the ONLY things that differ from predict.py
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("models", "best_model_tabpfn.pkl")
DEFAULT_INPUT = os.path.join("data", "raw", "AmesHousing.csv")
OUTPUT_PATH = os.path.join("data", "processed", "predictions_tabpfn.csv")

TARGET = "SalePrice"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(input_path: str = DEFAULT_INPUT, output_path: str = OUTPUT_PATH):
    """
    Identical flow to predict.run(), but pointed at the TabPFN bundle and
    its own output file. We re-implement only this thin orchestration —
    not because the logic differs, but because run() in predict.py hardcodes
    its own MODEL_PATH/OUTPUT_PATH constants; the steps it chains are all
    the shared, imported functions below.

    Because it reuses predict.py's predict(), it inherits interval support
    automatically: the TabPFN bundle is flagged interval_kind="tabpfn_native",
    so predict() decodes 5th/50th/95th percentiles from TabPFN's distribution
    and build_output emits the same PriceLower/PriceUpper/IntervalWidth
    columns as the XGBoost side — symmetric output, different mechanism.
    """
    bundle = load_model(MODEL_PATH)
    model = bundle["model"]

    original = load_new_data(input_path)
    df = original.drop(columns=[TARGET], errors="ignore")

    df = clean_new_data(df)
    df = engineer_new_data(df)
    X = align_to_model(df, model)

    preds = predict(bundle, X)
    out = build_output(original, preds)
    save_output(out, output_path)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    run(path)