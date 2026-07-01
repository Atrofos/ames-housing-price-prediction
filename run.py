"""
run.py
------
Pipeline orchestrator. Runs the full pipeline end to end in order, so you
don't have to cycle through each file manually or pass console flags.

    data_cleaning  ->  feature_selection  ->  model_training  ->  predict

Optional stages are controlled by the toggles below.

  RUN_TABPFN     — also run the TabPFN foundation-model track (train +
                   predict). Needs `pip install tabpfn` (+ ideally a GPU).
                   Wrapped in try/except so the core pipeline still runs
                   without TabPFN installed.

  RUN_TEST_DATA  — generate the synthetic test houses and score them through
                   the available predictor(s). Zero-effort way to probe the
                   models on the crafted outliers / pair experiments without
                   typing a path in a console — just leave this True and run.

  TEST_HOUSE_COUNT — how many synthetic houses to generate (approximate;
                   see generate_test_data.py for why it's not exact).

  RUN_COMPARISON — final stage: head-to-head analysis of XGBoost vs TabPFN
                   predictions (accuracy on the demo data, behaviour on the
                   test data) plus saved charts in ./analysis/. Needs
                   matplotlib; reads whatever prediction CSVs exist, so it
                   degrades gracefully if only one model ran.

Output files are kept SEPARATE so nothing gets overwritten:
  - the raw-data demo prediction -> predictions_demo.csv
  - the synthetic test prediction -> predictions_test.csv
  (TabPFN equivalents add a _tabpfn suffix.)

Each stage's own run() does its own loading/saving/validation; this file
just calls them in sequence with a banner between stages.


Usage:
    python run.py
"""

import os

import data_cleaning
import feature_selection
import model_training
import predict


# --- Optional stage toggles -------------------------------------------------
RUN_TABPFN = True         # also run the TabPFN track (needs `pip install tabpfn`)
RUN_TEST_DATA = True       # generate + score the synthetic test houses
RUN_COMPARISON = True      # final head-to-head analysis + charts (needs matplotlib)
TEST_HOUSE_COUNT = 50      # approximate number of synthetic houses to generate

# --- Output paths (kept separate so demo and test never overwrite) ----------
PROC = os.path.join("data", "processed")
DEMO_OUT = os.path.join(PROC, "predictions_demo.csv")
TEST_OUT = os.path.join(PROC, "predictions_test.csv")
DEMO_OUT_TABPFN = os.path.join(PROC, "predictions_demo_tabpfn.csv")
TEST_OUT_TABPFN = os.path.join(PROC, "predictions_test_tabpfn.csv")

TEST_DATA_CSV = "AmesHousing_test.csv"


def banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}\n")


def run():
    banner("STAGE 1 — DATA CLEANING")
    data_cleaning.run()

    banner("STAGE 2 — FEATURE ENGINEERING")
    feature_selection.run()

    banner("STAGE 3 — MODEL TRAINING & COMPARISON")
    model_training.run()

    banner("STAGE 4 — PREDICTION DEMO (raw dataset -> predictions_demo.csv)")
    predict.run(output_path=DEMO_OUT)

    if RUN_TABPFN:
        try:
            import model_training_tabpfn
            import predict_tabpfn
            banner("BONUS — TABPFN TRAINING (foundation model)")
            model_training_tabpfn.run()
            banner("BONUS — TABPFN PREDICTION DEMO (-> predictions_demo_tabpfn.csv)")
            predict_tabpfn.run(output_path=DEMO_OUT_TABPFN)
        except ImportError:
            banner("TabPFN skipped — package not installed "
                   "(`pip install tabpfn` to enable).")

    if RUN_TEST_DATA:
        # Generate the synthetic test set, then score it through whichever
        # predictors ran above. Writes to its OWN files so the demo
        # predictions above are never overwritten.
        import generate_test_data

        # Let run.py drive the house count from one place.
        generate_test_data.TOTAL_HOUSES = TEST_HOUSE_COUNT
        generate_test_data.N_SPREAD = max(
            10, TEST_HOUSE_COUNT
            - generate_test_data.N_OUTLIERS
            - generate_test_data.N_PAIRS
        )

        banner(f"TEST DATA — GENERATING ~{TEST_HOUSE_COUNT} SYNTHETIC HOUSES")
        generate_test_data.run()

        banner("TEST DATA — SCORING WITH XGBOOST (-> predictions_test.csv)")
        predict.run(TEST_DATA_CSV, output_path=TEST_OUT)

        if RUN_TABPFN:
            try:
                import predict_tabpfn
                banner("TEST DATA — SCORING WITH TABPFN (-> predictions_test_tabpfn.csv)")
                predict_tabpfn.run(TEST_DATA_CSV, output_path=TEST_OUT_TABPFN)
            except ImportError:
                pass

    if RUN_COMPARISON:
        # Head-to-head analysis of whatever prediction CSVs now exist, plus
        # saved charts. Reads files defensively, so it works even if only the
        # XGBoost side ran (it just shows fewer comparisons).
        try:
            import compare_predictions
            banner("COMPARISON — XGBOOST vs TABPFN (analysis + charts)")
            compare_predictions.run()
        except ImportError as e:
            banner(f"Comparison skipped — {e} "
                   "(needs matplotlib: `pip install matplotlib`).")

    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    run()