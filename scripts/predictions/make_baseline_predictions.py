from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

import pandas as pd


DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "results"

CAND_PATH = DATASET_ROOT / "candidates" / "test_candidate_features_refined.csv"

OUT_DIR = RESULTS_ROOT / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "rule_candidate_baseline_predictions.csv"


def normalize_score_per_subset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    scores = []

    for subset, sub in df.groupby("subset"):
        s = sub["snr_proxy"].astype(float)
        lo = s.quantile(0.01)
        hi = s.quantile(0.99)

        if hi <= lo:
            norm = pd.Series(0.5, index=sub.index)
        else:
            norm = ((s - lo) / (hi - lo)).clip(0, 1)

        scores.append(norm)

    df["score"] = pd.concat(scores).sort_index()
    return df


def main():
    if not CAND_PATH.exists():
        raise FileNotFoundError(f"candidate file not found: {CAND_PATH}")

    cand = pd.read_csv(CAND_PATH)
    cand = cand.loc[:, ~cand.columns.duplicated()].copy()

    required = ["image_name", "subset", "x", "y", "snr_proxy", "log_response"]
    for c in required:
        if c not in cand.columns:
            raise ValueError(f"missing column: {c}")

    pred = cand.copy()
    pred = normalize_score_per_subset(pred)

    pred = pred.rename(columns={
        "x": "pred_x",
        "y": "pred_y",
    })

    keep_cols = [
        "image_path",
        "image_name",
        "stem",
        "subset",
        "pred_x",
        "pred_y",
        "score",
        "snr_proxy",
        "log_response",
        "nearest_neighbor_distance",
        "distance_to_boundary",
    ]
    keep_cols = [c for c in keep_cols if c in pred.columns]

    pred = pred[keep_cols].copy()
    pred["model_name"] = "rule_candidate_baseline"

    pred.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {OUT_PATH}")
    print(f"prediction rows: {len(pred)}")

    print("Done.")
    print(pred.groupby("subset").size())

    print("\n===== score summary =====")
    print(pred.groupby("subset")["score"].describe())

    print("\nDone.")


if __name__ == "__main__":
    main()