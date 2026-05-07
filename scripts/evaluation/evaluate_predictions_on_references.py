from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "results"

PRED_PATH = RESULTS_ROOT / "predictions" / "rule_candidate_baseline_predictions.csv"

DEG_SUMMARY_PATH = (
    DATASET_ROOT
    / "annotations"
    / "degraded_references"
    / "degraded_reference_summary.csv"
)

OUT_DIR = RESULTS_ROOT / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_IMAGE_METRICS = OUT_DIR / "rule_candidate_baseline_image_metrics.csv"
OUT_SUMMARY = OUT_DIR / "rule_candidate_baseline_reference_summary.csv"

MATCH_RADIUS = 3.0


def greedy_match(ref_pts: np.ndarray, pred_pts: np.ndarray, radius: float):
    n_ref = len(ref_pts)
    n_pred = len(pred_pts)

    if n_ref == 0 or n_pred == 0:
        return 0

    tree = cKDTree(pred_pts)
    pairs = []

    for ri, p in enumerate(ref_pts):
        pred_idxs = tree.query_ball_point(p, r=radius)
        for pi in pred_idxs:
            dist = float(np.linalg.norm(ref_pts[ri] - pred_pts[pi]))
            pairs.append((dist, ri, pi))

    pairs.sort(key=lambda x: x[0])

    used_ref = set()
    used_pred = set()
    matched = 0

    for dist, ri, pi in pairs:
        if ri in used_ref or pi in used_pred:
            continue
        used_ref.add(ri)
        used_pred.add(pi)
        matched += 1

    return matched


def read_reference(path: Path) -> pd.DataFrame:
    ref = pd.read_csv(path)
    ref = ref.loc[:, ~ref.columns.duplicated()].copy()

    if "ref_x" not in ref.columns or "ref_y" not in ref.columns:
        raise ValueError(f"reference must have ref_x/ref_y: {path}")

    return ref


def evaluate_one_reference(pred: pd.DataFrame, ref_path: Path, meta: dict):
    ref = read_reference(ref_path)

    rows = []

    image_names = sorted(set(ref["image_name"].unique()) | set(pred["image_name"].unique()))

    for image_name in image_names:
        ref_img = ref[ref["image_name"] == image_name].copy()
        pred_img = pred[pred["image_name"] == image_name].copy()

        subset = None
        if len(ref_img) > 0 and "subset" in ref_img.columns:
            subset = ref_img["subset"].iloc[0]
        elif len(pred_img) > 0 and "subset" in pred_img.columns:
            subset = pred_img["subset"].iloc[0]

        ref_pts = ref_img[["ref_x", "ref_y"]].values.astype(float) if len(ref_img) > 0 else np.empty((0, 2))
        pred_pts = pred_img[["pred_x", "pred_y"]].values.astype(float) if len(pred_img) > 0 else np.empty((0, 2))

        tp = greedy_match(ref_pts, pred_pts, MATCH_RADIUS)
        fp = len(pred_pts) - tp
        fn = len(ref_pts) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan

        count_error = len(pred_pts) - len(ref_pts)
        abs_count_error = abs(count_error)
        relative_count_error = abs_count_error / len(ref_pts) if len(ref_pts) > 0 else np.nan

        rows.append({
            **meta,
            "image_name": image_name,
            "subset": subset,
            "n_ref": len(ref_pts),
            "n_pred": len(pred_pts),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count_error": count_error,
            "abs_count_error": abs_count_error,
            "relative_count_error": relative_count_error,
            "match_radius": MATCH_RADIUS,
        })

    return pd.DataFrame(rows)


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"prediction file not found: {PRED_PATH}")
    if not DEG_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"degraded summary not found: {DEG_SUMMARY_PATH}")

    pred = pd.read_csv(PRED_PATH)
    pred = pred.loc[:, ~pred.columns.duplicated()].copy()

    deg_summary = pd.read_csv(DEG_SUMMARY_PATH)

    all_metric_parts = []

    print(f"[1] prediction LOAD: {PRED_PATH}")
    print(f"pred rows: {len(pred)}")
    print(f"[2] degraded summary LOAD: {DEG_SUMMARY_PATH}")
    print(f"reference files: {len(deg_summary)}")

    for idx, row in deg_summary.iterrows():
        ref_path = Path(row["path"])

        meta = {
            "model_name": "rule_candidate_baseline",
            "base_reference": row["base_reference"],
            "policy": row["policy"],
            "completeness": int(row["completeness"]),
            "seed": int(row["seed"]),
            "reference_path": str(ref_path),
        }

        try:
            metrics = evaluate_one_reference(pred, ref_path, meta)
            all_metric_parts.append(metrics)
        except Exception as e:
            print(f"[ERROR] {ref_path}: {e}")

        if (idx + 1) % 25 == 0:
            print(f"[PROGRESS] {idx + 1}/{len(deg_summary)}")

    image_metrics = pd.concat(all_metric_parts, ignore_index=True)
    image_metrics.to_csv(OUT_IMAGE_METRICS, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] image metrics: {OUT_IMAGE_METRICS}")

    summary = (
        image_metrics
        .groupby(["model_name", "base_reference", "policy", "completeness", "seed", "subset"])
        .agg(
            n_images=("image_name", "count"),
            total_ref=("n_ref", "sum"),
            total_pred=("n_pred", "sum"),
            total_tp=("tp", "sum"),
            total_fp=("fp", "sum"),
            total_fn=("fn", "sum"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_abs_count_error=("abs_count_error", "mean"),
            mean_relative_count_error=("relative_count_error", "mean"),
        )
        .reset_index()
    )

    summary["global_precision"] = summary["total_tp"] / (summary["total_tp"] + summary["total_fp"])
    summary["global_recall"] = summary["total_tp"] / (summary["total_tp"] + summary["total_fn"])
    summary["global_f1"] = (
        2 * summary["global_precision"] * summary["global_recall"]
        / (summary["global_precision"] + summary["global_recall"])
    )

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"[SAVED] summary: {OUT_SUMMARY}")

    print("\n===== Raw/Strict 100% summary =====")
    q = summary[
        (summary["completeness"] == 100)
        & (summary["policy"].isin(["random", "weak_evidence_first"]))
        & (summary["seed"].isin([-1, 0]))
    ]
    print(q[[
        "base_reference",
        "policy",
        "completeness",
        "seed",
        "subset",
        "global_precision",
        "global_recall",
        "global_f1",
        "total_ref",
        "total_pred",
    ]].head(30))

    print("\nDone.")


if __name__ == "__main__":
    main()