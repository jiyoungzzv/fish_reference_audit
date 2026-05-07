from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError as e:
    raise ImportError(
        "pip install scipy\n"
        "conda install scipy"
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT

PRED_PATH = PROJECT_ROOT / "results" / "predictions" / "standardized" / "all_predictions.csv"
REF_DIR = PROJECT_ROOT / "results" / "references" / "standardized"

OUT_DIR = PROJECT_ROOT / "results" / "evaluation" / "reference_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_NAMES = ["raw", "lenient", "strict"]

METHODS = [
    "intensity_peak",
    "logdog",
    "tophat_peak",
    "unet_heatmap",
    "unet_heatmap_pos",
    "deepblink_official",
    "yolo_pointbox",
    "rule_candidate",
]

SUBSETS = ["MERFISH", "RCA", "SeqFISH"]

MATCH_RADIUS = 3.0

# score threshold sweep
THRESHOLDS = [
    0.00, 0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
    0.975, 0.99
]


def greedy_match_points(pred_xy, pred_scores, ref_xy, radius):
    n_pred = len(pred_xy)
    n_ref = len(ref_xy)

    if n_pred == 0 and n_ref == 0:
        return 0, 0, 0

    if n_pred == 0:
        return 0, 0, n_ref

    if n_ref == 0:
        return 0, n_pred, 0

    tree = cKDTree(ref_xy)

    order = np.argsort(-pred_scores)
    matched_ref = np.zeros(n_ref, dtype=bool)

    tp = 0
    fp = 0

    for idx in order:
        p = pred_xy[idx]

        candidate_indices = tree.query_ball_point(p, r=radius)

        if len(candidate_indices) == 0:
            fp += 1
            continue

        candidate_indices = [j for j in candidate_indices if not matched_ref[j]]

        if len(candidate_indices) == 0:
            fp += 1
            continue

        cand_xy = ref_xy[candidate_indices]
        dists = np.sqrt(((cand_xy - p) ** 2).sum(axis=1))
        best_local = int(np.argmin(dists))
        best_ref_idx = candidate_indices[best_local]

        matched_ref[best_ref_idx] = True
        tp += 1

    fn = n_ref - tp

    return tp, fp, fn


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def evaluate_one(pred_df, ref_df, method, ref_name, subset, threshold):
    p = pred_df[
        (pred_df["method"] == method)
        & (pred_df["subset"] == subset)
        & (pred_df["score"] >= threshold)
    ].copy()

    r = ref_df[ref_df["subset"] == subset].copy()

    image_ids = sorted(set(p["image_id"].unique()).union(set(r["image_id"].unique())))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_id in image_ids:
        p_img = p[p["image_id"] == image_id]
        r_img = r[r["image_id"] == image_id]

        pred_xy = p_img[["x", "y"]].to_numpy(dtype=float)
        pred_scores = p_img["score"].to_numpy(dtype=float)
        ref_xy = r_img[["x", "y"]].to_numpy(dtype=float)

        tp, fp, fn = greedy_match_points(
            pred_xy=pred_xy,
            pred_scores=pred_scores,
            ref_xy=ref_xy,
            radius=MATCH_RADIUS,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)

    return {
        "ref_name": ref_name,
        "method": method,
        "subset": subset,
        "threshold": threshold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": int(len(p)),
        "n_ref": int(len(r)),
        "n_images_pred": int(p["image_id"].nunique()),
        "n_images_ref": int(r["image_id"].nunique()),
    }


def evaluate_all_subset_and_overall(pred_df, ref_df, method, ref_name, threshold):
    rows = []

    subset_rows = []

    for subset in SUBSETS:
        row = evaluate_one(
            pred_df=pred_df,
            ref_df=ref_df,
            method=method,
            ref_name=ref_name,
            subset=subset,
            threshold=threshold,
        )
        subset_rows.append(row)
        rows.append(row)

    total_tp = sum(r["tp"] for r in subset_rows)
    total_fp = sum(r["fp"] for r in subset_rows)
    total_fn = sum(r["fn"] for r in subset_rows)

    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)

    overall = {
        "ref_name": ref_name,
        "method": method,
        "subset": "ALL",
        "threshold": threshold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": sum(r["n_pred"] for r in subset_rows),
        "n_ref": sum(r["n_ref"] for r in subset_rows),
        "n_images_pred": sum(r["n_images_pred"] for r in subset_rows),
        "n_images_ref": sum(r["n_images_ref"] for r in subset_rows),
    }

    rows.append(overall)

    return rows


def pick_best_threshold(eval_df):
    sorted_df = eval_df.sort_values(
        ["ref_name", "method", "subset", "f1", "threshold"],
        ascending=[True, True, True, False, True],
    )

    best = (
        sorted_df.groupby(["ref_name", "method", "subset"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return best


def make_ranking(best_df):
    overall = best_df[best_df["subset"] == "ALL"].copy()

    overall["rank"] = (
        overall.groupby("ref_name")["f1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    overall = overall.sort_values(["ref_name", "rank", "method"])

    return overall


def main():
    print("[INFO] Evaluate reference sensitivity")
    print(f"PRED_PATH: {PRED_PATH}")
    print(f"REF_DIR: {REF_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"MATCH_RADIUS: {MATCH_RADIUS}")

    pred_df = pd.read_csv(PRED_PATH)

    required_pred = ["image_id", "subset", "method", "x", "y", "score"]
    missing_pred = [c for c in required_pred if c not in pred_df.columns]
    if missing_pred:
        raise ValueError(f"Prediction file missing columns: {missing_pred}")

    print("\n[PREDICTION SUMMARY]")
    print(
        pred_df.groupby(["method", "subset"])
        .agg(n_images=("image_id", "nunique"), n_pred=("x", "count"))
        .reset_index()
        .to_string(index=False)
    )

    all_rows = []

    for ref_name in REF_NAMES:
        ref_path = REF_DIR / f"{ref_name}.csv"
        print("\n" + "=" * 100)
        print(f"[REFERENCE] {ref_name}")
        print(f"path: {ref_path}")

        ref_df = pd.read_csv(ref_path)

        required_ref = ["image_id", "subset", "x", "y"]
        missing_ref = [c for c in required_ref if c not in ref_df.columns]
        if missing_ref:
            raise ValueError(f"Reference file missing columns: {missing_ref}")

        print(
            ref_df.groupby("subset")
            .agg(n_images=("image_id", "nunique"), n_ref=("x", "count"))
            .reset_index()
            .to_string(index=False)
        )

        for method in METHODS:
            print(f"\n[METHOD] {method}")

            for threshold in THRESHOLDS:
                rows = evaluate_all_subset_and_overall(
                    pred_df=pred_df,
                    ref_df=ref_df,
                    method=method,
                    ref_name=ref_name,
                    threshold=threshold,
                )
                all_rows.extend(rows)

            print("done")

    eval_df = pd.DataFrame(all_rows)

    eval_path = OUT_DIR / "threshold_sweep_all.csv"
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")

    best_df = pick_best_threshold(eval_df)
    best_path = OUT_DIR / "best_f1_by_ref_method_subset.csv"
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")

    ranking_df = make_ranking(best_df)
    ranking_path = OUT_DIR / "overall_ranking_by_reference.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    pivot_f1 = ranking_df.pivot(
        index="method",
        columns="ref_name",
        values="f1",
    ).reset_index()

    pivot_rank = ranking_df.pivot(
        index="method",
        columns="ref_name",
        values="rank",
    ).reset_index()

    pivot_f1_path = OUT_DIR / "overall_f1_pivot.csv"
    pivot_rank_path = OUT_DIR / "overall_rank_pivot.csv"

    pivot_f1.to_csv(pivot_f1_path, index=False, encoding="utf-8-sig")
    pivot_rank.to_csv(pivot_rank_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("[SAVED]")
    print(eval_path)
    print(best_path)
    print(ranking_path)
    print(pivot_f1_path)
    print(pivot_rank_path)

    print("\n[OVERALL RANKING]")
    show_cols = [
        "ref_name",
        "rank",
        "method",
        "f1",
        "precision",
        "recall",
        "threshold",
        "tp",
        "fp",
        "fn",
        "n_pred",
        "n_ref",
    ]
    print(ranking_df[show_cols].to_string(index=False))

    print("\n[OVERALL F1 PIVOT]")
    print(pivot_f1.to_string(index=False))

    print("\n[OVERALL RANK PIVOT]")
    print(pivot_rank.to_string(index=False))


if __name__ == "__main__":
    main()