from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception as e:
    raise SystemExit(
    )

PRED_PATH = Path("results/predictions/standardized/all_predictions.csv")
BEST_PATH = Path("results/evaluation/reference_sensitivity/best_f1_by_ref_method_subset.csv")

DEGRADED_DIR = Path("dataset/annotations/degraded_references")
OUT_DIR = Path("results/evaluation/missingness_all_methods")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MATCH_RADIUS = 3.0
INCLUDE_RULE_CANDIDATE = True

METHOD_LABEL = {
    "deepblink_official": "DeepBlink official",
    "intensity_peak": "Intensity peak",
    "logdog": "LoG/DoG",
    "rule_candidate": "Rule-candidate",
    "tophat_peak": "Top-hat peak",
    "unet_heatmap": "U-Net heatmap",
    "unet_heatmap_pos": "U-Net heatmap + POS",
    "yolo_pointbox": "YOLO point-box",
}

SUBSET_ORDER = ["MERFISH", "RCA", "SeqFISH"]

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def compute_metrics(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def greedy_match_points(ref_xy, pred_xy, radius):
    """
    One-to-one nearest-neighbor matching within radius.
    ref_xy: numpy array [N, 2] as x,y
    pred_xy: numpy array [M, 2] as x,y
    """
    n_ref = len(ref_xy)
    n_pred = len(pred_xy)

    if n_ref == 0 and n_pred == 0:
        return 0, 0, 0
    if n_ref == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_ref

    ref_tree = cKDTree(ref_xy)
    dists, idxs = ref_tree.query(pred_xy, distance_upper_bound=radius)

    candidates = []
    for pred_i, (d, ref_i) in enumerate(zip(dists, idxs)):
        if np.isfinite(d) and ref_i < n_ref:
            candidates.append((float(d), int(pred_i), int(ref_i)))

    candidates.sort(key=lambda x: x[0])

    used_pred = set()
    used_ref = set()
    tp = 0

    for d, pred_i, ref_i in candidates:
        if pred_i in used_pred or ref_i in used_ref:
            continue
        used_pred.add(pred_i)
        used_ref.add(ref_i)
        tp += 1

    fp = n_pred - tp
    fn = n_ref - tp
    return tp, fp, fn

def parse_degraded_file(path):
    """
    Expected:
    dataset/annotations/degraded_references/{base_reference}/{policy}/{file}.csv

    Filename examples:
    raw_100_seed00.csv
    raw_50_seed-1.csv
    strict_70_seed00.csv
    """
    base_reference = path.parent.parent.name
    policy = path.parent.name

    name = path.stem

    comp = None
    seed = None

    m = re.search(r"_(50|60|70|80|90|100)(?:_|$)", name)
    if m:
        comp = int(m.group(1))

    m2 = re.search(r"seed(-?\d+)", name)
    if m2:
        seed = int(m2.group(1))
    else:
        seed = -999

    if comp is None:
        raise ValueError(f"Cannot parse completeness from filename: {path}")

    return base_reference, policy, comp, seed

# Load predictions
print("[LOAD] predictions")
pred = pd.read_csv(PRED_PATH)

if not INCLUDE_RULE_CANDIDATE:
    pred = pred[pred["method"] != "rule_candidate"].copy()

pred["method_label"] = pred["method"].map(METHOD_LABEL).fillna(pred["method"])

methods = sorted(pred["method"].unique())
print("[METHODS]")
for m in methods:
    print(" -", m)

# Load full-reference best thresholds
# Threshold rule:
# degraded raw    -> best threshold from raw full reference
# degraded strict -> best threshold from strict full reference
# subset-specific evaluation uses subset-specific threshold.
# ALL evaluation uses ALL threshold later.
print("\n[LOAD] best thresholds")
best = pd.read_csv(BEST_PATH)

if not INCLUDE_RULE_CANDIDATE:
    best = best[best["method"] != "rule_candidate"].copy()

threshold_map = {}
for _, r in best.iterrows():
    threshold_map[(r["ref_name"], r["subset"], r["method"])] = float(r["threshold"])

# fallback: if subset threshold missing, use ALL
def get_threshold(base_reference, subset, method):
    if (base_reference, subset, method) in threshold_map:
        return threshold_map[(base_reference, subset, method)]
    if (base_reference, "ALL", method) in threshold_map:
        return threshold_map[(base_reference, "ALL", method)]
    return 0.0

# Discover degraded references
ref_files = sorted(DEGRADED_DIR.rglob("*.csv"))

if not ref_files:
    raise SystemExit(f"[ERROR] degraded reference CSV not found: {DEGRADED_DIR}")

items = []
for p in ref_files:
    try:
        base_reference, policy, comp, seed = parse_degraded_file(p)
        if base_reference not in set(best["ref_name"].unique()):
            continue
        items.append({
            "path": p,
            "base_reference": base_reference,
            "policy": policy,
            "completeness": comp,
            "seed": seed,
        })
    except Exception as e:
        print("[SKIP]", p, e)

refs_index = pd.DataFrame(items)
refs_index = refs_index.sort_values(["base_reference", "policy", "completeness", "seed"])

index_path = OUT_DIR / "degraded_reference_index.csv"
refs_index.to_csv(index_path, index=False, encoding="utf-8-sig")

print("\n[DEGRADED REFERENCES]")
print(refs_index.groupby(["base_reference", "policy", "completeness"]).size().reset_index(name="n_files").to_string(index=False))
print(f"\nN degraded files: {len(refs_index)}")

# Pre-cache predictions by base_reference/subset/method
# because threshold depends on base reference and subset.
print("\n[CACHE] thresholded predictions")

pred_cache = {}

for base_reference in sorted(refs_index["base_reference"].unique()):
    for subset in SUBSET_ORDER:
        for method in methods:
            thr = get_threshold(base_reference, subset, method)

            q = pred[
                (pred["subset"] == subset) &
                (pred["method"] == method) &
                (pred["score"] >= thr)
            ].copy()

            by_image = {}
            for image_id, g in q.groupby("image_id"):
                by_image[image_id] = g[["x", "y"]].to_numpy(dtype=float)

            pred_cache[(base_reference, subset, method)] = {
                "threshold": thr,
                "n_pred_total": len(q),
                "by_image": by_image,
            }

print("[CACHE DONE]")

# Evaluate
image_rows = []
summary_rows = []

print("\n[EVALUATE]")
for idx, row in refs_index.iterrows():
    ref_path = Path(row["path"])
    base_reference = row["base_reference"]
    policy = row["policy"]
    completeness = int(row["completeness"])
    seed = int(row["seed"])

    print(f"[{idx+1}/{len(refs_index)}] {base_reference} | {policy} | {completeness} | seed={seed}")

    ref = pd.read_csv(ref_path)

    # Standardize degraded reference columns
    # Degraded files use ref_x/ref_y as the final reference point.
    x_candidates = ["x", "ref_x", "axis-1", "axis_1", "col", "column", "cx", "center_x", "centroid_x"]
    y_candidates = ["y", "ref_y", "axis-0", "axis_0", "row", "cy", "center_y", "centroid_y"]

    x_col = next((c for c in x_candidates if c in ref.columns), None)
    y_col = next((c for c in y_candidates if c in ref.columns), None)

    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not find x/y coordinate columns in {ref_path}. columns={list(ref.columns)}"
        )

    ref["x"] = ref[x_col]
    ref["y"] = ref[y_col]

    if "image_name" not in ref.columns:
        image_name_candidates = ["filename", "file_name", "image", "name", "stem"]
        image_name_col = next((c for c in image_name_candidates if c in ref.columns), None)
        if image_name_col is None:
            raise ValueError(
                f"Could not find image_name column in {ref_path}. columns={list(ref.columns)}"
            )
        ref["image_name"] = ref[image_name_col]

    if "subset" not in ref.columns:
        def infer_subset(name):
            s = str(name)
            if s.startswith("MERFISH"):
                return "MERFISH"
            if s.startswith("RCA"):
                return "RCA"
            if s.startswith("SeqFISH"):
                return "SeqFISH"
            return "UNKNOWN"
        ref["subset"] = ref["image_name"].map(infer_subset)

    if "image_id" not in ref.columns:
        name_to_id = (
            pred[["image_name", "image_id"]]
            .drop_duplicates()
            .set_index("image_name")["image_id"]
            .to_dict()
        )
        ref["image_id"] = ref["image_name"].map(name_to_id)

        if ref["image_id"].isna().any() and "stem" in ref.columns:
            pred_stem = pred[["image_name", "image_id"]].drop_duplicates().copy()
            pred_stem["stem"] = pred_stem["image_name"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
            stem_to_id = pred_stem.drop_duplicates("stem").set_index("stem")["image_id"].to_dict()
            ref.loc[ref["image_id"].isna(), "image_id"] = ref.loc[ref["image_id"].isna(), "stem"].map(stem_to_id)

        ref["image_id"] = ref["image_id"].fillna(ref["image_name"])

    required = {"image_id", "image_name", "subset", "x", "y"}
    missing = required - set(ref.columns)
    if missing:
        raise ValueError(f"{ref_path} missing columns after standardization: {missing}")

    for subset in SUBSET_ORDER:
        ref_s = ref[ref["subset"] == subset].copy()

        ref_by_image = {}
        image_name_map = {}
        for image_id, g in ref_s.groupby("image_id"):
            ref_by_image[image_id] = g[["x", "y"]].to_numpy(dtype=float)
            image_name_map[image_id] = str(g["image_name"].iloc[0])

        image_ids = sorted(set(ref_by_image.keys()))

        for method in methods:
            cache = pred_cache[(base_reference, subset, method)]
            pred_by_image = cache["by_image"]
            threshold = cache["threshold"]

            total_ref = 0
            total_pred = 0
            total_tp = 0
            total_fp = 0
            total_fn = 0

            image_f1s = []
            image_precs = []
            image_recs = []
            abs_count_errors = []
            rel_count_errors = []

            for image_id in image_ids:
                ref_xy = ref_by_image.get(image_id, np.empty((0, 2), dtype=float))
                pred_xy = pred_by_image.get(image_id, np.empty((0, 2), dtype=float))

                tp, fp, fn = greedy_match_points(ref_xy, pred_xy, MATCH_RADIUS)
                precision, recall, f1 = compute_metrics(tp, fp, fn)

                n_ref = len(ref_xy)
                n_pred = len(pred_xy)
                count_error = n_pred - n_ref
                abs_count_error = abs(count_error)
                relative_count_error = safe_div(abs_count_error, n_ref)

                total_ref += n_ref
                total_pred += n_pred
                total_tp += tp
                total_fp += fp
                total_fn += fn

                image_f1s.append(f1)
                image_precs.append(precision)
                image_recs.append(recall)
                abs_count_errors.append(abs_count_error)
                rel_count_errors.append(relative_count_error)

                image_rows.append({
                    "method": method,
                    "method_label": METHOD_LABEL.get(method, method),
                    "base_reference": base_reference,
                    "policy": policy,
                    "completeness": completeness,
                    "seed": seed,
                    "subset": subset,
                    "image_id": image_id,
                    "image_name": image_name_map.get(image_id, str(image_id)),
                    "threshold": threshold,
                    "n_ref": n_ref,
                    "n_pred": n_pred,
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

            gp, gr, gf1 = compute_metrics(total_tp, total_fp, total_fn)

            summary_rows.append({
                "method": method,
                "method_label": METHOD_LABEL.get(method, method),
                "base_reference": base_reference,
                "policy": policy,
                "completeness": completeness,
                "seed": seed,
                "subset": subset,
                "n_images": len(image_ids),
                "threshold": threshold,
                "total_ref": total_ref,
                "total_pred": total_pred,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "mean_precision": np.mean(image_precs) if image_precs else 0,
                "mean_recall": np.mean(image_recs) if image_recs else 0,
                "mean_f1": np.mean(image_f1s) if image_f1s else 0,
                "mean_abs_count_error": np.mean(abs_count_errors) if abs_count_errors else 0,
                "mean_relative_count_error": np.mean(rel_count_errors) if rel_count_errors else 0,
                "global_precision": gp,
                "global_recall": gr,
                "global_f1": gf1,
                "match_radius": MATCH_RADIUS,
            })

# Save image-level and subset-level summaries
image_df = pd.DataFrame(image_rows)
summary = pd.DataFrame(summary_rows)

image_path = OUT_DIR / "missingness_all_methods_image_metrics.csv"
summary_path = OUT_DIR / "missingness_all_methods_reference_summary.csv"

image_df.to_csv(image_path, index=False, encoding="utf-8-sig")
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\n[SAVED]")
print(image_path)
print(summary_path)

# Add ALL subset summary by summing subsets
group_cols = ["method", "method_label", "base_reference", "policy", "completeness", "seed"]

all_summary = (
    summary.groupby(group_cols, as_index=False)
    .agg(
        n_images=("n_images", "sum"),
        total_ref=("total_ref", "sum"),
        total_pred=("total_pred", "sum"),
        total_tp=("total_tp", "sum"),
        total_fp=("total_fp", "sum"),
        total_fn=("total_fn", "sum"),
        threshold=("threshold", "mean"),
    )
)

all_summary["subset"] = "ALL"
all_summary["global_precision"] = all_summary["total_tp"] / (all_summary["total_tp"] + all_summary["total_fp"])
all_summary["global_recall"] = all_summary["total_tp"] / (all_summary["total_tp"] + all_summary["total_fn"])
all_summary["global_f1"] = (
    2 * all_summary["global_precision"] * all_summary["global_recall"]
    / (all_summary["global_precision"] + all_summary["global_recall"])
)
all_summary = all_summary.fillna(0)

for c in ["mean_precision", "mean_recall", "mean_f1", "mean_abs_count_error", "mean_relative_count_error", "match_radius"]:
    all_summary[c] = np.nan

summary_full = pd.concat([summary, all_summary[summary.columns]], ignore_index=True)

summary_full_path = OUT_DIR / "missingness_all_methods_reference_summary_with_ALL.csv"
summary_full.to_csv(summary_full_path, index=False, encoding="utf-8-sig")

print(summary_full_path)

# F1 curve master
curve = (
    summary_full.groupby(
        ["subset", "base_reference", "policy", "completeness", "method", "method_label"],
        as_index=False
    )
    .agg(
        global_f1_mean=("global_f1", "mean"),
        global_f1_std=("global_f1", "std"),
        global_precision_mean=("global_precision", "mean"),
        global_recall_mean=("global_recall", "mean"),
        n_runs=("global_f1", "count"),
    )
)

curve["global_f1_std"] = curve["global_f1_std"].fillna(0)

curve_path = OUT_DIR / "missingness_all_methods_f1_curves_master.csv"
curve.to_csv(curve_path, index=False, encoding="utf-8-sig")

# Ranking stability
# Compare each degraded condition ranking to 100% reference ranking
# within the same base_reference and policy.
records = []

for subset in sorted(curve["subset"].unique()):
    for base_reference in sorted(curve["base_reference"].unique()):
        for policy in sorted(curve["policy"].unique()):

            temp = curve[
                (curve["subset"] == subset) &
                (curve["base_reference"] == base_reference) &
                (curve["policy"] == policy)
            ].copy()

            if temp.empty:
                continue

            full100 = temp[temp["completeness"] == 100].copy()
            if full100.empty:
                continue

            full_scores = full100.set_index("method_label")["global_f1_mean"]
            full_top1 = full_scores.idxmax()

            for comp in sorted(temp["completeness"].unique()):
                cur = temp[temp["completeness"] == comp].copy()
                cur_scores = cur.set_index("method_label")["global_f1_mean"]

                common = sorted(set(full_scores.index) & set(cur_scores.index))
                if len(common) < 2:
                    continue

                full_rank = full_scores.loc[common].rank(ascending=False, method="average")
                cur_rank = cur_scores.loc[common].rank(ascending=False, method="average")

                spearman = cur_rank.corr(full_rank, method="spearman")
                current_top1 = cur_scores.idxmax()

                records.append({
                    "subset": subset,
                    "base_reference": base_reference,
                    "policy": policy,
                    "completeness": comp,
                    "spearman_rank_correlation": spearman,
                    "top1_agreement": int(current_top1 == full_top1),
                    "full_reference_top1": full_top1,
                    "current_top1": current_top1,
                    "current_ranking": " > ".join(cur_scores.sort_values(ascending=False).index.tolist()),
                })

rank_df = pd.DataFrame(records)
rank_path = OUT_DIR / "missingness_all_methods_rank_stability_master.csv"
rank_df.to_csv(rank_path, index=False, encoding="utf-8-sig")

print(curve_path)
print(rank_path)

print("\n[PREVIEW: ALL / raw / random]")
preview = curve[
    (curve["subset"] == "ALL") &
    (curve["base_reference"] == "raw") &
    (curve["policy"] == "random")
].sort_values(["completeness", "global_f1_mean"], ascending=[True, False])

print(preview.head(30).to_string(index=False))

print("\n[DONE]")
