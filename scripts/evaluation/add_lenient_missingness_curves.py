from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise ImportError("Required optional dependency is missing. Please install the required package.")



WORK = PROJECT_ROOT

PRED_PATH = WORK / "results/predictions/standardized/all_predictions.csv"
REF_DIR = WORK / "results/references/standardized"

OLD_EVAL_DIR = WORK / "results/evaluation/missingness_all_methods"
OLD_INDEX_PATH = OLD_EVAL_DIR / "degraded_reference_index.csv"
OLD_MASTER_PATH = OLD_EVAL_DIR / "missingness_all_methods_f1_curves_master.csv"

OUT_REF_DIR = WORK / "dataset/annotations/degraded_references/lenient"
OUT_EVAL_DIR = OLD_EVAL_DIR
OUT_FIG_DIR = WORK / "results/figures/missingness"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_LENIENT_CURVES = OUT_EVAL_DIR / "missingness_all_methods_f1_curves_lenient_only.csv"
OUT_MASTER_WITH_LENIENT = OUT_EVAL_DIR / "missingness_all_methods_f1_curves_master.csv"
OUT_INDEX_WITH_LENIENT = OUT_EVAL_DIR / "degraded_reference_index.csv"

MATCH_RADIUS = 3.0

DEFAULT_LEVELS = [50, 60, 70, 80, 90, 100]
DEFAULT_RANDOM_SEEDS = list(range(20))

METHOD_LABELS = {
    "deepblink_official": "DeepBlink",
    "intensity_peak": "Intensity peak",
    "logdog": "LoG/DoG",
    "rule_candidate": "Rule candidate",
    "tophat_peak": "Top-hat peak",
    "unet_heatmap": "U-Net heatmap",
    "unet_heatmap_pos": "U-Net heatmap + POS",
    "yolo_pointbox": "YOLO point-box",
}

POLICY_ORDER = ["random", "weak_evidence_first", "dense_first", "boundary_first"]
POLICY_TITLES = {
    "random": "Random",
    "weak_evidence_first": "Weak-evidence-first",
    "dense_first": "Dense-first",
    "boundary_first": "Boundary-first",
}

REFERENCE_ORDER = ["raw", "lenient", "strict"]
REFERENCE_TITLES = {
    "raw": "Raw reference",
    "lenient": "Lenient reference",
    "strict": "Strict reference",
}



def pick_col(df, candidates, required=True, name="column"):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Cannot find {name}. Tried={candidates}, columns={list(df.columns)}")
    return None


def detect_xy(df, role="ref"):
    if role == "pred":
        pairs = [
            ("pred_x", "pred_y"),
            ("x", "y"),
            ("cx", "cy"),
            ("center_x", "center_y"),
            ("candidate_x", "candidate_y"),
            ("ref_x", "ref_y"),
            ("axis-1", "axis-0"),
            ("axis_1", "axis_0"),
        ]
    else:
        pairs = [
            ("ref_x", "ref_y"),
            ("x", "y"),
            ("raw_x", "raw_y"),
            ("candidate_x", "candidate_y"),
            ("axis-1", "axis-0"),
            ("axis_1", "axis_0"),
        ]

    lower = {c.lower(): c for c in df.columns}
    for x, y in pairs:
        if x.lower() in lower and y.lower() in lower:
            return lower[x.lower()], lower[y.lower()]
    raise ValueError(f"Cannot detect x/y columns for {role}. columns={list(df.columns)}")


def find_lenient_reference():
    files = list(REF_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No reference csv files found in {REF_DIR}")

    priority_keywords = [
        "raw_plus_lenient",
        "raw+lenient",
        "lenient_plus_raw",
        "lenient_raw",
        "raw_lenient",
        "lenient",
    ]

    candidates = []
    for f in files:
        name = f.name.lower()
        if any(bad in name for bad in ["weak", "rejected", "ambiguous"]):
            continue

        score = 0
        for i, kw in enumerate(priority_keywords):
            if kw in name:
                score += 100 - i
        if score > 0:
            candidates.append((score, f))

    if not candidates:
        print("\n[AVAILABLE REFERENCE FILES]")
        for f in files:
            print(" -", f.name)
        raise FileNotFoundError(
        )

    selected = sorted(candidates, key=lambda x: (-x[0], len(x[1].name)))[0][1]
    print(f"[SELECTED LENIENT REFERENCE] {selected}")
    print("[CHECK] This must mean: Lenient = Raw + additional accepted candidate spots")
    return selected


def standardize_reference(path):
    df = pd.read_csv(path)

    image_col = pick_col(df, ["image_name", "filename", "file_name", "image", "img", "stem"], True, "image")
    subset_col = pick_col(df, ["subset", "dataset", "split"], False, "subset")
    x_col, y_col = detect_xy(df, role="ref")

    out = df.copy()
    out["image_name"] = df[image_col].astype(str)
    out["subset"] = df[subset_col].astype(str) if subset_col else "ALL"
    out["x"] = pd.to_numeric(df[x_col], errors="coerce")
    out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    out = out.dropna(subset=["x", "y"]).reset_index(drop=True)

    return out


def standardize_predictions(path):
    df = pd.read_csv(path)

    image_col = pick_col(df, ["image_name", "filename", "file_name", "image", "img", "stem"], True, "image")
    subset_col = pick_col(df, ["subset", "dataset", "split"], False, "subset")
    method_col = pick_col(df, ["method", "model", "detector", "name"], True, "method")
    x_col, y_col = detect_xy(df, role="pred")

    out = pd.DataFrame()
    out["image_name"] = df[image_col].astype(str)
    out["subset"] = df[subset_col].astype(str) if subset_col else "ALL"
    out["method"] = df[method_col].astype(str)
    out["x"] = pd.to_numeric(df[x_col], errors="coerce")
    out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    out = out.dropna(subset=["x", "y"]).reset_index(drop=True)

    return out


def get_levels_and_seeds():
    if not OLD_INDEX_PATH.exists():
        return DEFAULT_LEVELS, DEFAULT_RANDOM_SEEDS

    idx = pd.read_csv(OLD_INDEX_PATH)
    levels = sorted(idx["completeness"].dropna().astype(int).unique().tolist())

    if "seed" in idx.columns:
        seeds = sorted(
            idx[(idx["policy"] == "random") & (idx["seed"] >= 0)]["seed"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    else:
        seeds = []

    if not levels:
        levels = DEFAULT_LEVELS
    if not seeds:
        seeds = DEFAULT_RANDOM_SEEDS

    return levels, seeds


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mu = s.mean()
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def add_policy_scores(df):
    out = df.copy()

    # Weak-evidence score
    score_parts = []
    for col in ["snr_proxy", "log_response", "score", "confidence", "response"]:
        if col in out.columns:
            score_parts.append(zscore(out[col]))

    if score_parts:
        out["_weak_keep_score"] = sum(score_parts)
    else:
        out["_weak_keep_score"] = np.arange(len(out), dtype=float)

    # Dense score
    if "nearest_neighbor_distance" in out.columns:
        out["_dense_keep_score"] = pd.to_numeric(out["nearest_neighbor_distance"], errors="coerce").fillna(0)
    elif "nnd" in out.columns:
        out["_dense_keep_score"] = pd.to_numeric(out["nnd"], errors="coerce").fillna(0)
    else:
        out["_dense_keep_score"] = np.arange(len(out), dtype=float)

    # Boundary score
    if "distance_to_boundary" in out.columns:
        out["_boundary_keep_score"] = pd.to_numeric(out["distance_to_boundary"], errors="coerce").fillna(0)
    elif "boundary_distance" in out.columns:
        out["_boundary_keep_score"] = pd.to_numeric(out["boundary_distance"], errors="coerce").fillna(0)
    else:
        out["_boundary_keep_score"] = np.arange(len(out), dtype=float)

    return out


def select_keep_by_subset(df, completeness, policy, seed=None):
    frac = completeness / 100.0
    kept = []

    for subset, g in df.groupby("subset", sort=False):
        n = len(g)
        if n == 0:
            continue

        k = int(round(n * frac))
        k = max(0, min(n, k))

        if completeness >= 100:
            keep_g = g.copy()

        elif policy == "random":
            rng = np.random.default_rng(seed)
            keep_idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
            keep_g = g.loc[keep_idx].copy()

        elif policy == "weak_evidence_first":
            keep_g = g.sort_values("_weak_keep_score", ascending=False).head(k).copy()

        elif policy == "dense_first":
            keep_g = g.sort_values("_dense_keep_score", ascending=False).head(k).copy()

        elif policy == "boundary_first":
            keep_g = g.sort_values("_boundary_keep_score", ascending=False).head(k).copy()

        else:
            raise ValueError(f"Unknown policy: {policy}")

        kept.append(keep_g)

    if not kept:
        return df.iloc[0:0].copy()

    return pd.concat(kept, ignore_index=True)


def save_degraded_lenient_references(lenient_df, levels, seeds):
    OUT_REF_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for policy in POLICY_ORDER:
        policy_dir = OUT_REF_DIR / policy
        policy_dir.mkdir(parents=True, exist_ok=True)

        if policy == "random":
            for level in levels:
                for seed in seeds:
                    degraded = select_keep_by_subset(lenient_df, level, policy, seed=seed)

                    rel_path = Path("dataset/annotations/degraded_references/lenient") / policy / f"lenient_{level}_seed{seed:02d}.csv"
                    abs_path = WORK / rel_path
                    degraded.to_csv(abs_path, index=False)

                    rows.append({
                        "path": str(rel_path),
                        "base_reference": "lenient",
                        "policy": policy,
                        "completeness": int(level),
                        "seed": int(seed),
                    })
        else:
            for level in levels:
                degraded = select_keep_by_subset(lenient_df, level, policy, seed=-999)

                rel_path = Path("dataset/annotations/degraded_references/lenient") / policy / f"lenient_{level}.csv"
                abs_path = WORK / rel_path
                degraded.to_csv(abs_path, index=False)

                rows.append({
                    "path": str(rel_path),
                    "base_reference": "lenient",
                    "policy": policy,
                    "completeness": int(level),
                    "seed": -999,
                })

    idx_lenient = pd.DataFrame(rows)
    print(f"[SAVE] lenient degraded references: {OUT_REF_DIR}")
    print(f"[INFO] n degraded reference files: {len(idx_lenient)}")
    return idx_lenient


def match_points(pred_xy, ref_xy, radius):
    if len(pred_xy) == 0 and len(ref_xy) == 0:
        return 0, 0, 0
    if len(pred_xy) == 0:
        return 0, 0, len(ref_xy)
    if len(ref_xy) == 0:
        return 0, len(pred_xy), 0

    tree = cKDTree(ref_xy)
    matched = set()
    tp = 0

    for p in pred_xy:
        cand = tree.query_ball_point(p, r=radius)
        cand = [i for i in cand if i not in matched]
        if not cand:
            continue
        d = np.linalg.norm(ref_xy[cand] - p, axis=1)
        best = cand[int(np.argmin(d))]
        matched.add(best)
        tp += 1

    fp = len(pred_xy) - tp
    fn = len(ref_xy) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate_one_run(pred, ref):
    rows = []

    subsets = sorted(ref["subset"].dropna().unique().tolist())

    for subset in subsets + ["ALL"]:
        if subset == "ALL":
            pred_s = pred
            ref_s = ref
        else:
            pred_s = pred[pred["subset"] == subset]
            ref_s = ref[ref["subset"] == subset]

        for method, pred_m in pred_s.groupby("method"):
            tp_total, fp_total, fn_total = 0, 0, 0

            image_names = sorted(set(pred_m["image_name"]) | set(ref_s["image_name"]))

            pred_groups = {
                k: v[["x", "y"]].to_numpy(float)
                for k, v in pred_m.groupby("image_name")
            }
            ref_groups = {
                k: v[["x", "y"]].to_numpy(float)
                for k, v in ref_s.groupby("image_name")
            }

            for img in image_names:
                pred_xy = pred_groups.get(img, np.empty((0, 2)))
                ref_xy = ref_groups.get(img, np.empty((0, 2)))

                tp, fp, fn = match_points(pred_xy, ref_xy, MATCH_RADIUS)
                tp_total += tp
                fp_total += fp
                fn_total += fn

            precision, recall, f1 = prf(tp_total, fp_total, fn_total)

            rows.append({
                "subset": subset,
                "method": method,
                "global_f1": f1,
                "global_precision": precision,
                "global_recall": recall,
            })

    return pd.DataFrame(rows)


def evaluate_lenient_curves(pred, idx_lenient):
    all_rows = []

    for _, row in idx_lenient.iterrows():
        ref_path = WORK / row["path"]
        ref = standardize_reference(ref_path)

        metrics = evaluate_one_run(pred, ref)
        metrics["base_reference"] = "lenient"
        metrics["policy"] = row["policy"]
        metrics["completeness"] = int(row["completeness"])
        metrics["seed"] = int(row["seed"])

        all_rows.append(metrics)

        print(
            f"[EVAL] lenient | {row['policy']:20s} | "
            f"{int(row['completeness']):3d}% | seed={int(row['seed']):3d}"
        )

    per_run = pd.concat(all_rows, ignore_index=True)

    curves = (
        per_run
        .groupby(["subset", "base_reference", "policy", "completeness", "method"], as_index=False)
        .agg(
            global_f1_mean=("global_f1", "mean"),
            global_f1_std=("global_f1", "std"),
            global_precision_mean=("global_precision", "mean"),
            global_recall_mean=("global_recall", "mean"),
            n_runs=("global_f1", "size"),
        )
    )

    curves["global_f1_std"] = curves["global_f1_std"].fillna(0.0)
    curves["method_label"] = curves["method"].map(METHOD_LABELS).fillna(curves["method"])

    curves = curves[
        [
            "subset",
            "base_reference",
            "policy",
            "completeness",
            "method",
            "method_label",
            "global_f1_mean",
            "global_f1_std",
            "global_precision_mean",
            "global_recall_mean",
            "n_runs",
        ]
    ]

    return curves


def append_to_existing_master(curves_lenient, idx_lenient):
    old_master = pd.read_csv(OLD_MASTER_PATH)
    old_index = pd.read_csv(OLD_INDEX_PATH)

    old_master = old_master[old_master["base_reference"] != "lenient"].copy()
    old_index = old_index[old_index["base_reference"] != "lenient"].copy()

    combined_master = pd.concat([old_master, curves_lenient], ignore_index=True)
    combined_index = pd.concat([old_index, idx_lenient], ignore_index=True)

    combined_master.to_csv(OUT_MASTER_WITH_LENIENT, index=False)
    combined_index.to_csv(OUT_INDEX_WITH_LENIENT, index=False)
    curves_lenient.to_csv(OUT_LENIENT_CURVES, index=False)

    print(f"[SAVE] {OUT_LENIENT_CURVES}")
    print(f"[SAVE] {OUT_MASTER_WITH_LENIENT}")
    print(f"[SAVE] {OUT_INDEX_WITH_LENIENT}")

    return combined_master


def plot_all_methods_curves(master):
    for subset in ["ALL", "MERFISH", "RCA", "SeqFISH"]:
        df_subset = master[master["subset"] == subset].copy()
        if df_subset.empty:
            continue

        fig, axes = plt.subplots(
            nrows=3,
            ncols=4,
            figsize=(22, 13),
            sharex=True,
            sharey=True,
        )

        for i, ref in enumerate(REFERENCE_ORDER):
            for j, policy in enumerate(POLICY_ORDER):
                ax = axes[i, j]

                sub = df_subset[
                    (df_subset["base_reference"] == ref) &
                    (df_subset["policy"] == policy)
                ].copy()

                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{REFERENCE_TITLES[ref]}\n{POLICY_TITLES[policy]}")
                    continue

                for method in sorted(sub["method"].unique()):
                    m = sub[sub["method"] == method].sort_values("completeness")
                    label = METHOD_LABELS.get(method, method)
                    ax.plot(
                        m["completeness"],
                        m["global_f1_mean"],
                        marker="o",
                        linewidth=1.8,
                        markersize=4,
                        label=label,
                    )

                ax.set_title(f"{REFERENCE_TITLES[ref]}\n{POLICY_TITLES[policy]}", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(48, 102)
                ax.set_ylim(0, 1.02)

                if i == 2:
                    ax.set_xlabel("Reference completeness (%)")
                if j == 0:
                    ax.set_ylabel("Global F1")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, 0.0),
        )

        fig.suptitle(f"{subset}: F1 curves across reference missingness conditions", fontsize=18)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])

        out_png = OUT_FIG_DIR / f"fig_{subset}_all_methods_f1_curves_raw_lenient_strict.png"
        out_pdf = OUT_FIG_DIR / f"fig_{subset}_all_methods_f1_curves_raw_lenient_strict.pdf"

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"[SAVE] {out_png}")
        print(f"[SAVE] {out_pdf}")


def main():
    print("[INFO] Add Lenient missingness curves")
    print(f"WORK = {WORK}")

    if not PRED_PATH.exists():
        raise FileNotFoundError(PRED_PATH)
    if not OLD_MASTER_PATH.exists():
        raise FileNotFoundError(OLD_MASTER_PATH)
    if not OLD_INDEX_PATH.exists():
        raise FileNotFoundError(OLD_INDEX_PATH)

    lenient_ref_path = find_lenient_reference()
    lenient = standardize_reference(lenient_ref_path)
    lenient = add_policy_scores(lenient)

    print("\n[LENIENT REFERENCE SUMMARY]")
    print("n points:", len(lenient))
    print(lenient.groupby("subset").size().rename("n_ref").reset_index().to_string(index=False))

    levels, seeds = get_levels_and_seeds()
    print("\n[LEVELS]", levels)
    print("[RANDOM SEEDS]", seeds)

    pred = standardize_predictions(PRED_PATH)
    print("\n[PREDICTIONS]")
    print("n predictions:", len(pred))
    print("methods:", sorted(pred["method"].unique()))
    print("subsets:", sorted(pred["subset"].unique()))

    idx_lenient = save_degraded_lenient_references(lenient, levels, seeds)
    curves_lenient = evaluate_lenient_curves(pred, idx_lenient)
    master_lenient = append_to_existing_master(curves_lenient, idx_lenient)

    print("\n[CHECK base_reference]")
    print(master_lenient["base_reference"].value_counts())

    plot_all_methods_curves(master_lenient)

    print("\n[DONE]")
    print(f"Lenient curves: {OUT_LENIENT_CURVES}")
    print(f"Combined master: {OUT_MASTER_WITH_LENIENT}")
    print(f"Figures: {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
