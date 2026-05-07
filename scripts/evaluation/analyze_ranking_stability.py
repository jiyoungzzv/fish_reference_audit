from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"

OUT_DIR = PROJECT_ROOT / "results" / "ranking_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Rule-candidate": EVAL_DIR / "rule_candidate_baseline_reference_summary.csv",
    "LoG/DoG": EVAL_DIR / "logdog_baseline_reference_summary.csv",
    "U-Net heatmap": EVAL_DIR / "unet_heatmap_full_xyfixed_thr090_reference_summary.csv",
}

OUT_LONG = OUT_DIR / "all_models_all_references_long.csv"
OUT_RANKING = OUT_DIR / "all_models_all_references_ranking.csv"
OUT_STABILITY = OUT_DIR / "ranking_stability_vs_full_reference.csv"
OUT_TOP1 = OUT_DIR / "top1_stability_summary.csv"
OUT_PIVOT = OUT_DIR / "f1_curves_pivot.csv"


def load_model_summary(model_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model summary: {path}")

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["model_name"] = model_name

    required = [
        "base_reference", "policy", "completeness", "seed", "subset",
        "global_precision", "global_recall", "global_f1",
        "total_ref", "total_pred", "total_tp", "total_fp", "total_fn"
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{model_name} summary missing column: {c}")

    return df


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    group_cols = ["base_reference", "policy", "completeness", "seed", "subset"]

    out["rank_by_f1"] = (
        out
        .groupby(group_cols)["global_f1"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    out = out.sort_values(group_cols + ["rank_by_f1", "model_name"]).reset_index(drop=True)
    return out


def safe_rank_corr(rank_a: pd.Series, rank_b: pd.Series, method: str):
    """
    rank_a/rank_b index must be model_name.
    Returns correlation if at least 2 common models exist.
    """
    common = sorted(set(rank_a.index) & set(rank_b.index))
    if len(common) < 2:
        return np.nan

    a = rank_a.loc[common].astype(float)
    b = rank_b.loc[common].astype(float)

    if a.nunique() <= 1 or b.nunique() <= 1:
        return np.nan

    return a.corr(b, method=method)


def get_top_model(group: pd.DataFrame) -> str:
    g = group.sort_values(["rank_by_f1", "model_name"])
    return g.iloc[0]["model_name"]


def main():
    print("===== Ranking stability analysis =====")

    # 1. Load summaries
    parts = []
    print("\n[1] Load model summaries")
    for model_name, path in MODEL_FILES.items():
        print(f"- {model_name}: {path}")
        parts.append(load_model_summary(model_name, path))

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUT_LONG}")

    print("\nLoaded rows:", len(df))
    print(df[["model_name", "base_reference", "policy", "completeness", "seed", "subset"]].head())

    # 2. Rank models for every reference setting
    ranking = rank_models(df)
    ranking.to_csv(OUT_RANKING, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUT_RANKING}")

    # 3. Full-reference top models
    #    baseline = same base_reference, same subset, completeness=100,
    #               policy=random, seed=0
    full_ref = ranking[
        (ranking["completeness"] == 100)
        & (ranking["policy"] == "random")
        & (ranking["seed"] == 0)
    ].copy()

    baseline_rank = {}
    baseline_top = {}

    for (base_reference, subset), g in full_ref.groupby(["base_reference", "subset"]):
        s = g.set_index("model_name")["rank_by_f1"]
        baseline_rank[(base_reference, subset)] = s
        baseline_top[(base_reference, subset)] = get_top_model(g)

    # 4. Stability for every setting vs its own full reference
    stability_rows = []

    group_cols = ["base_reference", "policy", "completeness", "seed", "subset"]

    for key, g in ranking.groupby(group_cols):
        base_reference, policy, completeness, seed, subset = key

        current_rank = g.set_index("model_name")["rank_by_f1"]
        current_top = get_top_model(g)

        ref_key = (base_reference, subset)
        ref_rank = baseline_rank.get(ref_key, None)
        ref_top = baseline_top.get(ref_key, None)

        if ref_rank is None:
            spearman = np.nan
            kendall = np.nan
            top1_same = np.nan
            rank_flip_count = np.nan
            ref_top_model = None
        else:
            spearman = safe_rank_corr(current_rank, ref_rank, method="spearman")
            kendall = safe_rank_corr(current_rank, ref_rank, method="kendall")

            ref_top_model = ref_top
            top1_same = int(current_top == ref_top_model)

            common = sorted(set(current_rank.index) & set(ref_rank.index))
            rank_flip_count = int(
                sum(current_rank.loc[m] != ref_rank.loc[m] for m in common)
            )

        stability_rows.append({
            "base_reference": base_reference,
            "policy": policy,
            "completeness": completeness,
            "seed": seed,
            "subset": subset,
            "current_top_model": current_top,
            "full_reference_top_model": ref_top_model,
            "top1_same_as_full_reference": top1_same,
            "spearman_vs_full_reference": spearman,
            "kendall_vs_full_reference": kendall,
            "rank_flip_count_vs_full_reference": rank_flip_count,
        })

    stability = pd.DataFrame(stability_rows)
    stability = stability.sort_values(
        ["base_reference", "subset", "policy", "completeness", "seed"]
    ).reset_index(drop=True)

    stability.to_csv(OUT_STABILITY, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUT_STABILITY}")

    # 5. Top-1 stability summary
    top1_summary = (
        stability
        .groupby(["base_reference", "policy", "completeness", "subset"])
        .agg(
            n_settings=("top1_same_as_full_reference", "count"),
            top1_agreement_rate=("top1_same_as_full_reference", "mean"),
            mean_spearman=("spearman_vs_full_reference", "mean"),
            mean_kendall=("kendall_vs_full_reference", "mean"),
            mean_rank_flip_count=("rank_flip_count_vs_full_reference", "mean"),
        )
        .reset_index()
        .sort_values(["base_reference", "subset", "policy", "completeness"])
    )

    top1_summary.to_csv(OUT_TOP1, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUT_TOP1}")

    # 6. F1 pivot for plotting curves later
    pivot = ranking.pivot_table(
        index=["base_reference", "policy", "completeness", "seed", "subset"],
        columns="model_name",
        values="global_f1",
        aggfunc="first"
    ).reset_index()

    pivot.to_csv(OUT_PIVOT, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUT_PIVOT}")

    # Print key outputs
    print("\n===== Full-reference ranking, 100% only =====")
    full_show = full_ref[
        [
            "base_reference", "subset", "rank_by_f1", "model_name",
            "global_precision", "global_recall", "global_f1"
        ]
    ].sort_values(["base_reference", "subset", "rank_by_f1"])
    print(full_show.to_string(index=False))

    print("\n===== Top-1 stability summary preview =====")
    preview = top1_summary[
        top1_summary["completeness"].isin([50, 60, 70, 80, 90, 100])
    ].copy()
    print(preview.head(60).to_string(index=False))

    print("\n===== Cases where top-1 changed vs full reference =====")
    changed = stability[stability["top1_same_as_full_reference"] == 0].copy()
    if len(changed) == 0:
        print("No top-1 changes found.")
    else:
        print(
            changed[
                [
                    "base_reference", "subset", "policy", "completeness", "seed",
                    "current_top_model", "full_reference_top_model",
                    "spearman_vs_full_reference", "kendall_vs_full_reference",
                    "rank_flip_count_vs_full_reference"
                ]
            ].to_string(index=False)
        )

    print("\nDone.")


if __name__ == "__main__":
    main()