from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "results" / "evaluation" / "missingness_all_methods"
OUT_DIR = PROJECT_ROOT / "results" / "ranking_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANK_MASTER = SRC_DIR / "missingness_all_methods_rank_stability_master.csv"
F1_MASTER = SRC_DIR / "missingness_all_methods_f1_curves_master.csv"

OUT_STABILITY = OUT_DIR / "ranking_stability_vs_full_reference.csv"
OUT_TOP1 = OUT_DIR / "top1_stability_summary.csv"
OUT_LONG = OUT_DIR / "all_models_all_references_long.csv"
OUT_PIVOT = OUT_DIR / "f1_curves_pivot.csv"


def pick_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def main():
    print("===== Ranking stability analysis =====")

    if not RANK_MASTER.exists():
        raise FileNotFoundError(
            f"Missing rank-stability master file: {RANK_MASTER}\n"
            "Please run this first:\n"
            "  python scripts/evaluation/evaluate_missingness_all_methods.py"
        )

    print(f"[LOAD] {RANK_MASTER}")
    stability = pd.read_csv(RANK_MASTER)
    stability.to_csv(OUT_STABILITY, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUT_STABILITY}")

    cols = list(stability.columns)

    group_cols = [
        c for c in ["base_reference", "policy", "completeness", "subset"]
        if c in stability.columns
    ]

    top1_col = pick_first_existing(
        cols,
        [
            "top1_same_as_full_reference",
            "top1_agreement",
            "top1_same",
            "top1_match",
        ],
    )

    spearman_col = next((c for c in cols if "spearman" in c.lower()), None)
    kendall_col = next((c for c in cols if "kendall" in c.lower()), None)
    flip_col = next((c for c in cols if "flip" in c.lower()), None)

    if group_cols and top1_col is not None:
        agg_dict = {
            "n_settings": (top1_col, "count"),
            "top1_agreement_rate": (top1_col, "mean"),
        }

        if spearman_col is not None:
            agg_dict["mean_spearman"] = (spearman_col, "mean")
        if kendall_col is not None:
            agg_dict["mean_kendall"] = (kendall_col, "mean")
        if flip_col is not None:
            agg_dict["mean_rank_flip_count"] = (flip_col, "mean")

        top1_summary = (
            stability
            .groupby(group_cols)
            .agg(**agg_dict)
            .reset_index()
            .sort_values(group_cols)
        )

        top1_summary.to_csv(OUT_TOP1, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {OUT_TOP1}")
    else:
        print("[SKIP] Could not build top-1 summary because required columns were not found.")
        print("[INFO] Available columns:")
        print(cols)

    if F1_MASTER.exists():
        print(f"[LOAD] {F1_MASTER}")
        f1 = pd.read_csv(F1_MASTER)
        f1.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {OUT_LONG}")

        method_col = pick_first_existing(
            list(f1.columns),
            ["method", "method_name", "model_name", "method_label"],
        )

        value_col = pick_first_existing(
            list(f1.columns),
            ["global_f1", "global_f1_mean", "f1", "mean_f1"],
        )

        index_cols = [
            c for c in ["base_reference", "policy", "completeness", "seed", "subset"]
            if c in f1.columns
        ]

        if method_col is not None and value_col is not None and index_cols:
            pivot = (
                f1.pivot_table(
                    index=index_cols,
                    columns=method_col,
                    values=value_col,
                    aggfunc="first",
                )
                .reset_index()
            )

            pivot.to_csv(OUT_PIVOT, index=False, encoding="utf-8-sig")
            print(f"[SAVED] {OUT_PIVOT}")
        else:
            print("[SKIP] Could not build F1 pivot because required columns were not found.")
            print("[INFO] Available F1 columns:")
            print(list(f1.columns))
    else:
        print(f"[SKIP] F1 master file not found: {F1_MASTER}")

    print("\nDone.")


if __name__ == "__main__":
    main()
