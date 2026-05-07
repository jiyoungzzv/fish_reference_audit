from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


ROOT = PROJECT_ROOT

F1_CANDIDATES = [
    ROOT / "results/evaluation/missingness_all_methods/missingness_all_methods_f1_curves_master.csv",
    ROOT / "results/evaluation/missingness_all_methods/missingness_all_methods_f1_curves_master.csv",
]

RANK_CANDIDATES = [
    ROOT / "results/evaluation/missingness_all_methods/degraded_reference_index.csv",
    ROOT / "results/evaluation/missingness_all_methods/degraded_reference_index.csv",
]

OUT_DIR = PROJECT_ROOT / "figures" / "appendix_redraw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Helper functions
def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"No existing file found in: {paths}")


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of {candidates} found. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_reference(x):
    s = str(x).strip().lower()
    if s == "raw":
        return "Raw"
    if s == "strict":
        return "Strict"
    if s == "lenient":
        return "Lenient"
    return str(x)


def normalize_policy(x):
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")

    mapping = {
        "random": "Random",
        "rand": "Random",
        "rnd": "Random",

        "weak": "Weak-evidence",
        "weak_evidence": "Weak-evidence",
        "weak_evidence_first": "Weak-evidence",
        "weakevidence": "Weak-evidence",
        "weakevidencefirst": "Weak-evidence",

        "dense": "Dense-first",
        "dense_first": "Dense-first",
        "densefirst": "Dense-first",

        "boundary": "Boundary-first",
        "boundary_first": "Boundary-first",
        "boundaryfirst": "Boundary-first",
        "bdry": "Boundary-first",
    }

    return mapping.get(s, str(x))


def normalize_method(x):
    s0 = str(x).strip()
    s = s0.lower().replace("-", "_").replace(" ", "_").replace("+", "_plus_")

    mapping = {
        "unet_heatmap": "U-Net",
        "u_net": "U-Net",
        "unet": "U-Net",
        "u_net_heatmap": "U-Net",

        "unet_heatmap_pos": "U-Net+POS",
        "unet_heatmap_plus_pos": "U-Net+POS",
        "u_net_heatmap_pos": "U-Net+POS",
        "u_net_plus_pos": "U-Net+POS",
        "unet_pos": "U-Net+POS",
        "u_net_pos": "U-Net+POS",
        "u_net_pos_": "U-Net+POS",

        "yolo_pointbox": "YOLO",
        "yolo_point_box": "YOLO",
        "yolo": "YOLO",

        "logdog": "LoG/DoG",
        "log_dog": "LoG/DoG",
        "log/dog": "LoG/DoG",
        "log_dog_": "LoG/DoG",

        "rule_candidate": "Rule-cand.",
        "rule_candidate_": "Rule-cand.",
        "rule_cand": "Rule-cand.",

        "tophat_peak": "Top-hat",
        "top_hat_peak": "Top-hat",
        "top_hat": "Top-hat",
        "tophat": "Top-hat",

        "intensity_peak": "Intensity",
        "intensity": "Intensity",

        "deepblink_official": "deepBlink",
        "deepblink": "deepBlink",
    }

    # already-pretty labels from method_label column
    if s0 in [
        "U-Net", "YOLO", "U-Net+POS", "U-Net + POS",
        "LoG/DoG", "Rule-cand.", "Top-hat", "Intensity", "deepBlink"
    ]:
        if s0 == "U-Net + POS":
            return "U-Net+POS"
        return s0

    return mapping.get(s, s0)


def spearman_from_scores(base_scores, level_scores):
    """
    Compute Spearman rank correlation between two method-score series.
    Both inputs: pd.Series indexed by method name.
    """
    common = sorted(set(base_scores.index) & set(level_scores.index))
    if len(common) < 2:
        return np.nan

    a = base_scores.loc[common].rank(ascending=False, method="average")
    b = level_scores.loc[common].rank(ascending=False, method="average")

    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan

    return float(a.corr(b, method="pearson"))


# Orders
SUBSET_ORDER = ["MERFISH", "RCA", "SeqFISH"]

REF_ORDER = ["Raw", "Lenient", "Strict"]

POLICY_ORDER = [
    "Random",
    "Weak-evidence",
    "Dense-first",
    "Boundary-first",
]

METHOD_ORDER = [
    "U-Net",
    "YOLO",
    "U-Net+POS",
    "LoG/DoG",
    "Rule-cand.",
    "Top-hat",
    "Intensity",
    "deepBlink",
]

LEVEL_ORDER_ASC = [50, 60, 70, 80, 90, 100]
LEVEL_ORDER_DESC = [100, 90, 80, 70, 60, 50]


# Load data
F1_PATH = first_existing(F1_CANDIDATES)
RANK_PATH = first_existing(RANK_CANDIDATES)

print(f"[LOAD F1]   {F1_PATH}")
print(f"[LOAD RANK] {RANK_PATH}")

df_f1 = pd.read_csv(F1_PATH)
df_rank = pd.read_csv(RANK_PATH)

print("\n[F1 columns]")
print(list(df_f1.columns))

print("\n[RANK columns]")
print(list(df_rank.columns))


# Detect F1 columns
subset_col_f1 = pick_col(df_f1, ["subset"])
ref_col_f1 = pick_col(df_f1, ["base_reference", "ref_family", "reference", "reference_view"])
policy_col_f1 = pick_col(df_f1, ["policy"])
level_col_f1 = pick_col(df_f1, ["level", "completeness", "alpha", "retained_fraction"])
method_col_f1 = pick_col(df_f1, ["method_label", "method", "detector"])
value_col_f1 = pick_col(df_f1, ["f1_mean", "global_f1_mean", "f1"])

df_f1 = df_f1.copy()
df_f1["subset_norm"] = df_f1[subset_col_f1].astype(str)
df_f1["ref_norm"] = df_f1[ref_col_f1].map(normalize_reference)
df_f1["policy_norm"] = df_f1[policy_col_f1].map(normalize_policy)
df_f1["method_norm"] = df_f1[method_col_f1].map(normalize_method)
df_f1["level_norm"] = pd.to_numeric(df_f1[level_col_f1], errors="coerce")
df_f1["f1_value"] = pd.to_numeric(df_f1[value_col_f1], errors="coerce")


# Detect rank columns or compute Spearman fallback
rank_ok = True

try:
    subset_col_rank = pick_col(df_rank, ["subset"])
    ref_col_rank = pick_col(df_rank, ["base_reference", "ref_family", "reference", "reference_view"])
    policy_col_rank = pick_col(df_rank, ["policy"])
    level_col_rank = pick_col(df_rank, ["level", "completeness", "alpha", "retained_fraction"])

    value_col_rank = pick_col(
        df_rank,
        [
            "spearman_mean",
            "spearman",
            "spearman_rank_corr",
            "spearman_rank_correlation",
            "rank_corr_mean",
            "rank_correlation_mean",
            "mean_rank_correlation",
            "rho_mean",
            "rho",
        ],
        required=False,
    )

    if value_col_rank is None:
        rank_ok = False

except Exception as e:
    print("[WARN] Could not parse rank table. Will compute Spearman from F1 table.")
    print(e)
    rank_ok = False


if rank_ok:
    print(f"\n[USE RANK FILE] Spearman column = {value_col_rank}")

    df_rank = df_rank.copy()
    df_rank["subset_norm"] = df_rank[subset_col_rank].astype(str)
    df_rank["ref_norm"] = df_rank[ref_col_rank].map(normalize_reference)
    df_rank["policy_norm"] = df_rank[policy_col_rank].map(normalize_policy)
    df_rank["level_norm"] = pd.to_numeric(df_rank[level_col_rank], errors="coerce")
    df_rank["spearman_value"] = pd.to_numeric(df_rank[value_col_rank], errors="coerce")

else:
    print("\n[COMPUTE] Spearman rank correlation from F1 table")

    rows = []
    for subset in sorted(df_f1["subset_norm"].dropna().unique()):
        for ref in REF_ORDER:
            for policy in POLICY_ORDER:
                cell = df_f1[
                    (df_f1["subset_norm"] == subset) &
                    (df_f1["ref_norm"] == ref) &
                    (df_f1["policy_norm"] == policy)
                ].copy()

                if cell.empty:
                    continue

                base = cell[cell["level_norm"] == 100]
                if base.empty:
                    continue

                base_scores = base.groupby("method_norm")["f1_value"].mean()

                for level in sorted(cell["level_norm"].dropna().unique()):
                    lvl = cell[cell["level_norm"] == level]
                    level_scores = lvl.groupby("method_norm")["f1_value"].mean()

                    rho = spearman_from_scores(base_scores, level_scores)

                    rows.append({
                        "subset_norm": subset,
                        "ref_norm": ref,
                        "policy_norm": policy,
                        "level_norm": level,
                        "spearman_value": rho,
                    })

    df_rank = pd.DataFrame(rows)


# Plot style
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# Draw F1 curves
def draw_f1_subset(subset_name):
    dsub = df_f1[df_f1["subset_norm"] == subset_name].copy()

    if dsub.empty:
        print(f"[SKIP] No F1 data for subset: {subset_name}")
        return

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(22, 15),
        sharex=True,
        sharey=True,
    )

    fig.suptitle(
        f"{subset_name}: subset-wise F1 curves under structured missingness",
        fontsize=22,
        y=0.98,
    )

    handles_for_legend = {}

    if subset_name == "RCA":
        y_min, y_max = 0.0, 0.90
    else:
        y_min, y_max = 0.0, 1.00

    for r, ref in enumerate(REF_ORDER):
        for c, policy in enumerate(POLICY_ORDER):
            ax = axes[r, c]

            cell = dsub[
                (dsub["ref_norm"] == ref) &
                (dsub["policy_norm"] == policy)
            ].copy()

            for method in METHOD_ORDER:
                dm = cell[cell["method_norm"] == method].copy()
                if dm.empty:
                    continue

                dm = dm.sort_values("level_norm")

                line, = ax.plot(
                    dm["level_norm"],
                    dm["f1_value"],
                    marker="o",
                    linewidth=2.2,
                    markersize=4,
                )

                if method not in handles_for_legend:
                    handles_for_legend[method] = line

            if r == 0:
                ax.set_title(policy, fontsize=18, pad=10)

            if c == 0:
                ax.set_ylabel(f"{ref}\nF1", fontsize=16, color="black")

            if r == 2:
                ax.set_xlabel("Reference completeness (%)", fontsize=15)

            ax.set_xlim(48, 102)
            ax.set_xticks(LEVEL_ORDER_ASC)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25)

    legend_handles = [
        handles_for_legend[m]
        for m in METHOD_ORDER
        if m in handles_for_legend
    ]
    legend_labels = [
        m
        for m in METHOD_ORDER
        if m in handles_for_legend
    ]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    png_path = OUT_DIR / f"app_f1_{subset_name}.png"
    pdf_path = OUT_DIR / f"app_f1_{subset_name}.pdf"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")


# Draw separated Spearman heatmaps
def draw_spearman_heatmap(subset_name, ref_name):
    d = df_rank[
        (df_rank["subset_norm"] == subset_name) &
        (df_rank["ref_norm"] == ref_name)
    ].copy()

    if d.empty:
        print(f"[SKIP] No rank data for subset={subset_name}, ref={ref_name}")
        return

    pivot = (
        d.pivot_table(
            index="level_norm",
            columns="policy_norm",
            values="spearman_value",
            aggfunc="mean",
        )
        .reindex(index=LEVEL_ORDER_DESC, columns=POLICY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.7))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="YlGnBu",
    )

    ax.set_title(f"{subset_name} – {ref_name} reference", fontsize=18, pad=12)
    ax.set_xlabel("Missingness policy", fontsize=15)
    ax.set_ylabel("Reference completeness (%)", fontsize=15)

    ax.set_xticks(np.arange(len(POLICY_ORDER)))
    ax.set_xticklabels(
        ["Random", "Weak", "Dense", "Boundary"],
        rotation=25,
        ha="right",
        color="black",
    )

    ax.set_yticks(np.arange(len(LEVEL_ORDER_DESC)))
    ax.set_yticklabels(
        [str(x) for x in LEVEL_ORDER_DESC],
        color="black",
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]

            if pd.isna(val):
                text = "--"
                color = "black"
            else:
                text = f"{val:.2f}"
                color = "white" if val >= 0.70 else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=12,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    ref_tag = ref_name.lower()

    png_path = OUT_DIR / f"app_spearman_{subset_name}_{ref_tag}.png"
    pdf_path = OUT_DIR / f"app_spearman_{subset_name}_{ref_tag}.pdf"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")


def main():
    print("\n[CHECK] Available normalized F1 values")
    print("Subsets:", sorted(df_f1["subset_norm"].dropna().unique()))
    print("References:", sorted(df_f1["ref_norm"].dropna().unique()))
    print("Policies:", sorted(df_f1["policy_norm"].dropna().unique()))
    print("Completeness:", sorted(df_f1["level_norm"].dropna().unique()))
    print("Methods:", sorted(df_f1["method_norm"].dropna().unique()))

    print("\n[DRAW] F1 curves")
    for subset in SUBSET_ORDER:
        draw_f1_subset(subset)

    print("\n[DRAW] Spearman heatmaps")
    for subset in SUBSET_ORDER:
        for ref in REF_ORDER:
            draw_spearman_heatmap(subset, ref)

    print("\n[DONE] Appendix redraw figures saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()