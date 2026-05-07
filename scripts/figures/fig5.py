from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Fig. 5: Aggregate rank-correlation stability
# One-panel version: Raw / Lenient / Strict lines

CSV_PATH = PROJECT_ROOT / "results" / "evaluation" / "missingness_all_methods" / "missingness_all_methods_f1_curves_master.csv"

OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "fig5.png"
OUT_PDF = OUT_DIR / "fig5.pdf"
OUT_CSV = OUT_DIR / "fig5_rank_correlation_values.csv"


def pick_column(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            print(f"[COL] {name}: {c}")
            return c
    raise ValueError(
        f"Could not find column for {name}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_ref(x):
    x = str(x).lower().strip()
    mapping = {
        "raw": "raw",
        "strict": "strict",
        "lenient": "lenient",
    }
    return mapping.get(x, x)


def normalize_policy(x):
    x = str(x).lower().strip().replace("-", "_")

    mapping = {
        "random": "random",

        "weak": "weak_evidence_first",
        "weak_first": "weak_evidence_first",
        "weak_evidence": "weak_evidence_first",
        "weak_evidence_first": "weak_evidence_first",

        "dense": "dense_first",
        "density": "dense_first",
        "density_first": "dense_first",
        "dense_first": "dense_first",

        "boundary": "boundary_first",
        "boundary_first": "boundary_first",
    }

    return mapping.get(x, x)



print(f"[LOAD] {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print("[COLUMNS]")
print(df.columns.tolist())
print(df.head())


# Column detection

COL_REF = pick_column(
    df,
    ["base_reference", "ref_family", "reference", "reference_type"],
    "reference"
)

COL_POLICY = pick_column(
    df,
    ["policy", "missingness_policy", "removal_policy"],
    "policy"
)

COL_LEVEL = pick_column(
    df,
    ["completeness", "level", "retention", "retained_percent", "reference_completeness"],
    "reference completeness"
)

COL_METHOD = pick_column(
    df,
    ["method", "detector", "model", "detector_name", "method_name"],
    "method"
)

COL_F1 = pick_column(
    df,
    [
        "f1_mean",
        "global_f1_mean",
        "f1",
        "mean_f1",
        "f1_score",
        "f1_global",
        "f1_mean_global",
    ],
    "F1"
)

COL_SUBSET = None
for c in ["subset", "dataset", "data_subset"]:
    if c in df.columns:
        COL_SUBSET = c
        print(f"[COL] subset: {COL_SUBSET}")
        break

COL_SEED = None
for c in ["seed", "run", "repeat"]:
    if c in df.columns:
        COL_SEED = c
        print(f"[COL] seed: {COL_SEED}")
        break


# Clean

df = df.copy()

df[COL_REF] = df[COL_REF].map(normalize_ref)
df[COL_POLICY] = df[COL_POLICY].map(normalize_policy)
df[COL_METHOD] = df[COL_METHOD].astype(str).str.strip()
df[COL_LEVEL] = pd.to_numeric(df[COL_LEVEL], errors="coerce")
df[COL_F1] = pd.to_numeric(df[COL_F1], errors="coerce")

df = df.dropna(subset=[COL_REF, COL_POLICY, COL_LEVEL, COL_METHOD, COL_F1])

if df[COL_LEVEL].max() <= 1.5:
    df[COL_LEVEL] = df[COL_LEVEL] * 100

df[COL_LEVEL] = df[COL_LEVEL].round(6)


# Average F1 by condition first

group_cols = [COL_REF, COL_POLICY, COL_LEVEL, COL_METHOD]

if COL_SUBSET is not None:
    group_cols = [COL_SUBSET] + group_cols

if COL_SEED is not None:
    group_cols = group_cols + [COL_SEED]

f1_df = (
    df.groupby(group_cols, as_index=False)[COL_F1]
    .mean()
)

print("\n[CHECK F1]")
print(f1_df.head(20).to_string(index=False))


# Compute rank correlation

unit_cols = [COL_REF, COL_POLICY]

if COL_SUBSET is not None:
    unit_cols = [COL_SUBSET] + unit_cols

if COL_SEED is not None:
    unit_cols = unit_cols + [COL_SEED]

rho_rows = []

for keys, g in f1_df.groupby(unit_cols):
    if not isinstance(keys, tuple):
        keys = (keys,)

    key_dict = dict(zip(unit_cols, keys))

    complete_level = g[COL_LEVEL].max()
    complete = g[g[COL_LEVEL] == complete_level]
    complete_scores = complete.set_index(COL_METHOD)[COL_F1]

    for level, gl in g.groupby(COL_LEVEL):
        cur_scores = gl.set_index(COL_METHOD)[COL_F1]
        common_methods = sorted(set(complete_scores.index) & set(cur_scores.index))

        if len(common_methods) < 2:
            rho = float("nan")
        else:
            rho = cur_scores.loc[common_methods].corr(
                complete_scores.loc[common_methods],
                method="spearman"
            )

        row = dict(key_dict)
        row[COL_LEVEL] = level
        row["rank_corr"] = rho
        row["n_methods"] = len(common_methods)
        rho_rows.append(row)

rho_df = pd.DataFrame(rho_rows).dropna(subset=["rank_corr"])

print("\n[CHECK RANK CORR RAW]")
print(rho_df.head(30).to_string(index=False))


plot_df = (
    rho_df.groupby([COL_REF, COL_LEVEL], as_index=False)
    .agg(
        rank_corr_mean=("rank_corr", "mean"),
        rank_corr_std=("rank_corr", "std"),
        n_units=("rank_corr", "count"),
    )
)

plot_df.to_csv(OUT_CSV, index=False)
print(f"\n[SAVE] {OUT_CSV}")
print("\n[PLOT DATA]")
print(plot_df.to_string(index=False))


# Plot: one-panel, 3 lines
# darker colors + larger fonts

ref_order = ["raw", "lenient", "strict"]

label_map = {
    "raw": "Raw",
    "strict": "Strict",
    "lenient": "Lenient",
}

marker_map = {
    "raw": "o",
    "strict": "^",
    "lenient": "s",
}

color_map = {
    "raw": "#9EADD1",      # RGB(158, 173, 209)
    "strict": "#AFCCB2",   # RGB(175, 204, 178)
    "lenient": "#DFBAD5",  # RGB(223, 186, 213)
}

plt.figure(figsize=(5.4, 3.8))

for ref in ref_order:
    sub = plot_df[plot_df[COL_REF] == ref].copy()
    sub = sub.sort_values(COL_LEVEL, ascending=False)

    if len(sub) == 0:
        print(f"[WARN] No data for reference: {ref}")
        continue

    plt.plot(
        sub[COL_LEVEL],
        sub["rank_corr_mean"],
        marker=marker_map.get(ref, "o"),
        color=color_map.get(ref, "black"),
        linewidth=2.4,
        markersize=6.2,
        markeredgewidth=0.8,
        label=label_map.get(ref, ref),
    )

plt.axhline(1.0, linestyle="--", linewidth=1.2, color="#BDBDBD", alpha=0.9)
 
plt.title("", fontsize=13, pad=8)
plt.xlabel("Reference completeness (%)", fontsize=18)
plt.ylabel("Rank correlation", fontsize=18)

plt.xlim(100, 50)
plt.xticks([100, 90, 80, 70, 60, 50], fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0.0, 1.05)

plt.grid(True, alpha=0.22)
plt.legend(frameon=False, fontsize=13)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")

print(f"[SAVE] {OUT_PNG}")
print(f"[SAVE] {OUT_PDF}")

plt.show()