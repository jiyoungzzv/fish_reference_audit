import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "results" / "evaluation" / "missingness_all_methods" / "missingness_all_methods_f1_curves_master.csv"
OUT_DIR = PROJECT_ROOT / "figures"

OUT_PNG = os.path.join(OUT_DIR, "figure4B_missingness_drop_summary.png")
OUT_PDF = os.path.join(OUT_DIR, "figure4B_missingness_drop_summary.pdf")

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

print("[LOAD]", INPUT_CSV)
print("[COLUMNS]", df.columns.tolist())

COL_SUBSET = "subset"
COL_REFERENCE = "base_reference"
COL_POLICY = "policy"
COL_COMPLETENESS = "completeness"
COL_METHOD = "method"
COL_F1 = "global_f1_mean"

if "ALL" in df[COL_SUBSET].astype(str).unique():
    df = df[df[COL_SUBSET].astype(str) == "ALL"].copy()
    print("[FILTER] subset == ALL")

df[COL_REFERENCE] = df[COL_REFERENCE].astype(str).str.lower()

policy_map = {
    "random": "random",
    "Random": "random",

    "weak_evidence_first": "weak_evidence_first",
    "weak-evidence-first": "weak_evidence_first",
    "Weak-evidence-first": "weak_evidence_first",

    "dense_first": "dense_first",
    "dense-first": "dense_first",
    "Dense-first": "dense_first",

    "boundary_first": "boundary_first",
    "boundary-first": "boundary_first",
    "Boundary-first": "boundary_first",
}
df[COL_POLICY] = df[COL_POLICY].map(lambda x: policy_map.get(str(x), str(x)))

df[COL_COMPLETENESS] = pd.to_numeric(df[COL_COMPLETENESS], errors="coerce")
df[COL_F1] = pd.to_numeric(df[COL_F1], errors="coerce")

# completeness -> retained_pct
if df[COL_COMPLETENESS].max() <= 1.5:
    df["retained_pct"] = (df[COL_COMPLETENESS] * 100).round().astype(int)
else:
    df["retained_pct"] = df[COL_COMPLETENESS].round().astype(int)

reference_order = ["raw", "lenient", "strict"]
policy_order = ["random", "weak_evidence_first", "dense_first", "boundary_first"]
level_order = [100, 90, 80, 70, 60, 50]

df = df[df[COL_REFERENCE].isin(reference_order)].copy()
df = df[df[COL_POLICY].isin(policy_order)].copy()
df = df[df["retained_pct"].isin(level_order)].copy()

curve_df = (
    df.groupby([COL_REFERENCE, COL_POLICY, "retained_pct"], as_index=False)[COL_F1]
      .mean()
      .rename(columns={COL_F1: "f1_plot"})
)

pivot_df = curve_df.pivot_table(
    index=[COL_REFERENCE, COL_POLICY],
    columns="retained_pct",
    values="f1_plot"
).reset_index()

pivot_df["f1_drop"] = pivot_df[100] - pivot_df[50]

print(pivot_df[[COL_REFERENCE, COL_POLICY, 100, 50, "f1_drop"]].to_string(index=False))

reference_display = {
    "raw": "Raw",
    "strict": "Strict",
    "lenient": "Lenient",
}

policy_display = {
    "random": "Random",
    "weak_evidence_first": "Weak",
    "dense_first": "Dense",
    "boundary_first": "Boundary",
}

ref_colors = {
    "raw": "#A9B8D5",
    "strict": "#B2D2B6",
    "lenient": "#ECC6DF",
}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig, ax = plt.subplots(figsize=(8.2, 3.5))

x = np.arange(len(policy_order)) * 0.6
bar_width = 0.15

for i, ref in enumerate(reference_order):
    vals = []

    for pol in policy_order:
        row = pivot_df[
            (pivot_df[COL_REFERENCE] == ref) &
            (pivot_df[COL_POLICY] == pol)
        ]

        if len(row) == 0:
            vals.append(np.nan)
        else:
            vals.append(float(row["f1_drop"].values[0]))

    xpos = x + (i - 1) * bar_width

    ax.bar(
        xpos,
        vals,
        width=bar_width,
        label=reference_display[ref],
        color=ref_colors[ref],
        alpha=1.0,
        edgecolor="gray",
        linewidth=0.6
    )

    for xx, yy in zip(xpos, vals):
        if not np.isnan(yy):
            ax.text(xx, yy + 0.006, f"{yy:.2f}", ha="center", va="bottom", fontsize=13)

ax.set_ylabel("ΔF1", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels([policy_display[p] for p in policy_order], fontsize=18)
ax.grid(True, axis="y", alpha=0.25)
ax.tick_params(axis="y", labelsize=15)
ax.legend(frameon=True, fontsize=9)

ymax = pivot_df["f1_drop"].max()
ax.set_ylim(0.0, ymax * 1.2 if pd.notna(ymax) else 0.3)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
plt.close()

print("[SAVE]", OUT_PNG)
print("[SAVE]", OUT_PDF)
print("[DONE]")