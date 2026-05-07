import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "results" / "evaluation" / "missingness_all_methods" / "missingness_all_methods_f1_curves_master.csv"
OUT_DIR = PROJECT_ROOT / "figures"

OUT_PNG = os.path.join(OUT_DIR, "figure4A_missingness_curves.png")
OUT_PDF = os.path.join(OUT_DIR, "figure4A_missingness_curves.pdf")

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

print("[CURVE_DF]")
print(curve_df.head(20).to_string(index=False))

reference_display = {
    "raw": "Raw",
    "lenient": "Lenient",
    "strict": "Strict",
}

policy_display = {
    "random": "Random",
    "weak_evidence_first": "Weak",
    "dense_first": "Dense",
    "boundary_first": "Boundary",
}

policy_color = {
    "random": "C0",
    "weak_evidence_first": "C1",
    "dense_first": "C2",
    "boundary_first": "C4",
}

policy_marker = {
    "random": "o",
    "weak_evidence_first": "s",
    "dense_first": "^",
    "boundary_first": "D",
}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.5), sharey=True)

for ax, ref in zip(axes, reference_order):
    sub_ref = curve_df[curve_df[COL_REFERENCE] == ref].copy()

    for pol in policy_order:
        sub_pol = sub_ref[sub_ref[COL_POLICY] == pol].copy()
        sub_pol = sub_pol.sort_values("retained_pct", ascending=False)

        if len(sub_pol) == 0:
            continue

        ax.plot(
            sub_pol["retained_pct"],
            sub_pol["f1_plot"],
            marker=policy_marker[pol],
            linewidth=1.6,
            markersize=4.2,
            color=policy_color[pol],
            label=policy_display[pol],
        )

    ax.set_title(reference_display[ref], fontsize=18, fontweight="normal")

    # ax.set_xlabel("Retained reference (%)", fontsize=18)

    ax.set_xticks(level_order)
    ax.set_xlim(102, 48)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=15)

axes[1].set_xlabel("Retained reference (%)", fontsize=18)

axes[0].set_ylabel("F1", fontsize=18)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles, labels,
    loc="lower left",
    fontsize=10,
    frameon=True,
    borderpad=0.35,
    handlelength=1.4,
    labelspacing=0.3
)

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
plt.close()

print("[SAVE]", OUT_PNG)
print("[SAVE]", OUT_PDF)
print("[DONE]")