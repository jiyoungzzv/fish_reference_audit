from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]



WORK_DIR = PROJECT_ROOT

INPUT_METRICS = WORK_DIR / "results/figures/missingness/fig5_ranking_stability_metrics.csv"

OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "fig6B_top1_agreement_split_by_reference_fig6.png"
OUT_PDF = OUT_DIR / "fig6B_top1_agreement_split_by_reference_fig6.pdf"


# 1. Load
print(f"[LOAD] {INPUT_METRICS}")
df = pd.read_csv(INPUT_METRICS)

df["base_reference"] = df["base_reference"].astype(str).str.lower().str.strip()
df["policy"] = df["policy"].astype(str).str.lower().str.strip()
df["retained_reference"] = pd.to_numeric(df["retained_reference"], errors="coerce")

ref_order = ["raw", "lenient", "strict"]
policy_order = ["random", "weak", "dense", "boundary"]
level_order = sorted(df["retained_reference"].dropna().unique().tolist())

ref_label = {
    "raw": "Raw",
    "lenient": "Lenient",
    "strict": "Strict",
}

policy_label = {
    "random": "Random",
    "weak": "Weak",
    "dense": "Dense",
    "boundary": "Boundary",
}


# 2. Aggregate
heat_df = (
    df.groupby(["base_reference", "policy", "retained_reference"], as_index=False)
      .agg(top1_mean=("top1_agreement", "mean"))
)


# 3. Plot style: Fig. 3-like
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1, 3,
    figsize=(11.8, 2.9),
    sharey=True,
    gridspec_kw={"wspace": 0.05}
)

ims = []

for k, (ax, ref) in enumerate(zip(axes, ref_order)):
    sub = heat_df[heat_df["base_reference"] == ref]

    mat = (
        sub.pivot(index="policy", columns="retained_reference", values="top1_mean")
           .reindex(index=policy_order, columns=level_order)
    )

    im = ax.imshow(
        mat.values,
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        aspect="auto",
        interpolation="nearest"
    )
    ims.append(im)

    ax.set_title(ref_label[ref], pad=7, fontweight="normal")

    ax.set_xticks(np.arange(len(level_order)))
    ax.set_xticklabels([str(int(x)) for x in level_order], rotation=0)

    ax.set_yticks(np.arange(len(policy_order)))
    if k == 0:
        ax.set_yticklabels([policy_label[p] for p in policy_order])
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    ax.set_xlabel("Completeness (%)")

    ax.grid(False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            text = "NA" if pd.isna(val) else f"{val:.2f}"

            if pd.isna(val):
                txt_color = "black"
            elif val >= 0.65:
                txt_color = "white"
            else:
                txt_color = "black"

            ax.text(
                j, i, text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=txt_color
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

axes[0].set_ylabel("Missingness policy")

cbar = fig.colorbar(
    ims[0],
    ax=axes,
    fraction=0.026,
    pad=0.018
)
cbar.set_label("Top-1 agreement", fontsize=13)
cbar.ax.tick_params(labelsize=11)

fig.suptitle(
    "",
    fontsize=16,
    fontweight="normal",
    y=1.03
)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print(f"[SAVE] {OUT_PNG}")
print(f"[SAVE] {OUT_PDF}")
print("[DONE]")