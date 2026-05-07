import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT

INPUT_CSV = os.path.join(
    ROOT,
    "results/evaluation/missingness_all_methods/missingness_all_methods_f1_curves_master.csv"
)

OUT_DIR = PROJECT_ROOT / "figures"
OUT_COMBINED_PNG = os.path.join(OUT_DIR, "fig2_reference_sensitivity_combined.png")
OUT_COMBINED_PDF = os.path.join(OUT_DIR, "fig2_reference_sensitivity_combined.pdf")

os.makedirs(OUT_DIR, exist_ok=True)

OUT_SUMMARY = os.path.join(OUT_DIR, "fig2_fig3_reference_sensitivity_summary.csv")

OUT_FIG2_PNG = os.path.join(OUT_DIR, "fig2.png")
OUT_FIG2_PDF = os.path.join(OUT_DIR, "fig2.pdf")

# Main heatmap: subset-first
# MER-Raw, MER-Str, MER-Len, RCA-Raw, ...
OUT_FIG3_SUBSET_FIRST_PNG = os.path.join(OUT_DIR, "fig3_subset_first.png")
OUT_FIG3_SUBSET_FIRST_PDF = os.path.join(OUT_DIR, "fig3_subset_first.pdf")

# Additional heatmap: reference-first
# MER-Raw, RCA-Raw, SEQ-Raw, MER-Str, ...
OUT_FIG3_REFERENCE_FIRST_PNG = os.path.join(OUT_DIR, "fig3_reference_first.png")
OUT_FIG3_REFERENCE_FIRST_PDF = os.path.join(OUT_DIR, "fig3_reference_first.pdf")

# 1. Load

df = pd.read_csv(INPUT_CSV)

print("[LOAD]", INPUT_CSV)
print("[COLUMNS]", df.columns.tolist())
print(df.head())

# 2. Column setup

method_col = "method"
method_label_col = "method_label"
ref_col = "base_reference"
subset_col = "subset"
f1_col = "global_f1_mean"
level_col = "completeness"

required_cols = [
    method_col,
    method_label_col,
    ref_col,
    subset_col,
    f1_col,
    level_col,
]

for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# 3. Style

# Reference order: Raw -> Lenient -> Strict
ref_order = ["raw", "lenient", "strict"]

ref_label = {
    "raw": "Raw",
    "lenient": "Lenient",
    "strict": "Strict"
}


PALETTE = {
    "raw": "#A9B8D5",
    "lenient": "#ECC6DF",
    "strict": "#B2D2B6",
}

EDGE = "#8E8E8E"
GRID = "#C7C7C7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 4. Keep full-reference condition

max_level = df[level_col].max()
df_full = df[df[level_col] == max_level].copy()

print("[INFO] Using full-reference completeness:", max_level)
print("[INFO] n rows before filtering:", len(df_full))

df_full[ref_col] = df_full[ref_col].astype(str).str.lower()
df_full = df_full[df_full[ref_col].isin(ref_order)].copy()

print("[INFO] n rows after reference filtering:", len(df_full))

# 5. Summarize

label_map_df = (
    df_full[[method_col, method_label_col]]
    .dropna()
    .drop_duplicates(subset=[method_col])
)

method_label_map = dict(
    zip(label_map_df[method_col], label_map_df[method_label_col])
)

summary = (
    df_full
    .groupby([subset_col, ref_col, method_col], as_index=False)[f1_col]
    .mean()
)

summary[method_label_col] = summary[method_col].map(method_label_map)

summary.to_csv(OUT_SUMMARY, index=False)
print("[SAVE]", OUT_SUMMARY)

print("\n[SUMMARY HEAD]")
print(summary.head())

# 6. Split overall / subset

overall_summary = summary[summary[subset_col] == "ALL"].copy()
subset_summary = summary[summary[subset_col] != "ALL"].copy()

if overall_summary.empty:
    print("[WARN] subset == ALL not found. Using average over subsets for overall figure.")
    overall_summary = (
        subset_summary
        .groupby([ref_col, method_col], as_index=False)[f1_col]
        .mean()
    )
    overall_summary[subset_col] = "ALL"
    overall_summary[method_label_col] = overall_summary[method_col].map(method_label_map)

subset_order = ["MERFISH", "RCA", "SeqFISH"]
subset_order = [s for s in subset_order if s in subset_summary[subset_col].unique()]

# 7. Unified method order

method_order_df = (
    overall_summary
    .groupby(method_col, as_index=False)[f1_col]
    .mean()
    .sort_values(f1_col, ascending=False)
)

method_order = method_order_df[method_col].drop_duplicates().tolist()

method_label_map = (
    summary[[method_col, method_label_col]]
    .dropna()
    .drop_duplicates(subset=[method_col])
    .set_index(method_col)[method_label_col]
    .to_dict()
)

# Short labels for compact paper figures

METHOD_SHORT = {
    "unet_heatmap": "U-Net",
    "unet_heatmap_pos": "U-Net+",
    "rule_candidate": "Rule",
    "yolo_pointbox": "YOLO",
    "logdog": "LoG/DoG",
    "tophat_peak": "Top-hat",
    "intensity_peak": "Intensity",
    "deepblink_official": "DeepBlink",
}

SUBSET_SHORT = {
    "MERFISH": "MER",
    "RCA": "RCA",
    "SeqFISH": "SEQ",
}

REF_SHORT = {
    "raw": "Raw",
    "lenient": "Len",
    "strict": "Str",
}

method_labels = [
    METHOD_SHORT.get(m, method_label_map.get(m, m))
    for m in method_order
]

print("[METHOD ORDER]", method_order)
print("[METHOD LABELS]", method_labels)
print("[SUBSET ORDER]", subset_order)

# 8. Fig. 2 — Overall F1 grouped bar plot

panel_a = overall_summary.copy()
panel_a[ref_col] = pd.Categorical(panel_a[ref_col], categories=ref_order, ordered=True)
panel_a[method_col] = pd.Categorical(panel_a[method_col], categories=method_order, ordered=True)
panel_a = panel_a.sort_values([method_col, ref_col])

fig, ax = plt.subplots(figsize=(10.0, 5.8))

x = np.arange(len(method_order))
width = 0.24

offsets = {
    "raw": -width,
    "lenient": 0,
    "strict": width,
}

for r in ref_order:
    tmp = panel_a[panel_a[ref_col] == r]

    vals = []
    for m in method_order:
        v = tmp[tmp[method_col] == m][f1_col]
        vals.append(v.iloc[0] if len(v) > 0 else np.nan)

    ax.bar(
        x + offsets[r],
        vals,
        width=width,
        label=ref_label[r],
        color=PALETTE[r],
        edgecolor=EDGE,
        linewidth=0.8
    )

ax.set_ylabel("F1", fontsize=20)
ax.set_xlabel("Detector", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(method_labels, rotation=28, ha="right", fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55, color=GRID)

ax.legend(
    title="",
    frameon=True,
    ncol=1,
    loc="upper right",
    fontsize=18,
    title_fontsize=18
)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_FIG2_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_FIG2_PDF, bbox_inches="tight")
print("[SAVE]", OUT_FIG2_PNG)
print("[SAVE]", OUT_FIG2_PDF)
plt.close(fig)

# 9. Heatmap drawing function

def draw_heatmap(heat_pivot, out_png, out_pdf, block_mode="subset_first"):
    """
    block_mode:
        - subset_first:
            MER-Raw, MER-Str, MER-Len,
            RCA-Raw, RCA-Str, RCA-Len,
            SEQ-Raw, SEQ-Str, SEQ-Len

        - reference_first:
            MER-Raw, RCA-Raw, SEQ-Raw,
            MER-Str, RCA-Str, SEQ-Str,
            MER-Len, RCA-Len, SEQ-Len
    """

    heat_values = heat_pivot.values.astype(float)

    heat_ylabels = [
        METHOD_SHORT.get(m, method_label_map.get(m, m))
        for m in heat_pivot.index
    ]

    # Compact x-axis labels: MER-Raw, MER-Str, MER-Len ...

    heat_xlabels = []

    for col in heat_pivot.columns:
        if "-" in col:
            subset_name, ref_name = col.split("-", 1)

            subset_short = SUBSET_SHORT.get(subset_name, subset_name)

            ref_short_map = {
                "Raw": "Raw",
                "Lenient": "Len",
                "Strict": "Str"
            }
            ref_short = ref_short_map.get(ref_name, ref_name)

            heat_xlabels.append(f"{subset_short}-{ref_short}")
        else:
            heat_xlabels.append(col)

    # Draw heatmap

    fig, ax = plt.subplots(figsize=(9.8, 5.6))

    im = ax.imshow(
        heat_values,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="magma"
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_yticks(np.arange(len(heat_pivot.index)))
    ax.set_yticklabels(heat_ylabels, fontsize=15)

    ax.set_xticks(np.arange(len(heat_pivot.columns)))
    ax.set_xticklabels(
        heat_xlabels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=13
    )

    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=5)

    ax.set_xticks(np.arange(-0.5, len(heat_pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heat_pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.55, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # cell text
    for i in range(heat_values.shape[0]):
        for j in range(heat_values.shape[1]):
            val = heat_values[i, j]
            if not np.isnan(val):
                txt_color = "white" if val < 0.60 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight="bold",
                    color=txt_color
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("F1", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(bottom=0.20, left=0.18, right=0.96, top=0.98)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print("[SAVE]", out_png)
    print("[SAVE]", out_pdf)
    plt.close(fig)

# 10. Fig. 3A — Main version: subset-first
#      MER-Raw, MER-Str, MER-Len, RCA-Raw, ...

heat = subset_summary.copy()
heat["subset_ref"] = heat[subset_col].astype(str) + "-" + heat[ref_col].map(ref_label)

heat_col_order_subset_first = []

for s in subset_order:
    for r in ref_order:
        col = f"{s}-{ref_label[r]}"
        if col in heat["subset_ref"].unique():
            heat_col_order_subset_first.append(col)

heat_pivot_subset_first = heat.pivot_table(
    index=method_col,
    columns="subset_ref",
    values=f1_col,
    aggfunc="mean"
)

heat_pivot_subset_first = heat_pivot_subset_first.reindex(
    index=method_order,
    columns=heat_col_order_subset_first
)

# Combined Figure 2 — Panel A bar plot + Panel B heatmap
# Same font size and same y-axis height

def draw_combined_reference_sensitivity(
    panel_a,
    heat_pivot,
    out_png,
    out_pdf
):
    # Unified font sizes
    FONT_PANEL = 13
    FONT_LABEL = 9
    FONT_TICK = 8
    FONT_LEGEND = 8
    FONT_CELL = 6.5
    FONT_CBAR = 8

    # Figure canvas
    fig = plt.figure(figsize=(7.2, 3.15))

    # [left, bottom, width, height]
    ax_bar = fig.add_axes([0.075, 0.24, 0.365, 0.66])
    ax_heat = fig.add_axes([0.515, 0.24, 0.365, 0.66])
    cax = fig.add_axes([0.895, 0.24, 0.018, 0.66])

    # Panel A: Overall grouped bar plot

    x = np.arange(len(method_order))
    width = 0.24

    offsets = {
        "raw": -width,
        "lenient": 0,
        "strict": width,
    }

    for r in ref_order:
        tmp = panel_a[panel_a[ref_col] == r]

        vals = []
        for m in method_order:
            v = tmp[tmp[method_col] == m][f1_col]
            vals.append(v.iloc[0] if len(v) > 0 else np.nan)

        ax_bar.bar(
            x + offsets[r],
            vals,
            width=width,
            label=ref_label[r],
            color=PALETTE[r],
            edgecolor=EDGE,
            linewidth=0.6
        )

    ax_bar.set_ylabel("F1", fontsize=FONT_LABEL)
    ax_bar.set_xlabel("Detector", fontsize=FONT_LABEL)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        method_labels,
        rotation=32,
        ha="right",
        fontsize=FONT_TICK
    )
    ax_bar.tick_params(axis="y", labelsize=FONT_TICK)
    ax_bar.set_ylim(0, 1.0)

    ax_bar.grid(
        axis="y",
        linestyle="--",
        linewidth=0.45,
        alpha=0.55,
        color=GRID
    )

    ax_bar.legend(
        frameon=True,
        ncol=1,
        loc="upper right",
        fontsize=FONT_LEGEND,
        borderpad=0.25,
        handlelength=1.1,
        handletextpad=0.4,
        labelspacing=0.25
    )

    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    ax_bar.text(
        -0.16,
        1.04,
        "A",
        transform=ax_bar.transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

    # Panel B: Heatmap

    heat_values = heat_pivot.values.astype(float)

    heat_ylabels = [
        METHOD_SHORT.get(m, method_label_map.get(m, m))
        for m in heat_pivot.index
    ]

    heat_xlabels = []
    for col in heat_pivot.columns:
        if "-" in col:
            subset_name, ref_name = col.split("-", 1)
            subset_short = SUBSET_SHORT.get(subset_name, subset_name)

            ref_short_map = {
                "Raw": "Raw",
                "Lenient": "Len",
                "Strict": "Str"
            }
            ref_short = ref_short_map.get(ref_name, ref_name)

            heat_xlabels.append(f"{subset_short}-{ref_short}")
        else:
            heat_xlabels.append(col)

    im = ax_heat.imshow(
        heat_values,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="magma"
    )

    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")

    ax_heat.set_yticks(np.arange(len(heat_pivot.index)))
    ax_heat.set_yticklabels(heat_ylabels, fontsize=FONT_TICK)

    ax_heat.set_xticks(np.arange(len(heat_pivot.columns)))
    ax_heat.set_xticklabels(
        heat_xlabels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=FONT_TICK
    )

    ax_heat.tick_params(axis="x", pad=4)
    ax_heat.tick_params(axis="y", pad=3)

    # Cell boundary
    ax_heat.set_xticks(np.arange(-0.5, len(heat_pivot.columns), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(heat_pivot.index), 1), minor=True)
    ax_heat.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.4,
        alpha=0.25
    )
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    # Cell text
    for i in range(heat_values.shape[0]):
        for j in range(heat_values.shape[1]):
            val = heat_values[i, j]
            if not np.isnan(val):
                txt_color = "white" if val < 0.60 else "black"
                ax_heat.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=FONT_CELL,
                    fontweight="bold",
                    color=txt_color
                )

    for spine in ["top", "right", "left", "bottom"]:
        ax_heat.spines[spine].set_visible(False)

    ax_heat.text(
        -0.16,
        1.04,
        "B",
        transform=ax_heat.transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

    # Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("F1", fontsize=FONT_CBAR)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print("[SAVE]", out_png)
    print("[SAVE]", out_pdf)
    plt.close(fig)


draw_combined_reference_sensitivity(
    panel_a=panel_a,
    heat_pivot=heat_pivot_subset_first,
    out_png=OUT_COMBINED_PNG,
    out_pdf=OUT_COMBINED_PDF
)

draw_heatmap(
    heat_pivot_subset_first,
    OUT_FIG3_SUBSET_FIRST_PNG,
    OUT_FIG3_SUBSET_FIRST_PDF,
    block_mode="subset_first"
)

# 11. Fig. 3B — Additional version: reference-first
#      MER-Raw, RCA-Raw, SEQ-Raw, MER-Str, ...

heat_col_order_reference_first = []

for r in ref_order:
    for s in subset_order:
        col = f"{s}-{ref_label[r]}"
        if col in heat["subset_ref"].unique():
            heat_col_order_reference_first.append(col)

heat_pivot_reference_first = heat.pivot_table(
    index=method_col,
    columns="subset_ref",
    values=f1_col,
    aggfunc="mean"
)

heat_pivot_reference_first = heat_pivot_reference_first.reindex(
    index=method_order,
    columns=heat_col_order_reference_first
)

draw_heatmap(
    heat_pivot_reference_first,
    OUT_FIG3_REFERENCE_FIRST_PNG,
    OUT_FIG3_REFERENCE_FIRST_PDF,
    block_mode="reference_first"
)

print("[DONE] Fig. 2 and both Fig. 3 heatmaps generated.")