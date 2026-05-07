from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT

INPUT_PATH = ROOT / r"results\models\unet_heatmap_pos\predictions\unet_heatmap_pos_predictions.csv"

OUT_DIR = ROOT / r"results\predictions\standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "unet_heatmap_pos.csv"
SUMMARY_PATH = OUT_DIR / "unet_heatmap_pos_summary.csv"

METHOD_NAME = "unet_heatmap_pos"


def rebuild_all_predictions():
    method_files = [
        "intensity_peak.csv",
        "logdog.csv",
        "tophat_peak.csv",
        "unet_heatmap.csv",
        "unet_heatmap_pos.csv",
        "deepblink_style.csv",
        "yolo_pointbox.csv",
        "rule_candidate.csv",
    ]

    dfs = []

    for fname in method_files:
        path = OUT_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"[LOAD] {fname}: {df.shape}")
        else:
            print(f"[SKIP] {fname}")

    all_df = pd.concat(dfs, ignore_index=True)

    all_path = OUT_DIR / "all_predictions.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    summary = (
        all_df.groupby(["method", "subset"])
        .agg(
            n_images=("image_id", "nunique"),
            n_predictions=("x", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )

    summary_path = OUT_DIR / "all_predictions_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[ALL PREDICTIONS UPDATED]")
    print(all_path)
    print(summary_path)
    print(summary.to_string(index=False))


def main():
    print("[INFO] Standardize U-Net positive sampling predictions")
    print(f"INPUT_PATH: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("original shape:", df.shape)
    print("original columns:", list(df.columns))

    out = pd.DataFrame({
        "image_id": df["stem"].astype(str),
        "image_name": df["image_name"].astype(str),
        "subset": df["subset"].astype(str),
        "method": METHOD_NAME,
        "x": pd.to_numeric(df["pred_x"], errors="coerce"),
        "y": pd.to_numeric(df["pred_y"], errors="coerce"),
        "score": pd.to_numeric(df["score"], errors="coerce"),
    })

    out = out.dropna(subset=["image_id", "subset", "x", "y", "score"]).copy()

    out = out.sort_values(
        ["subset", "image_id", "score"],
        ascending=[True, True, False],
    )

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    summary = (
        out.groupby("subset")
        .agg(
            n_images=("image_id", "nunique"),
            n_predictions=("x", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\n[SAVED]")
    print(OUT_PATH)
    print(SUMMARY_PATH)
    print(summary.to_string(index=False))

    rebuild_all_predictions()


if __name__ == "__main__":
    main()