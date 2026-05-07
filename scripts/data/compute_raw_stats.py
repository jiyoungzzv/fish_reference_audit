from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


import pandas as pd
import tifffile


DATASET_ROOT = PROJECT_ROOT / "dataset"

SPLIT_PATH = DATASET_ROOT / "splits" / "rule_based_test_refined.csv"
OUT_META_DIR = DATASET_ROOT / "metadata"

OUT_META_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_META_DIR / "test_image_statistics_refined.csv"
SUMMARY_PATH = OUT_META_DIR / "test_image_statistics_summary_refined.csv"


def read_spot_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    df = df.dropna(how="all").reset_index(drop=True)

    return df


def get_image_shape(image_path: Path):
    img = tifffile.imread(str(image_path))

    if img.ndim == 2:
        h, w = img.shape
        c = 1
    elif img.ndim == 3:
        if img.shape[-1] in [3, 4]:
            h, w, c = img.shape
        else:
            c, h, w = img.shape
    else:
        raise ValueError(f"Unsupported image ndim={img.ndim}, shape={img.shape}")

    return h, w, c, img.dtype, img.shape


def main():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"split File not found: {SPLIT_PATH}")

    split_df = pd.read_csv(SPLIT_PATH)

    rows = []

    print(f"[1] rule_based_test LOAD: {SPLIT_PATH}")
    print(f"Number of images: {len(split_df)}")

    for idx, row in split_df.iterrows():
        image_path = Path(row["image_path"])
        csv_path = Path(row["csv_path"])
        subset = row["subset"]
        stem = row["stem"]

        if not image_path.exists():
            print(f"[WARNING] image not found: {image_path}")
            continue

        if not csv_path.exists():
            print(f"[WARNING] CSV not found: {csv_path}")
            continue

        try:
            h, w, c, dtype, shape = get_image_shape(image_path)
            spot_df = read_spot_csv(csv_path)
            raw_spot_count = len(spot_df)

            area_px = h * w
            spot_density_per_10k_px = raw_spot_count / area_px * 10000

            rows.append({
                "image_path": str(image_path),
                "csv_path": str(csv_path),
                "image_name": image_path.name,
                "csv_name": csv_path.name,
                "stem": stem,
                "subset": subset,
                "height": h,
                "width": w,
                "channels": c,
                "dtype": str(dtype),
                "shape": str(shape),
                "area_px": area_px,
                "raw_spot_count": raw_spot_count,
                "spot_density_per_10k_px": spot_density_per_10k_px,
            })

        except Exception as e:
            print(f"[ERROR] {stem}: {e}")

    stat_df = pd.DataFrame(rows)

    if len(stat_df) == 0:
        raise RuntimeError("Input validation failed. Please check the required files and paths.")

    stat_df["difficulty_by_count"] = "medium"

    for subset, sub_df in stat_df.groupby("subset"):
        q33 = sub_df["raw_spot_count"].quantile(0.33)
        q67 = sub_df["raw_spot_count"].quantile(0.67)

        mask_subset = stat_df["subset"] == subset
        stat_df.loc[mask_subset & (stat_df["raw_spot_count"] <= q33), "difficulty_by_count"] = "easy"
        stat_df.loc[mask_subset & (stat_df["raw_spot_count"] >= q67), "difficulty_by_count"] = "hard"

    stat_df = stat_df.sort_values(["subset", "raw_spot_count"]).reset_index(drop=True)
    stat_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {OUT_PATH}")

    summary = (
        stat_df
        .groupby(["subset", "difficulty_by_count"])
        .agg(
            n_images=("image_name", "count"),
            mean_spots=("raw_spot_count", "mean"),
            median_spots=("raw_spot_count", "median"),
            min_spots=("raw_spot_count", "min"),
            max_spots=("raw_spot_count", "max"),
            mean_density=("spot_density_per_10k_px", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"[summary SAVED] {SUMMARY_PATH}")

    print("Done.")
    print(
        stat_df
        .groupby("subset")["raw_spot_count"]
        .describe()
    )

    print("\n===== difficulty distribution =====")
    print(
        stat_df
        .groupby(["subset", "difficulty_by_count"])
        .size()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()