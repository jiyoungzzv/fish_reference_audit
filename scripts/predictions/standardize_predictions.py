from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT

INPUTS = {
    "logdog": ROOT / r"results\predictions\logdog_baseline_predictions.csv",
    "rule_candidate": ROOT / r"results\predictions\rule_candidate_baseline_predictions.csv",
    "unet_heatmap": ROOT / r"results\models\unet_heatmap_full_xyfixed\predictions\unet_heatmap_full_xyfixed_predictions.csv",
}

SPLIT_PATH = ROOT / r"dataset\splits\rule_based_test_refined.csv"

OUT_DIR = ROOT / r"results\predictions\standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_stem_from_path_or_name(series):
    return series.astype(str).apply(lambda x: Path(x).stem)


def detect_split_stems(split_df):
    candidate_cols = [
        "stem",
        "image_id",
        "image_name",
        "file_name",
        "filename",
        "image_path",
        "path",
    ]

    for col in candidate_cols:
        if col in split_df.columns:
            return make_stem_from_path_or_name(split_df[col])

    raise ValueError(
        f"Cannot detect image column in split file. Columns: {list(split_df.columns)}"
    )


def standardize_one(method_name, path, allowed_stems=None):
    print("\n" + "=" * 80)
    print(f"[{method_name}]")
    print(f"input: {path}")

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    print(f"original shape: {df.shape}")
    print(f"original columns: {list(df.columns)}")

    # image_id
    if "stem" in df.columns:
        image_id = df["stem"].astype(str)
    elif "image_id" in df.columns:
        image_id = make_stem_from_path_or_name(df["image_id"])
    elif "image_name" in df.columns:
        image_id = make_stem_from_path_or_name(df["image_name"])
    elif "image_path" in df.columns:
        image_id = make_stem_from_path_or_name(df["image_path"])
    else:
        raise ValueError(f"No image id column found in {path}")

    # image_name
    if "image_name" in df.columns:
        image_name = df["image_name"].astype(str)
    else:
        image_name = image_id.astype(str)

    # subset
    if "subset" in df.columns:
        subset = df["subset"].astype(str)
    else:
        subset = image_id.astype(str).str.split("_").str[0]

    # x, y
    if "pred_x" in df.columns:
        x = df["pred_x"]
    elif "x" in df.columns:
        x = df["x"]
    else:
        raise ValueError(f"No x column found in {path}")

    if "pred_y" in df.columns:
        y = df["pred_y"]
    elif "y" in df.columns:
        y = df["y"]
    else:
        raise ValueError(f"No y column found in {path}")

    # score
    if "score" in df.columns:
        score = df["score"]
    elif "score_raw" in df.columns:
        score = df["score_raw"]
    else:
        score = 1.0

    out = pd.DataFrame({
        "image_id": image_id,
        "image_name": image_name,
        "subset": subset,
        "method": method_name,
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
        "score": pd.to_numeric(score, errors="coerce"),
    })

    before_dropna = len(out)
    out = out.dropna(subset=["image_id", "subset", "x", "y", "score"]).copy()
    print(f"dropna removed: {before_dropna - len(out)}")

    if allowed_stems is not None:
        before_filter = len(out)
        out = out[out["image_id"].isin(allowed_stems)].copy()
        print(f"filtered by refined test split: {before_filter} -> {len(out)}")

    out = out.sort_values(["subset", "image_id", "score"], ascending=[True, True, False])

    out_path = OUT_DIR / f"{method_name}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

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

    summary_path = OUT_DIR / f"{method_name}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"saved: {out_path}")
    print(f"saved: {summary_path}")
    print(summary.to_string(index=False))

    return out


def main():
    print("[INFO] Standardizing prediction files")
    print(f"ROOT: {ROOT}")
    print(f"OUT_DIR: {OUT_DIR}")

    allowed_stems = None

    if SPLIT_PATH.exists():
        split_df = pd.read_csv(SPLIT_PATH)
        print("\n[INFO] refined test split loaded")
        print(f"split path: {SPLIT_PATH}")
        print(f"split shape: {split_df.shape}")
        print(f"split columns: {list(split_df.columns)}")

        split_stems = detect_split_stems(split_df)
        allowed_stems = set(split_stems.astype(str).tolist())
        print(f"n refined test images: {len(allowed_stems)}")
    else:
        print("\n[WARNING] refined test split not found. No split filtering will be applied.")

    all_outputs = []

    for method_name, path in INPUTS.items():
        out = standardize_one(method_name, path, allowed_stems=allowed_stems)
        all_outputs.append(out)

    all_df = pd.concat(all_outputs, ignore_index=True)
    all_path = OUT_DIR / "all_predictions.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

    all_summary = (
        all_df.groupby(["method", "subset"])
        .agg(
            n_images=("image_id", "nunique"),
            n_predictions=("x", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )

    all_summary_path = OUT_DIR / "all_predictions_summary.csv"
    all_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[DONE]")
    print(f"saved: {all_path}")
    print(f"saved: {all_summary_path}")
    print("\n[ALL SUMMARY]")
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    main()