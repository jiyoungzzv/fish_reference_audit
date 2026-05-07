from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT

PRED_DIR = ROOT / r"results\models\deepblink_official\smfish1_gray"
META_PATH = ROOT / r"dataset\yolo_pointbox\test_metadata.csv"
SPLIT_PATH = ROOT / r"dataset\splits\rule_based_test_refined.csv"

STD_DIR = ROOT / r"results\predictions\standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = STD_DIR / "deepblink_official.csv"
SUMMARY_PATH = STD_DIR / "deepblink_official_summary.csv"

METHOD_NAME = "deepblink_official"


def load_test_meta():
    if META_PATH.exists():
        meta = pd.read_csv(META_PATH)
        if "stem" not in meta.columns:
            raise ValueError(f"test_metadata.csv has no stem column: {list(meta.columns)}")
        return meta

    df = pd.read_csv(SPLIT_PATH).copy()
    if "stem" not in df.columns:
        df["stem"] = df["image_name"].astype(str).apply(lambda x: Path(x).stem)
    return df


def find_col(df, candidates):
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


def read_deepblink_csv(path):
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["x", "y", "score"])

    if len(df) == 0:
        return pd.DataFrame(columns=["x", "y", "score"])

    print(f"\n[READ] {path.name}")
    print("columns:", list(df.columns))
    print(df.head(3).to_string(index=False))

    x_col = find_col(df, ["x", "pred_x", "axis-1", "column", "col"])
    y_col = find_col(df, ["y", "pred_y", "axis-0", "row"])

    score_col = find_col(df, ["p", "probability", "prob", "score", "confidence", "conf"])

    if x_col is None or y_col is None:
        numeric_cols = []
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                numeric_cols.append(c)

        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        else:
            raise ValueError(f"Cannot detect x/y columns in {path}: {list(df.columns)}")

    out = pd.DataFrame({
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": pd.to_numeric(df[y_col], errors="coerce"),
    })

    if score_col is not None:
        out["score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        out["score"] = 1.0

    out = out.dropna(subset=["x", "y", "score"]).copy()

    return out


def rebuild_all_predictions():
    method_files = [
        "intensity_peak.csv",
        "logdog.csv",
        "tophat_peak.csv",
        "unet_heatmap.csv",
        "unet_heatmap_pos.csv",
        "yolo_pointbox.csv",
        "deepblink_official.csv",
        "rule_candidate.csv",
    ]

    dfs = []

    for fname in method_files:
        path = STD_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"[LOAD] {fname}: {df.shape}")
        else:
            print(f"[SKIP] {fname}")

    all_df = pd.concat(dfs, ignore_index=True)

    all_path = STD_DIR / "all_predictions.csv"
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

    summary_path = STD_DIR / "all_predictions_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[ALL PREDICTIONS UPDATED]")
    print(all_path)
    print(summary_path)
    print(summary.to_string(index=False))


def main():
    print("[INFO] Standardize official DeepBlink smfish1 predictions")
    print(f"PRED_DIR: {PRED_DIR}")

    meta = load_test_meta()
    meta = meta.copy()

    if "stem" not in meta.columns:
        meta["stem"] = meta["image_name"].astype(str).apply(lambda x: Path(x).stem)
    if "image_name" not in meta.columns:
        meta["image_name"] = meta["stem"].astype(str) + ".tif"
    if "subset" not in meta.columns:
        meta["subset"] = meta["stem"].astype(str).str.split("_").str[0]

    stem_to_info = {
        str(row["stem"]): {
            "image_name": row["image_name"],
            "subset": row["subset"],
        }
        for _, row in meta.iterrows()
    }

    csv_files = sorted(PRED_DIR.glob("*.csv"))
    print(f"n csv files: {len(csv_files)}")

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {PRED_DIR}")

    outputs = []
    missing_meta = []

    for csv_path in csv_files:
        stem = csv_path.stem

        if stem not in stem_to_info:
            missing_meta.append(stem)
            continue

        pred = read_deepblink_csv(csv_path)

        if len(pred) == 0:
            continue

        info = stem_to_info[stem]

        pred.insert(0, "method", METHOD_NAME)
        pred.insert(0, "subset", info["subset"])
        pred.insert(0, "image_name", info["image_name"])
        pred.insert(0, "image_id", stem)

        outputs.append(pred)

    if len(missing_meta) > 0:
        print("\n[WARNING] CSV files not found in metadata:")
        for s in missing_meta[:20]:
            print(" -", s)
        if len(missing_meta) > 20:
            print(f"... {len(missing_meta) - 20} more")

    if len(outputs) == 0:
        out = pd.DataFrame(columns=["image_id", "image_name", "subset", "method", "x", "y", "score"])
    else:
        out = pd.concat(outputs, ignore_index=True)

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