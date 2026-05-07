from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT
IMAGE_ROOT = PROJECT_ROOT / "data" / "FISH_spots" / "2d" / "image"

SPLIT_PATH = ROOT / r"dataset\splits\rule_based_test_refined.csv"
OUT_DIR = ROOT / r"results\predictions\standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PEAKS_PER_IMAGE = {
    "MERFISH": 1800,
    "RCA": 1800,
    "SeqFISH": 1800,
}

MIN_DISTANCE = {
    "MERFISH": 3,
    "RCA": 3,
    "SeqFISH": 3,
}

PERCENTILE_THRESHOLD = {
    "MERFISH": 97.5,
    "RCA": 97.5,
    "SeqFISH": 97.5,
}


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    img = np.nan_to_num(img)
    return img


def robust_normalize(img):
    p1 = np.percentile(img, 1)
    p999 = np.percentile(img, 99.9)

    if p999 <= p1:
        return np.zeros_like(img, dtype=np.float32)

    out = (img - p1) / (p999 - p1)
    out = np.clip(out, 0, 1)
    return out.astype(np.float32)


def local_peak_detection(score_map, subset):
    min_dist = MIN_DISTANCE.get(subset, 3)
    k = 2 * min_dist + 1

    kernel = np.ones((k, k), np.uint8)
    local_max = score_map == cv2.dilate(score_map, kernel)

    perc = PERCENTILE_THRESHOLD.get(subset, 97.5)
    thr = np.percentile(score_map, perc)

    mask = local_max & (score_map >= thr)

    ys, xs = np.where(mask)
    scores = score_map[ys, xs]

    if len(scores) == 0:
        return pd.DataFrame(columns=["x", "y", "score"])

    order = np.argsort(-scores)
    xs = xs[order]
    ys = ys[order]
    scores = scores[order]

    max_peaks = MAX_PEAKS_PER_IMAGE.get(subset, 1800)

    xs = xs[:max_peaks]
    ys = ys[:max_peaks]
    scores = scores[:max_peaks]

    return pd.DataFrame({
        "x": xs.astype(float),
        "y": ys.astype(float),
        "score": scores.astype(float),
    })


def make_tophat_map(img_norm):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(img_norm, cv2.MORPH_OPEN, kernel)

    tophat = img_norm - opened
    tophat = np.clip(tophat, 0, None)
    tophat = robust_normalize(tophat)

    return tophat


def get_image_info(split_df):
    df = split_df.copy()

    if "stem" in df.columns:
        df["image_id"] = df["stem"].astype(str)
    elif "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str).apply(lambda x: Path(x).stem)
    elif "image_name" in df.columns:
        df["image_id"] = df["image_name"].astype(str).apply(lambda x: Path(x).stem)
    elif "image_path" in df.columns:
        df["image_id"] = df["image_path"].astype(str).apply(lambda x: Path(x).stem)
    else:
        raise ValueError(f"Cannot detect image id column. Columns: {list(df.columns)}")

    if "image_name" in df.columns:
        df["image_name"] = df["image_name"].astype(str)
    else:
        df["image_name"] = df["image_id"] + ".tif"

    if "subset" not in df.columns:
        df["subset"] = df["image_id"].astype(str).str.split("_").str[0]

    df["image_path_win"] = df["image_name"].apply(
        lambda x: IMAGE_ROOT / Path(str(x)).name
    )

    return df[["image_id", "image_name", "subset", "image_path_win"]].drop_duplicates()


def generate_method(image_info, method):
    outputs = []

    print("\n" + "=" * 80)
    print(f"[GENERATE] {method}")

    for idx, row in image_info.iterrows():
        image_id = row["image_id"]
        image_name = row["image_name"]
        subset = row["subset"]
        image_path = Path(row["image_path_win"])

        if not image_path.exists():
            print(f"[MISSING IMAGE] {image_path}")
            continue

        img = read_image(image_path)
        img_norm = robust_normalize(img)

        if method == "intensity_peak":
            score_map = img_norm
        elif method == "tophat_peak":
            score_map = make_tophat_map(img_norm)
        else:
            raise ValueError(method)

        peaks = local_peak_detection(score_map, subset)

        if len(peaks) == 0:
            print(f"[NO PEAK] {image_id}")
            continue

        peaks.insert(0, "method", method)
        peaks.insert(0, "subset", subset)
        peaks.insert(0, "image_name", image_name)
        peaks.insert(0, "image_id", image_id)

        outputs.append(peaks)

        if (idx + 1) % 20 == 0:
            print(f"processed {idx + 1}/{len(image_info)}")

    if len(outputs) == 0:
        out = pd.DataFrame(columns=["image_id", "image_name", "subset", "method", "x", "y", "score"])
    else:
        out = pd.concat(outputs, ignore_index=True)

    out = out.sort_values(["subset", "image_id", "score"], ascending=[True, True, False])

    out_path = OUT_DIR / f"{method}.csv"
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

    summary_path = OUT_DIR / f"{method}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {out_path}")
    print(f"[SAVED] {summary_path}")
    print(summary.to_string(index=False))

    return out


def rebuild_all_predictions():
    method_files = [
        "logdog.csv",
        "rule_candidate.csv",
        "unet_heatmap.csv",
        "intensity_peak.csv",
        "tophat_peak.csv",
        "unet_heatmap_pos.csv",
        "deepblink_style.csv",
        "yolo_pointbox.csv",
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

    print("\n" + "=" * 80)
    print("[ALL PREDICTIONS UPDATED]")
    print(f"saved: {all_path}")
    print(f"saved: {summary_path}")
    print(summary.to_string(index=False))


def main():
    print("[INFO] Generate intensity peak and top-hat peak baselines")
    print(f"ROOT: {ROOT}")
    print(f"IMAGE_ROOT: {IMAGE_ROOT}")
    print(f"SPLIT_PATH: {SPLIT_PATH}")
    print(f"OUT_DIR: {OUT_DIR}")

    split_df = pd.read_csv(SPLIT_PATH)
    image_info = get_image_info(split_df)

    print("\n[IMAGE COUNT BY SUBSET]")
    print(image_info.groupby("subset")["image_id"].nunique().to_string())

    generate_method(image_info, "intensity_peak")
    generate_method(image_info, "tophat_peak")

    rebuild_all_predictions()


if __name__ == "__main__":
    main()