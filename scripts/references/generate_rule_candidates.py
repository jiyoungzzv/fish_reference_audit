from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree


DATASET_ROOT = PROJECT_ROOT / "dataset"

TEST_STATS_PATH = DATASET_ROOT / "metadata" / "test_image_statistics_refined.csv"

OUT_CAND_DIR = DATASET_ROOT / "candidates"
OUT_CAND_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_CAND_DIR / "test_candidate_features_refined.csv"
OUT_SUMMARY_PATH = OUT_CAND_DIR / "test_candidate_summary_refined.csv"


# percentile normalization
P_LOW = 1.0
P_HIGH = 99.5

# background subtraction
BG_SIGMA = 8.0

# LoG scale
LOG_SIGMAS = [1.0, 1.5, 2.0]

# local maximum
MIN_DISTANCE = 2

# local window for SNR
LOCAL_RADIUS = 5

EPS = 1e-8


SUBSET_RULES = {
    "MERFISH": {
        "min_snr": 3.0,
        "log_percentile": 95,
        "max_candidates": 3000,
    },
    "RCA": {
        "min_snr": 3.0,
        "log_percentile": 96,
        "max_candidates": 2000,
    },
    "SeqFISH": {
        "min_snr": 3.0,
        "log_percentile": 95,
        "max_candidates": 3000,
    },
}

DEFAULT_RULE = {
    "min_snr": 3.0,
    "log_percentile": 95,
    "max_candidates": 3000,
}


def get_subset_rule(subset: str) -> dict:
    return SUBSET_RULES.get(subset, DEFAULT_RULE)


def load_image_as_float(image_path: Path) -> np.ndarray:
    img = tifffile.imread(str(image_path))

    # 2D grayscale
    if img.ndim == 2:
        arr = img.astype(np.float32)

    elif img.ndim == 3:
        if img.shape[-1] in [3, 4]:
            arr = img[..., 1].astype(np.float32)
        else:
            arr = img[0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return arr


def percentile_normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [P_LOW, P_HIGH])

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + EPS)

    return img.astype(np.float32)


def log_response_multi_scale(img: np.ndarray) -> np.ndarray:
    responses = []

    for sigma in LOG_SIGMAS:
        # LoG: sigma^2 * -Laplacian(Gaussian)
        gxx = gaussian_filter(img, sigma=sigma, order=(2, 0))
        gyy = gaussian_filter(img, sigma=sigma, order=(0, 2))
        log = -(gxx + gyy) * (sigma ** 2)
        responses.append(log)

    response = np.max(np.stack(responses, axis=0), axis=0)
    return response.astype(np.float32)


def get_local_stats(img: np.ndarray, y: int, x: int, r: int = LOCAL_RADIUS):
    h, w = img.shape

    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)
    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)

    patch = img[y1:y2, x1:x2].reshape(-1)

    local_median = float(np.median(patch))
    local_mad = float(np.median(np.abs(patch - local_median)))

    return local_median, local_mad


def detect_candidates(img_norm: np.ndarray, subset: str) -> pd.DataFrame:
    rule = get_subset_rule(subset)
    min_snr = rule["min_snr"]
    log_percentile = rule["log_percentile"]
    max_candidates = rule["max_candidates"]

    h, w = img_norm.shape

    # background subtraction
    bg = gaussian_filter(img_norm, sigma=BG_SIGMA)
    img_res = img_norm - bg
    img_res = np.clip(img_res, 0, None)

    # LoG response
    response = log_response_multi_scale(img_res)

    # local maximum mask
    size = 2 * MIN_DISTANCE + 1
    local_max = response == maximum_filter(response, size=size)

    ys, xs = np.where(local_max)

    if len(ys) == 0:
        return pd.DataFrame()

    local_responses = response[ys, xs]
    log_thr = np.percentile(local_responses, log_percentile)

    rows = []

    for y, x in zip(ys, xs):
        peak_intensity = float(img_res[y, x])
        log_resp = float(response[y, x])

        if log_resp < log_thr:
            continue

        local_median, local_mad = get_local_stats(img_res, int(y), int(x), LOCAL_RADIUS)

        snr_proxy = (peak_intensity - local_median) / (1.4826 * local_mad + EPS)

        if snr_proxy < min_snr:
            continue

        distance_to_boundary = float(min(x, y, w - 1 - x, h - 1 - y))

        rows.append({
            "x": float(x),
            "y": float(y),
            "peak_intensity": peak_intensity,
            "local_median": local_median,
            "local_mad": local_mad,
            "snr_proxy": float(snr_proxy),
            "log_response": log_resp,
            "distance_to_boundary": distance_to_boundary,
            "candidate_min_snr_rule": min_snr,
            "candidate_log_percentile_rule": log_percentile,
            "candidate_max_rule": max_candidates,
        })

    cand_df = pd.DataFrame(rows)

    if len(cand_df) == 0:
        return cand_df

    if len(cand_df) > max_candidates:
        cand_df = (
            cand_df
            .sort_values(["snr_proxy", "log_response"], ascending=False)
            .head(max_candidates)
            .reset_index(drop=True)
        )

    if len(cand_df) > 1:
        pts = cand_df[["x", "y"]].values
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        cand_df["nearest_neighbor_distance"] = dists[:, 1]
    elif len(cand_df) == 1:
        cand_df["nearest_neighbor_distance"] = np.nan

    return cand_df


def main():
    if not TEST_STATS_PATH.exists():
        raise FileNotFoundError(f"test statistics File not found: {TEST_STATS_PATH}")

    test_df = pd.read_csv(TEST_STATS_PATH)

    all_candidates = []
    summary_rows = []

    print(f"[1] test image statistics LOAD: {TEST_STATS_PATH}")
    print(f"Number of test images: {len(test_df)}")

    for idx, row in test_df.iterrows():
        image_path = Path(row["image_path"])
        subset = row["subset"]
        stem = row["stem"]
        image_name = row["image_name"]
        raw_spot_count = int(row["raw_spot_count"])

        rule = get_subset_rule(subset)

        try:
            img = load_image_as_float(image_path)
            img_norm = percentile_normalize(img)
            cand_df = detect_candidates(img_norm, subset)

            candidate_count = len(cand_df)

            if candidate_count > 0:
                cand_df.insert(0, "image_path", str(image_path))
                cand_df.insert(1, "image_name", image_name)
                cand_df.insert(2, "stem", stem)
                cand_df.insert(3, "subset", subset)
                cand_df.insert(4, "raw_spot_count", raw_spot_count)
                all_candidates.append(cand_df)

            summary_rows.append({
                "image_path": str(image_path),
                "image_name": image_name,
                "stem": stem,
                "subset": subset,
                "raw_spot_count": raw_spot_count,
                "candidate_count": candidate_count,
                "candidate_raw_ratio": candidate_count / max(raw_spot_count, 1),

                "mean_snr_proxy": cand_df["snr_proxy"].mean() if candidate_count > 0 else np.nan,
                "median_snr_proxy": cand_df["snr_proxy"].median() if candidate_count > 0 else np.nan,
                "max_snr_proxy": cand_df["snr_proxy"].max() if candidate_count > 0 else np.nan,

                "mean_log_response": cand_df["log_response"].mean() if candidate_count > 0 else np.nan,
                "median_log_response": cand_df["log_response"].median() if candidate_count > 0 else np.nan,
                "max_log_response": cand_df["log_response"].max() if candidate_count > 0 else np.nan,

                "median_nn_distance": cand_df["nearest_neighbor_distance"].median() if candidate_count > 0 else np.nan,
                "mean_nn_distance": cand_df["nearest_neighbor_distance"].mean() if candidate_count > 0 else np.nan,

                "candidate_min_snr_rule": rule["min_snr"],
                "candidate_log_percentile_rule": rule["log_percentile"],
                "candidate_max_rule": rule["max_candidates"],
            })

            if (idx + 1) % 20 == 0:
                print(f"[PROGRESS] {idx + 1}/{len(test_df)}")

        except Exception as e:
            print(f"[ERROR] {image_name}: {e}")

    if len(all_candidates) > 0:
        out_df = pd.concat(all_candidates, ignore_index=True)
    else:
        out_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] candidate features: {OUT_PATH}")
    print(f"[SAVED] candidate summary:  {OUT_SUMMARY_PATH}")

    print("Done.")
    print(
        summary_df
        .groupby("subset")[["raw_spot_count", "candidate_count", "candidate_raw_ratio"]]
        .describe()
    )

    print("Done.")
    print(
        summary_df
        .groupby("subset")[["candidate_min_snr_rule", "candidate_log_percentile_rule", "candidate_max_rule"]]
        .first()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()