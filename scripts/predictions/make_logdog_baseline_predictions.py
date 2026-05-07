from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.feature import peak_local_max
from tqdm import tqdm


PROJECT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT
FISH_ROOT = PROJECT_ROOT / "data" / "FISH_spots"

IMAGE_DIR = FISH_ROOT / "2d" / "image"

SPLIT_PATH = PROJECT_ROOT / "dataset" / "splits" / "rule_based_test_refined.csv"

OUT_DIR = PROJECT_ROOT / "results" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "logdog_baseline_predictions.csv"
OUT_SUMMARY = OUT_DIR / "logdog_baseline_prediction_summary.csv"


SUBSET_RULES = {
    "MERFISH": {
        "min_snr": 3.5,
        "log_percentile": 97.0,
        "min_distance": 2,
        "max_candidates": 2500,
        "sigmas": [1.0, 1.5, 2.0],
    },
    "RCA": {
        "min_snr": 3.5,
        "log_percentile": 97.0,
        "min_distance": 2,
        "max_candidates": 2000,
        "sigmas": [1.0, 1.5, 2.0],
    },
    "SeqFISH": {
        "min_snr": 3.5,
        "log_percentile": 97.0,
        "min_distance": 2,
        "max_candidates": 2000,
        "sigmas": [1.0, 1.5, 2.0],
    },
}

DEFAULT_RULE = {
    "min_snr": 3.5,
    "log_percentile": 97.0,
    "min_distance": 2,
    "max_candidates": 2000,
    "sigmas": [1.0, 1.5, 2.0],
}


def read_image(image_name: str) -> np.ndarray:
    path = IMAGE_DIR / image_name
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    if path.suffix.lower() in [".tif", ".tiff"]:
        img = tifffile.imread(str(path))
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"failed to read image: {path}")

    if img.ndim == 3:
        img = img[..., 0]

    img = img.astype(np.float32)

    # robust normalization
    p1, p99 = np.percentile(img, [1, 99.5])
    if p99 <= p1:
        img_norm = np.zeros_like(img, dtype=np.float32)
    else:
        img_norm = (img - p1) / (p99 - p1)
        img_norm = np.clip(img_norm, 0, 1)

    return img_norm


def robust_snr(img: np.ndarray) -> np.ndarray:
    med = np.median(img)
    mad = np.median(np.abs(img - med)) + 1e-6
    sigma = 1.4826 * mad
    snr = (img - med) / (sigma + 1e-6)
    return snr.astype(np.float32)


def multiscale_log_response(img: np.ndarray, sigmas):
    responses = []
    for sigma in sigmas:
        resp = -gaussian_laplace(img, sigma=sigma)
        # scale normalization
        resp = (sigma ** 2) * resp
        responses.append(resp)

    stack = np.stack(responses, axis=0)
    best_resp = np.max(stack, axis=0)
    best_scale_idx = np.argmax(stack, axis=0)
    best_sigma = np.take(np.array(sigmas, dtype=np.float32), best_scale_idx)

    return best_resp.astype(np.float32), best_sigma.astype(np.float32)


def detect_candidates(img: np.ndarray, subset: str):
    rule = SUBSET_RULES.get(subset, DEFAULT_RULE)

    snr_map = robust_snr(img)
    log_resp, sigma_map = multiscale_log_response(img, rule["sigmas"])

    # response threshold
    positive_resp = log_resp[log_resp > 0]
    if len(positive_resp) == 0:
        return []

    resp_thr = np.percentile(positive_resp, rule["log_percentile"])

    # candidate mask
    mask = (log_resp >= resp_thr) & (snr_map >= rule["min_snr"])

    if mask.sum() == 0:
        return []

    # peak local max
    coordinates = peak_local_max(
        log_resp,
        min_distance=rule["min_distance"],
        threshold_abs=resp_thr,
        exclude_border=False,
    )

    rows = []
    h, w = img.shape[:2]

    for y, x in coordinates:
        if not mask[y, x]:
            continue

        rows.append({
            "pred_x": float(x),
            "pred_y": float(y),
            "score_raw": float(log_resp[y, x]),
            "snr_proxy": float(snr_map[y, x]),
            "sigma": float(sigma_map[y, x]),
            "distance_to_boundary": float(min(x, y, w - 1 - x, h - 1 - y)),
        })

    if len(rows) == 0:
        return []

    # sort by score and cap
    rows = sorted(rows, key=lambda r: r["score_raw"], reverse=True)
    rows = rows[: int(rule["max_candidates"])]

    return rows


def normalize_score(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    out = []
    for subset, sub in df.groupby("subset"):
        s = sub["score_raw"].astype(float)
        lo = s.quantile(0.01)
        hi = s.quantile(0.99)
        temp = sub.copy()
        if hi <= lo:
            temp["score"] = 0.5
        else:
            temp["score"] = ((s - lo) / (hi - lo)).clip(0, 1)
        out.append(temp)

    return pd.concat(out, ignore_index=True)


def main():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"split not found: {SPLIT_PATH}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"image dir not found: {IMAGE_DIR}")

    split_df = pd.read_csv(SPLIT_PATH)
    split_df = split_df.loc[:, ~split_df.columns.duplicated()].copy()

    print(f"[LOAD] split: {SPLIT_PATH}")
    print(f"test images: {len(split_df)}")
    print(split_df["subset"].value_counts())

    pred_rows = []

    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc="LoG/DoG"):
        image_name = row["image_name"]
        subset = row["subset"]
        stem = row.get("stem", Path(image_name).stem)

        img = read_image(image_name)
        candidates = detect_candidates(img, subset)

        for c in candidates:
            pred_rows.append({
                "image_name": image_name,
                "stem": stem,
                "subset": subset,
                "pred_x": c["pred_x"],
                "pred_y": c["pred_y"],
                "score_raw": c["score_raw"],
                "score": np.nan,
                "snr_proxy": c["snr_proxy"],
                "sigma": c["sigma"],
                "distance_to_boundary": c["distance_to_boundary"],
                "model_name": "logdog_baseline",
            })

    pred_df = pd.DataFrame(pred_rows)

    if len(pred_df) > 0:
        pred_df = normalize_score(pred_df)

    pred_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    summary = (
        pred_df.groupby("subset")
        .agg(
            n_predictions=("pred_x", "count"),
            mean_score=("score", "mean"),
            mean_snr=("snr_proxy", "mean"),
        )
        .reset_index()
        if len(pred_df) > 0 else pd.DataFrame()
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {OUT_PATH}")
    print(f"prediction rows: {len(pred_df)}")
    print(f"[SAVED] {OUT_SUMMARY}")

    print("\n===== subset prediction count =====")
    if len(pred_df) > 0:
        print(pred_df.groupby("subset").size())
    else:
        print("No predictions")

    print("\nDone.")


if __name__ == "__main__":
    main()