from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DATASET_ROOT = PROJECT_ROOT / "dataset"

TEST_STATS_PATH = DATASET_ROOT / "metadata" / "test_image_statistics_refined.csv"
CAND_PATH = DATASET_ROOT / "candidates" / "test_candidate_features_refined.csv"

OUT_ANN_DIR = DATASET_ROOT / "annotations"
OUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATCH_PATH = OUT_ANN_DIR / "raw_candidate_matching_refined.csv"
OUT_SUMMARY_PATH = OUT_ANN_DIR / "raw_candidate_matching_summary_refined.csv"


MATCH_RADIUS = 3.0


def read_raw_points(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(columns=["x", "y"])

    cols_lower = {str(c).lower(): c for c in df.columns}

    x_col = None
    y_col = None

    for cand in ["x", "pos_x", "x_coord", "x_coordinate", "col", "column"]:
        if cand in cols_lower:
            x_col = cols_lower[cand]
            break

    for cand in ["y", "pos_y", "y_coord", "y_coordinate", "row"]:
        if cand in cols_lower:
            y_col = cols_lower[cand]
            break

    if "axis-0" in cols_lower and "axis-1" in cols_lower:
        y_col = cols_lower["axis-0"]
        x_col = cols_lower["axis-1"]

    if x_col is None or y_col is None:
        if df.shape[1] < 2:
            raise ValueError(f"CSV must contain at least two coordinate columns: {csv_path}")

        y_col = df.columns[0]
        x_col = df.columns[1]

    out = df[[x_col, y_col]].copy()
    out.columns = ["x", "y"]

    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["x", "y"]).reset_index(drop=True)

    return out


def greedy_match(raw_pts: np.ndarray, cand_pts: np.ndarray, radius: float):
    n_raw = len(raw_pts)
    n_cand = len(cand_pts)

    if n_raw == 0 or n_cand == 0:
        return [], set(), set()

    tree = cKDTree(cand_pts)
    candidate_pairs = []

    for raw_idx, p in enumerate(raw_pts):
        cand_indices = tree.query_ball_point(p, r=radius)

        for cand_idx in cand_indices:
            dist = float(np.linalg.norm(raw_pts[raw_idx] - cand_pts[cand_idx]))
            candidate_pairs.append((dist, raw_idx, cand_idx))

    candidate_pairs.sort(key=lambda x: x[0])

    matched = []
    used_raw = set()
    used_cand = set()

    for dist, raw_idx, cand_idx in candidate_pairs:
        if raw_idx in used_raw or cand_idx in used_cand:
            continue

        matched.append({
            "raw_index": raw_idx,
            "candidate_index": cand_idx,
            "distance": dist,
        })
        used_raw.add(raw_idx)
        used_cand.add(cand_idx)

    raw_only = set(range(n_raw)) - used_raw
    cand_only = set(range(n_cand)) - used_cand

    return matched, raw_only, cand_only


def main():
    if not TEST_STATS_PATH.exists():
        raise FileNotFoundError(f"test statistics File not found: {TEST_STATS_PATH}")

    if not CAND_PATH.exists():
        raise FileNotFoundError(f"candidate features File not found: {CAND_PATH}")

    test_df = pd.read_csv(TEST_STATS_PATH)
    cand_df_all = pd.read_csv(CAND_PATH)

    match_rows = []
    summary_rows = []

    print(f"[1] test stats LOAD: {TEST_STATS_PATH}")
    print(f"[2] candidates LOAD: {CAND_PATH}")
    print(f"test images: {len(test_df)}")
    print(f"candidate rows: {len(cand_df_all)}")
    print(f"matching radius: {MATCH_RADIUS}px")

    for idx, row in test_df.iterrows():
        image_name = row["image_name"]
        stem = row["stem"]
        subset = row["subset"]
        csv_path = Path(row["csv_path"])

        raw_df = read_raw_points(csv_path)

        cand_df = cand_df_all[cand_df_all["image_name"] == image_name].copy().reset_index(drop=True)

        raw_pts = raw_df[["x", "y"]].values.astype(float)
        cand_pts = cand_df[["x", "y"]].values.astype(float) if len(cand_df) > 0 else np.empty((0, 2))

        matched, raw_only, cand_only = greedy_match(raw_pts, cand_pts, MATCH_RADIUS)

        # matched rows
        for m in matched:
            ri = m["raw_index"]
            ci = m["candidate_index"]

            raw_x, raw_y = raw_pts[ri]
            cand_x, cand_y = cand_pts[ci]

            cand_info = cand_df.iloc[ci].to_dict()

            match_rows.append({
                "image_name": image_name,
                "stem": stem,
                "subset": subset,
                "status": "matched",
                "raw_index": ri,
                "candidate_index": ci,
                "raw_x": raw_x,
                "raw_y": raw_y,
                "candidate_x": cand_x,
                "candidate_y": cand_y,
                "distance": m["distance"],
                "snr_proxy": cand_info.get("snr_proxy", np.nan),
                "log_response": cand_info.get("log_response", np.nan),
                "nearest_neighbor_distance": cand_info.get("nearest_neighbor_distance", np.nan),
                "distance_to_boundary": cand_info.get("distance_to_boundary", np.nan),
            })

        # raw only rows
        for ri in sorted(raw_only):
            raw_x, raw_y = raw_pts[ri]

            match_rows.append({
                "image_name": image_name,
                "stem": stem,
                "subset": subset,
                "status": "raw_only",
                "raw_index": ri,
                "candidate_index": np.nan,
                "raw_x": raw_x,
                "raw_y": raw_y,
                "candidate_x": np.nan,
                "candidate_y": np.nan,
                "distance": np.nan,
                "snr_proxy": np.nan,
                "log_response": np.nan,
                "nearest_neighbor_distance": np.nan,
                "distance_to_boundary": np.nan,
            })

        # candidate only rows
        for ci in sorted(cand_only):
            cand_x, cand_y = cand_pts[ci]
            cand_info = cand_df.iloc[ci].to_dict()

            match_rows.append({
                "image_name": image_name,
                "stem": stem,
                "subset": subset,
                "status": "candidate_only",
                "raw_index": np.nan,
                "candidate_index": ci,
                "raw_x": np.nan,
                "raw_y": np.nan,
                "candidate_x": cand_x,
                "candidate_y": cand_y,
                "distance": np.nan,
                "snr_proxy": cand_info.get("snr_proxy", np.nan),
                "log_response": cand_info.get("log_response", np.nan),
                "nearest_neighbor_distance": cand_info.get("nearest_neighbor_distance", np.nan),
                "distance_to_boundary": cand_info.get("distance_to_boundary", np.nan),
            })

        raw_count = len(raw_df)
        candidate_count = len(cand_df)
        matched_count = len(matched)
        raw_only_count = len(raw_only)
        candidate_only_count = len(cand_only)

        summary_rows.append({
            "image_name": image_name,
            "stem": stem,
            "subset": subset,
            "raw_count": raw_count,
            "candidate_count": candidate_count,
            "matched_count": matched_count,
            "raw_only_count": raw_only_count,
            "candidate_only_count": candidate_only_count,
            "raw_recall_by_candidate": matched_count / raw_count if raw_count > 0 else np.nan,
            "candidate_precision_against_raw": matched_count / candidate_count if candidate_count > 0 else np.nan,
            "candidate_raw_ratio": candidate_count / max(raw_count, 1),
            "match_radius": MATCH_RADIUS,
        })

        if (idx + 1) % 20 == 0:
            print(f"[PROGRESS] {idx + 1}/{len(test_df)}")

    match_df = pd.DataFrame(match_rows)
    summary_df = pd.DataFrame(summary_rows)

    match_df.to_csv(OUT_MATCH_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {OUT_MATCH_PATH}")
    print(f"[SAVED] {OUT_SUMMARY_PATH}")

    print("Done.")
    print(
        summary_df
        .groupby("subset")[
            [
                "raw_count",
                "candidate_count",
                "matched_count",
                "raw_only_count",
                "candidate_only_count",
                "raw_recall_by_candidate",
                "candidate_precision_against_raw",
            ]
        ]
        .mean()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()