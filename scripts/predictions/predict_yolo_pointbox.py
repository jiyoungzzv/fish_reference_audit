from pathlib import Path

import pandas as pd
from ultralytics import YOLO


PROJECT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "results" / "models" / "yolo_pointbox" / "train" / "weights" / "best.pt"

TEST_META = PROJECT_ROOT / "dataset" / "yolo_pointbox" / "test_metadata.csv"
TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "yolo_pointbox" / "images" / "test"

OUT_RAW = PROJECT_ROOT / "results" / "models" / "yolo_pointbox" / "predictions" / "yolo_pointbox_predictions_raw.csv"

STD_DIR = PROJECT_ROOT / "results" / "predictions" / "standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)

OUT_STD = STD_DIR / "yolo_pointbox.csv"
OUT_SUMMARY = STD_DIR / "yolo_pointbox_summary.csv"

METHOD_NAME = "yolo_pointbox"


def rebuild_all_predictions():
    method_files = [
        "intensity_peak.csv",
        "logdog.csv",
        "tophat_peak.csv",
        "unet_heatmap.csv",
        "unet_heatmap_pos.csv",
        "yolo_pointbox.csv",
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
    print(summary.to_string(index=False))


def main():
    print("[INFO] Predict YOLO point-box")
    print(f"MODEL_PATH: {MODEL_PATH}")

    meta = pd.read_csv(TEST_META)
    stem_to_info = {
        row["stem"]: {
            "image_name": row["image_name"],
            "subset": row["subset"],
        }
        for _, row in meta.iterrows()
    }

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(TEST_IMAGE_DIR),
        imgsz=512,
        conf=0.001,
        iou=0.5,
        max_det=3000,
        device=0,
        save=False,
        verbose=True,
    )

    rows = []

    for r in results:
        path = Path(r.path)
        stem = path.stem

        info = stem_to_info.get(stem)
        if info is None:
            continue

        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()

        for box, score in zip(xyxy, conf):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            rows.append({
                "image_id": stem,
                "image_name": info["image_name"],
                "subset": info["subset"],
                "method": METHOD_NAME,
                "x": float(cx),
                "y": float(cy),
                "score": float(score),
            })

    pred = pd.DataFrame(rows)

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")
    pred.to_csv(OUT_STD, index=False, encoding="utf-8-sig")

    summary = (
        pred.groupby("subset")
        .agg(
            n_images=("image_id", "nunique"),
            n_predictions=("x", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n[SAVED]")
    print(OUT_RAW)
    print(OUT_STD)
    print(OUT_SUMMARY)
    print(summary.to_string(index=False))

    rebuild_all_predictions()


if __name__ == "__main__":
    main()