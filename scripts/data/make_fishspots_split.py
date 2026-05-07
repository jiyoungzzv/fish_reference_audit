from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


import pandas as pd
import tifffile


FISH_ROOT = Path(os.environ.get("FISH_SPOTS_ROOT", PROJECT_ROOT / "data" / "FISH_spots"))

IMAGE_DIR = FISH_ROOT / "2d" / "image"
CSV_DIR = FISH_ROOT / "2d" / "csv"

DATASET_ROOT = PROJECT_ROOT / "dataset"

OUT_SPLIT_DIR = DATASET_ROOT / "splits"
OUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42

MAIN_SUBSETS = ["MERFISH", "RCA", "SeqFISH"]

TEST_N_PER_SUBSET = 50
VAL_RATIO = 0.10

IMAGE_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def infer_subset_from_stem(stem: str) -> str:
    s = stem.lower()

    if s.startswith("merfish"):
        return "MERFISH"
    if s.startswith("rca"):
        return "RCA"
    if s.startswith("seqfish"):
        return "SeqFISH"
    if s.startswith("deepblink"):
        return "deepblink"
    if s.startswith("deepspot"):
        return "deepspot"
    if s.startswith("exseq"):
        return "ExSeq"
    if s.startswith("simfish"):
        return "simfish"

    return "unknown"


def is_readable_tif(path: Path):
    try:
        _ = tifffile.imread(str(path))
        return True, "ok"
    except Exception as e:
        return False, str(e)


def collect_2d_images() -> pd.DataFrame:
    rows = []
    excluded_rows = []

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV directory not found: {CSV_DIR}")

    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(sorted(IMAGE_DIR.glob(f"*{ext}")))

    for img_path in image_paths:
        if img_path.name.startswith("._"):
            excluded_rows.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "reason": "mac_hidden_file",
            })
            continue

        if img_path.suffix.lower() in [".tif", ".tiff"]:
            ok, msg = is_readable_tif(img_path)
            if not ok:
                excluded_rows.append({
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "reason": f"unreadable_tif: {msg}",
                })
                continue

        stem = img_path.stem
        subset = infer_subset_from_stem(stem)

        csv_path = CSV_DIR / f"{stem}.csv"
        if not csv_path.exists():
            excluded_rows.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "reason": f"missing_csv: {csv_path}",
            })
            continue

        rows.append({
            "image_path": str(img_path),
            "csv_path": str(csv_path),
            "image_name": img_path.name,
            "csv_name": csv_path.name,
            "stem": stem,
            "subset": subset,
            "parent_dir": str(img_path.parent),
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    excluded_df = pd.DataFrame(excluded_rows)
    excluded_path = OUT_SPLIT_DIR / "excluded_files.csv"
    excluded_df.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    print(f"[COLLECTED] valid 2D image-csv pairs: {len(df)}")
    print(f"[SAVED EXCLUDED] {excluded_path} / excluded {len(excluded_df)}")

    return df


def split_dataset(df: pd.DataFrame):
    test_parts = []

    for subset in MAIN_SUBSETS:
        sub_df = df[df["subset"] == subset].copy()

        if len(sub_df) < TEST_N_PER_SUBSET:
            raise RuntimeError(
                f"{subset}: not enough images for test split. required {TEST_N_PER_SUBSET}, current {len(sub_df)}"
            )

        sampled = sub_df.sample(n=TEST_N_PER_SUBSET, random_state=SEED)
        test_parts.append(sampled)

    test_df = pd.concat(test_parts, ignore_index=True)
    test_paths = set(test_df["image_path"].tolist())

    remain_df = df[~df["image_path"].isin(test_paths)].copy().reset_index(drop=True)

    val_parts = []
    for subset, sub_df in remain_df.groupby("subset"):
        n_val = int(len(sub_df) * VAL_RATIO)

        if len(sub_df) > 0 and n_val == 0:
            n_val = 1

        if n_val > 0:
            val_parts.append(sub_df.sample(n=n_val, random_state=SEED))

    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=remain_df.columns)
    val_paths = set(val_df["image_path"].tolist())

    train_df = remain_df[~remain_df["image_path"].isin(val_paths)].copy().reset_index(drop=True)

    train_df["split"] = "train_raw"
    val_df["split"] = "val_raw"
    test_df["split"] = "rule_based_test"

    return train_df, val_df, test_df


def save_summary(all_df, train_df, val_df, test_df):
    rows = []

    for split_name, d in [
        ("all_2d_valid", all_df),
        ("train_raw", train_df),
        ("val_raw", val_df),
        ("rule_based_test", test_df),
    ]:
        for subset, count in d["subset"].value_counts().to_dict().items():
            rows.append({
                "split": split_name,
                "subset": subset,
                "count": count,
            })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["split", "subset"]).reset_index(drop=True)

    summary_path = OUT_SPLIT_DIR / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n===== Split Summary =====")
    print(summary_df)
    print(f"\nSAVED: {summary_path}")


def main():
    print("Done.")
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    print(f"CSV_DIR:   {CSV_DIR}")
    print(f"MAIN_SUBSETS: {MAIN_SUBSETS}")

    all_df = collect_2d_images()

    if len(all_df) == 0:
        raise RuntimeError("Input validation failed. Please check the required files and paths.")

    print("\n[subset distribution]")
    print(all_df["subset"].value_counts())

    unknown_df = all_df[all_df["subset"] == "unknown"].copy()
    if len(unknown_df) > 0:
        unknown_path = OUT_SPLIT_DIR / "unknown_files.csv"
        unknown_df.to_csv(unknown_path, index=False, encoding="utf-8-sig")
        print(f"\n[WARNING] unknown subset found: {len(unknown_df)} / {unknown_path}")

    all_path = OUT_SPLIT_DIR / "all_2d_valid_images.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"\nSAVED: {all_path}")

    print("Done.")
    train_df, val_df, test_df = split_dataset(all_df)

    train_path = OUT_SPLIT_DIR / "train_raw.csv"
    val_path = OUT_SPLIT_DIR / "val_raw.csv"
    test_path = OUT_SPLIT_DIR / "rule_based_test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"SAVED: {train_path}")
    print(f"SAVED: {val_path}")
    print(f"SAVED: {test_path}")

    save_summary(all_df, train_df, val_df, test_df)

    print("\nDone.")


if __name__ == "__main__":
    main()