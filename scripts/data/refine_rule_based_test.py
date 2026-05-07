from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


import pandas as pd


DATASET_ROOT = PROJECT_ROOT / "dataset"

SPLIT_DIR = DATASET_ROOT / "splits"
CAND_DIR = DATASET_ROOT / "candidates"

ALL_IMAGES_PATH = SPLIT_DIR / "all_2d_valid_images.csv"
TRAIN_PATH = SPLIT_DIR / "train_raw.csv"
VAL_PATH = SPLIT_DIR / "val_raw.csv"
TEST_PATH = SPLIT_DIR / "rule_based_test.csv"
CAND_SUMMARY_PATH = CAND_DIR / "test_candidate_summary.csv"

OUT_TEST_REFINED = SPLIT_DIR / "rule_based_test_refined.csv"
OUT_TRAIN_REFINED = SPLIT_DIR / "train_raw_refined.csv"
OUT_VAL_REFINED = SPLIT_DIR / "val_raw_refined.csv"
OUT_CONTROL = SPLIT_DIR / "sparse_or_near_empty_control.csv"
OUT_SUMMARY = SPLIT_DIR / "refined_split_summary.csv"
OUT_REPLACEMENTS = SPLIT_DIR / "replacement_log.csv"


SEED = 42
MAIN_SUBSETS = ["MERFISH", "RCA", "SeqFISH"]
TARGET_TEST_N = 50

LOW_COUNT_QUANTILE = 0.10

EXTREME_RATIO_THRESHOLD = 50.0

REPLACEMENT_MIN_QUANTILE = 0.10


def read_spot_count(csv_path: str) -> int:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    df = df.dropna(how="all").reset_index(drop=True)
    return len(df)


def add_raw_spot_count_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "raw_spot_count" in df.columns:
        return df

    counts = []
    for _, row in df.iterrows():
        counts.append(read_spot_count(row["csv_path"]))

    df["raw_spot_count"] = counts
    return df


def make_initial_exclusion(test_df: pd.DataFrame, cand_summary_df: pd.DataFrame) -> pd.DataFrame:
    test_df = test_df.copy()

    cand_cols = [
        "image_name",
        "candidate_count",
        "candidate_raw_ratio",
        "median_snr_proxy",
        "median_log_response",
        "median_nn_distance",
    ]
    cand_cols = [c for c in cand_cols if c in cand_summary_df.columns]

    merged = test_df.merge(
        cand_summary_df[cand_cols],
        on="image_name",
        how="left"
    )

    merged["exclude_main_test"] = False
    merged["exclude_reason"] = ""

    for subset, sub_df in merged.groupby("subset"):
        q_low = sub_df["raw_spot_count"].quantile(LOW_COUNT_QUANTILE)

        mask_subset = merged["subset"] == subset
        mask_low = mask_subset & (merged["raw_spot_count"] <= q_low)

        merged.loc[mask_low, "exclude_main_test"] = True
        merged.loc[mask_low, "exclude_reason"] = (
            merged.loc[mask_low, "exclude_reason"].astype(str)
            + f"low_raw_count_q{LOW_COUNT_QUANTILE};"
        )

    if "candidate_raw_ratio" in merged.columns:
        mask_ratio = merged["candidate_raw_ratio"] > EXTREME_RATIO_THRESHOLD
        merged.loc[mask_ratio, "exclude_main_test"] = True
        merged.loc[mask_ratio, "exclude_reason"] = (
            merged.loc[mask_ratio, "exclude_reason"].astype(str)
            + f"extreme_candidate_raw_ratio_gt_{EXTREME_RATIO_THRESHOLD};"
        )

    return merged


def prepare_replacement_pool(all_df: pd.DataFrame,
                             current_test_df: pd.DataFrame,
                             train_df: pd.DataFrame,
                             val_df: pd.DataFrame) -> pd.DataFrame:
    all_df = all_df.copy()
    all_df = add_raw_spot_count_if_missing(all_df)

    current_test_paths = set(current_test_df["image_path"].tolist())

    pool = all_df[
        all_df["subset"].isin(MAIN_SUBSETS)
        & (~all_df["image_path"].isin(current_test_paths))
    ].copy()

    pool["eligible_replacement"] = True
    pool["replacement_filter_reason"] = ""

    for subset, sub_df in all_df[all_df["subset"].isin(MAIN_SUBSETS)].groupby("subset"):
        q_min = sub_df["raw_spot_count"].quantile(REPLACEMENT_MIN_QUANTILE)

        mask_subset = pool["subset"] == subset
        mask_low = mask_subset & (pool["raw_spot_count"] <= q_min)

        pool.loc[mask_low, "eligible_replacement"] = False
        pool.loc[mask_low, "replacement_filter_reason"] += f"below_subset_q{REPLACEMENT_MIN_QUANTILE};"

    pool = pool[pool["eligible_replacement"]].copy()

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["original_split"] = "train_raw"
    val_df["original_split"] = "val_raw"

    split_info = pd.concat([
        train_df[["image_path", "original_split"]],
        val_df[["image_path", "original_split"]],
    ], ignore_index=True)

    pool = pool.merge(split_info, on="image_path", how="left")
    pool["original_split"] = pool["original_split"].fillna("unknown")

    return pool


def select_replacements(pool: pd.DataFrame,
                        refined_test_df: pd.DataFrame,
                        excluded_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    replacement_rows = []
    log_rows = []

    used_paths = set(refined_test_df["image_path"].tolist())

    for subset in MAIN_SUBSETS:
        current_n = len(refined_test_df[refined_test_df["subset"] == subset])
        need_n = TARGET_TEST_N - current_n

        print(f"[replacement] {subset}: current {current_n}, required {need_n}")

        if need_n <= 0:
            continue

        sub_pool = pool[
            (pool["subset"] == subset)
            & (~pool["image_path"].isin(used_paths))
        ].copy()

        if len(sub_pool) < need_n:
            raise RuntimeError(
                f"{subset}: replacement not enough candidates. required {need_n}, candidates {len(sub_pool)}"
            )

        sampled = sub_pool.sample(n=need_n, random_state=SEED)

        for _, r in sampled.iterrows():
            log_rows.append({
                "subset": subset,
                "replacement_image": r["image_name"],
                "replacement_path": r["image_path"],
                "replacement_raw_spot_count": r["raw_spot_count"],
                "original_split": r.get("original_split", "unknown"),
                "reason": "replace_near_empty_or_outlier_in_rule_test",
            })

        replacement_rows.append(sampled)
        used_paths.update(sampled["image_path"].tolist())

    if replacement_rows:
        replacement_df = pd.concat(replacement_rows, ignore_index=True)
    else:
        replacement_df = pd.DataFrame(columns=pool.columns)

    replacement_log_df = pd.DataFrame(log_rows)

    return replacement_df, replacement_log_df


def rebuild_train_val(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      replacement_df: pd.DataFrame,
                      excluded_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    replacement_paths = set(replacement_df["image_path"].tolist())
    excluded_paths = set(excluded_df["image_path"].tolist())

    train_refined = train_df[~train_df["image_path"].isin(replacement_paths)].copy()
    val_refined = val_df[~val_df["image_path"].isin(replacement_paths)].copy()

    train_refined = train_refined[~train_refined["image_path"].isin(excluded_paths)].copy()
    val_refined = val_refined[~val_refined["image_path"].isin(excluded_paths)].copy()

    train_refined["split"] = "train_raw_refined"
    val_refined["split"] = "val_raw_refined"

    return train_refined.reset_index(drop=True), val_refined.reset_index(drop=True)


def save_summary(train_df, val_df, test_df, control_df):
    rows = []

    for split_name, d in [
        ("train_raw_refined", train_df),
        ("val_raw_refined", val_df),
        ("rule_based_test_refined", test_df),
        ("sparse_or_near_empty_control", control_df),
    ]:
        if len(d) == 0:
            continue

        for subset, count in d["subset"].value_counts().to_dict().items():
            rows.append({
                "split": split_name,
                "subset": subset,
                "count": count
            })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["split", "subset"]).reset_index(drop=True)
    summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n===== Refined Split Summary =====")
    print(summary_df)
    print(f"\n[SAVED] {OUT_SUMMARY}")


def main():
    print("Done.")

    all_df = pd.read_csv(ALL_IMAGES_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    cand_summary_df = pd.read_csv(CAND_SUMMARY_PATH)

    test_df = add_raw_spot_count_if_missing(test_df)

    print(f"all images: {len(all_df)}")
    print(f"train_raw: {len(train_df)}")
    print(f"val_raw: {len(val_df)}")
    print(f"rule_based_test initial: {len(test_df)}")

    print("Done.")
    test_with_flags = make_initial_exclusion(test_df, cand_summary_df)

    excluded_df = test_with_flags[test_with_flags["exclude_main_test"]].copy()
    kept_test_df = test_with_flags[~test_with_flags["exclude_main_test"]].copy()

    print(f"excluded candidates: {len(excluded_df)}")
    print(excluded_df.groupby("subset").size())

    print("\n[3] replacement pool create")
    pool = prepare_replacement_pool(all_df, test_df, train_df, val_df)

    print("\n[4] replacement select")
    replacement_df, replacement_log_df = select_replacements(pool, kept_test_df, excluded_df)

    refined_test_df = pd.concat([kept_test_df, replacement_df], ignore_index=True)

    refined_test_df["split"] = "rule_based_test_refined"

    final_counts = refined_test_df["subset"].value_counts()
    print("\n[final rule_based_test_refined distribution]")
    print(final_counts)

    for subset in MAIN_SUBSETS:
        n = final_counts.get(subset, 0)
        if n != TARGET_TEST_N:
            raise RuntimeError(f"{subset}: final test count is {n}. target {TARGET_TEST_N}.")

    print("\n[5] train/val refined rebuild")
    train_refined, val_refined = rebuild_train_val(train_df, val_df, replacement_df, excluded_df)

    refined_test_df.to_csv(OUT_TEST_REFINED, index=False, encoding="utf-8-sig")
    train_refined.to_csv(OUT_TRAIN_REFINED, index=False, encoding="utf-8-sig")
    val_refined.to_csv(OUT_VAL_REFINED, index=False, encoding="utf-8-sig")
    excluded_df.to_csv(OUT_CONTROL, index=False, encoding="utf-8-sig")
    replacement_log_df.to_csv(OUT_REPLACEMENTS, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {OUT_TEST_REFINED}")
    print(f"[SAVED] {OUT_TRAIN_REFINED}")
    print(f"[SAVED] {OUT_VAL_REFINED}")
    print(f"[SAVED] {OUT_CONTROL}")
    print(f"[SAVED] {OUT_REPLACEMENTS}")

    save_summary(train_refined, val_refined, refined_test_df, excluded_df)

    print("\nDone.")


if __name__ == "__main__":
    main()