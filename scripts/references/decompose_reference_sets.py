from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

import pandas as pd


DATASET_ROOT = PROJECT_ROOT / "dataset"

ANN_DIR = DATASET_ROOT / "annotations"

MATCH_PATH = ANN_DIR / "raw_candidate_matching_refined.csv"

OUT_STRICT = ANN_DIR / "strict_intersection_refined.csv"
OUT_WEAK_RAW = ANN_DIR / "weak_raw_refined.csv"
OUT_RULE_ONLY = ANN_DIR / "rule_only_disagreement_refined.csv"
OUT_SUMMARY = ANN_DIR / "reference_decomposition_summary_refined.csv"


def main():
    if not MATCH_PATH.exists():
        raise FileNotFoundError(f"matching file not found: {MATCH_PATH}")

    df = pd.read_csv(MATCH_PATH)

    df = df.loc[:, ~df.columns.duplicated()].copy()

    required_cols = ["image_name", "subset", "status"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"required column missing: {c}")

    # Reference decomposition
    strict_df = df[df["status"] == "matched"].copy()
    weak_raw_df = df[df["status"] == "raw_only"].copy()
    rule_only_df = df[df["status"] == "candidate_only"].copy()

    if len(strict_df) > 0:
        strict_df["ref_x"] = strict_df["raw_x"]
        strict_df["ref_y"] = strict_df["raw_y"]
        strict_df["reference_type"] = "strict_intersection_raw_and_rule"
        strict_df["is_ground_truth_for_main"] = True

    if len(weak_raw_df) > 0:
        weak_raw_df["ref_x"] = weak_raw_df["raw_x"]
        weak_raw_df["ref_y"] = weak_raw_df["raw_y"]
        weak_raw_df["reference_type"] = "weak_raw_evidence"
        weak_raw_df["is_ground_truth_for_main"] = True

    if len(rule_only_df) > 0:
        rule_only_df["ref_x"] = rule_only_df["candidate_x"]
        rule_only_df["ref_y"] = rule_only_df["candidate_y"]
        rule_only_df["reference_type"] = "rule_only_disagreement_signal"
        rule_only_df["is_ground_truth_for_main"] = False

    strict_df.to_csv(OUT_STRICT, index=False, encoding="utf-8-sig")
    weak_raw_df.to_csv(OUT_WEAK_RAW, index=False, encoding="utf-8-sig")
    rule_only_df.to_csv(OUT_RULE_ONLY, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {OUT_STRICT} / {len(strict_df)} rows")
    print(f"[SAVED] {OUT_WEAK_RAW} / {len(weak_raw_df)} rows")
    print(f"[SAVED] {OUT_RULE_ONLY} / {len(rule_only_df)} rows")

    id_cols = ["subset", "image_name"]
    if "stem" in df.columns:
        id_cols.append("stem")

    summary = (
        df
        .groupby(id_cols + ["status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["matched", "raw_only", "candidate_only"]:
        if col not in summary.columns:
            summary[col] = 0

    summary = summary.rename(columns={
        "matched": "strict_intersection_count",
        "raw_only": "weak_raw_count",
        "candidate_only": "rule_only_disagreement_count",
    })

    summary["raw_total_count"] = (
        summary["strict_intersection_count"] + summary["weak_raw_count"]
    )
    summary["rule_total_count"] = (
        summary["strict_intersection_count"] + summary["rule_only_disagreement_count"]
    )

    summary["strict_fraction_of_raw"] = (
        summary["strict_intersection_count"] / summary["raw_total_count"].replace(0, pd.NA)
    )
    summary["weak_fraction_of_raw"] = (
        summary["weak_raw_count"] / summary["raw_total_count"].replace(0, pd.NA)
    )
    summary["rule_only_fraction_of_rule"] = (
        summary["rule_only_disagreement_count"] / summary["rule_total_count"].replace(0, pd.NA)
    )

    summary = summary.sort_values(["subset", "image_name"]).reset_index(drop=True)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {OUT_SUMMARY} / {len(summary)} images")

    print("Done.")
    print(
        summary
        .groupby("subset")[
            [
                "raw_total_count",
                "strict_intersection_count",
                "weak_raw_count",
                "rule_only_disagreement_count",
                "strict_fraction_of_raw",
                "weak_fraction_of_raw",
                "rule_only_fraction_of_rule",
            ]
        ]
        .mean()
    )

    print("Done.")
    print(
        summary
        .groupby("subset")[
            [
                "raw_total_count",
                "strict_intersection_count",
                "weak_raw_count",
                "rule_only_disagreement_count",
                "strict_fraction_of_raw",
                "weak_fraction_of_raw",
                "rule_only_fraction_of_rule",
            ]
        ]
        .median()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()