from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT

STRICT_PATH = ROOT / r"dataset\annotations\strict_intersection_refined.csv"
WEAK_RAW_PATH = ROOT / r"dataset\annotations\weak_raw_refined.csv"
RULE_ONLY_PATH = ROOT / r"dataset\annotations\rule_only_disagreement_refined.csv"
SPLIT_PATH = ROOT / r"dataset\splits\rule_based_test_refined.csv"

OUT_DIR = ROOT / r"results\references\standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_split_stems():
    split_df = pd.read_csv(SPLIT_PATH)

    if "stem" in split_df.columns:
        image_id = split_df["stem"].astype(str)
    elif "image_id" in split_df.columns:
        image_id = split_df["image_id"].astype(str).apply(lambda x: Path(x).stem)
    elif "image_name" in split_df.columns:
        image_id = split_df["image_name"].astype(str).apply(lambda x: Path(x).stem)
    elif "image_path" in split_df.columns:
        image_id = split_df["image_path"].astype(str).apply(lambda x: Path(x).stem)
    else:
        raise ValueError(f"Cannot detect image id column in split. Columns: {list(split_df.columns)}")

    return set(image_id.tolist())


def standardize_component(path, component_name):
    df = pd.read_csv(path)

    required = ["image_name", "stem", "subset", "ref_x", "ref_y"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    out = pd.DataFrame({
        "image_id": df["stem"].astype(str),
        "image_name": df["image_name"].astype(str),
        "subset": df["subset"].astype(str),
        "x": pd.to_numeric(df["ref_x"], errors="coerce"),
        "y": pd.to_numeric(df["ref_y"], errors="coerce"),
        "component": component_name,
    })

    if "reference_type" in df.columns:
        out["reference_type"] = df["reference_type"].astype(str)
    else:
        out["reference_type"] = component_name

    out = out.dropna(subset=["image_id", "subset", "x", "y"]).copy()

    return out


def deduplicate_points(df, ref_name):
    out = df.copy()
    out["ref_name"] = ref_name

    out["x_round"] = out["x"].round(3)
    out["y_round"] = out["y"].round(3)

    before = len(out)

    out = out.drop_duplicates(
        subset=["image_id", "subset", "x_round", "y_round"]
    ).copy()

    after = len(out)

    out = out.drop(columns=["x_round", "y_round"])

    print(f"[DEDUP] {ref_name}: {before} -> {after}, removed {before - after}")

    cols = [
        "image_id",
        "image_name",
        "subset",
        "ref_name",
        "x",
        "y",
        "component",
        "reference_type",
    ]

    return out[cols]


def save_reference(df, ref_name):
    out_path = OUT_DIR / f"{ref_name}.csv"
    summary_path = OUT_DIR / f"{ref_name}_summary.csv"

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("subset")
        .agg(
            n_images=("image_id", "nunique"),
            n_points=("x", "count"),
            mean_points_per_image=("x", lambda s: len(s) / df.loc[s.index, "image_id"].nunique()),
        )
        .reset_index()
    )

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"[SAVED] {ref_name}")
    print(out_path)
    print(summary_path)
    print(summary.to_string(index=False))


def main():
    print("[INFO] Standardize references")
    print(f"OUT_DIR: {OUT_DIR}")

    allowed_stems = get_split_stems()
    print(f"[INFO] refined test images: {len(allowed_stems)}")

    strict = standardize_component(STRICT_PATH, "strict_intersection")
    weak_raw = standardize_component(WEAK_RAW_PATH, "weak_raw")
    rule_only = standardize_component(RULE_ONLY_PATH, "rule_only_disagreement")

    strict = strict[strict["image_id"].isin(allowed_stems)].copy()
    weak_raw = weak_raw[weak_raw["image_id"].isin(allowed_stems)].copy()
    rule_only = rule_only[rule_only["image_id"].isin(allowed_stems)].copy()

    print("\n[COMPONENT SUMMARY]")
    for name, df in [
        ("strict_intersection", strict),
        ("weak_raw", weak_raw),
        ("rule_only_disagreement", rule_only),
    ]:
        print("\n" + name)
        print(
            df.groupby("subset")
            .agg(n_images=("image_id", "nunique"), n_points=("x", "count"))
            .reset_index()
            .to_string(index=False)
        )

    raw = pd.concat([strict, weak_raw], ignore_index=True)
    lenient = pd.concat([strict, rule_only], ignore_index=True)
    raw_plus_lenient = pd.concat([strict, weak_raw, rule_only], ignore_index=True)

    raw = deduplicate_points(raw, "raw")
    lenient = deduplicate_points(lenient, "lenient")
    raw_plus_lenient = deduplicate_points(raw_plus_lenient, "raw_plus_lenient")

    save_reference(raw, "raw")
    save_reference(lenient, "lenient")
    save_reference(raw_plus_lenient, "raw_plus_lenient")

    strict_ref = deduplicate_points(strict, "strict")
    save_reference(strict_ref, "strict")

    all_refs = pd.concat(
        [raw, lenient, raw_plus_lenient, strict_ref],
        ignore_index=True
    )

    all_path = OUT_DIR / "all_references.csv"
    all_refs.to_csv(all_path, index=False, encoding="utf-8-sig")

    all_summary = (
        all_refs.groupby(["ref_name", "subset"])
        .agg(
            n_images=("image_id", "nunique"),
            n_points=("x", "count"),
        )
        .reset_index()
    )

    all_summary_path = OUT_DIR / "all_references_summary.csv"
    all_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("[DONE]")
    print(all_path)
    print(all_summary_path)
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    main()