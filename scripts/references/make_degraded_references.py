from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


DATASET_ROOT = PROJECT_ROOT / "dataset"

REF_DIR = PROJECT_ROOT / "results" / "references" / "standardized"

RAW_PATH = REF_DIR / "raw.csv"
LENIENT_PATH = REF_DIR / "lenient.csv"
STRICT_PATH = REF_DIR / "strict.csv"

OUT_ROOT = DATASET_ROOT / "annotations" / "degraded_references"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_ROOT / "degraded_reference_summary.csv"


COMPLETENESS_LEVELS = [100, 90, 80, 70, 60, 50]
POLICIES = [
    "random",
    "weak_evidence_first",
    "dense_first",
    "boundary_first",
]

RANDOM_SEEDS = list(range(20))  
BASE_SEED = 42


def prepare_raw_reference(strict_df: pd.DataFrame, weak_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = pd.concat([strict_df, weak_df], ignore_index=True)
    raw_df = raw_df.copy()
    raw_df["base_reference"] = "raw"
    return raw_df


def prepare_strict_reference(strict_df: pd.DataFrame) -> pd.DataFrame:
    strict = strict_df.copy()
    strict["base_reference"] = "strict_intersection"
    return strict


def add_required_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "reference_type" in df.columns:
        df["is_weak_raw"] = (df["reference_type"] == "weak_raw_evidence").astype(int)
    else:
        df["is_weak_raw"] = 0

    if "snr_proxy" not in df.columns:
        df["snr_proxy"] = np.nan

    if "nearest_neighbor_distance" not in df.columns:
        df["nearest_neighbor_distance"] = np.nan

    if "distance_to_boundary" not in df.columns:
        df["distance_to_boundary"] = np.nan

    return df


def select_keep_indices(group: pd.DataFrame,
                        completeness: int,
                        policy: str,
                        seed: int | None = None) -> pd.DataFrame:
    n = len(group)

    if n == 0:
        return group.copy()

    keep_n = int(round(n * completeness / 100.0))
    keep_n = max(0, min(keep_n, n))

    if completeness == 100:
        out = group.copy()
        out["deletion_rank"] = np.nan
        return out

    if keep_n == 0:
        return group.iloc[[]].copy()

    g = group.copy()

    if policy == "random":
        rng = np.random.default_rng(BASE_SEED if seed is None else BASE_SEED + seed)
        keep_idx = rng.choice(g.index.to_numpy(), size=keep_n, replace=False)
        out = g.loc[keep_idx].copy()
        out["deletion_rank"] = np.nan
        return out

    elif policy == "weak_evidence_first":
        g["_snr_fill"] = g["snr_proxy"].fillna(-1e9)
        g = g.sort_values(
            ["is_weak_raw", "_snr_fill"],
            ascending=[True, False]
        )
        out = g.head(keep_n).copy()
        out["deletion_rank"] = np.arange(len(out))
        out = out.drop(columns=["_snr_fill"], errors="ignore")
        return out

    elif policy == "dense_first":
        g["_nn_fill"] = g["nearest_neighbor_distance"].fillna(1e9)
        g = g.sort_values("_nn_fill", ascending=False)
        out = g.head(keep_n).copy()
        out["deletion_rank"] = np.arange(len(out))
        out = out.drop(columns=["_nn_fill"], errors="ignore")
        return out

    elif policy == "boundary_first":
        g["_bd_fill"] = g["distance_to_boundary"].fillna(1e9)
        g = g.sort_values("_bd_fill", ascending=False)
        out = g.head(keep_n).copy()
        out["deletion_rank"] = np.arange(len(out))
        out = out.drop(columns=["_bd_fill"], errors="ignore")
        return out

    else:
        raise ValueError(f"Unknown policy: {policy}")


def make_degraded(df: pd.DataFrame,
                  base_name: str,
                  completeness: int,
                  policy: str,
                  seed: int | None = None) -> pd.DataFrame:
    kept_parts = []

    for (subset, image_name), group in df.groupby(["subset", "image_name"]):
        kept = select_keep_indices(group, completeness, policy, seed)
        kept_parts.append(kept)

    if kept_parts:
        out = pd.concat(kept_parts, ignore_index=True)
    else:
        out = pd.DataFrame(columns=df.columns)

    out["base_reference"] = base_name
    out["completeness"] = completeness
    out["missing_policy"] = policy
    out["seed"] = seed if seed is not None else -1
    out["is_degraded_gt"] = True

    return out


def save_reference(df: pd.DataFrame,
                   base_name: str,
                   completeness: int,
                   policy: str,
                   seed: int | None = None) -> Path:
    out_dir = OUT_ROOT / base_name / policy
    out_dir.mkdir(parents=True, exist_ok=True)

    if policy == "random":
        filename = f"{base_name}_{completeness}_seed{seed:02d}.csv"
    else:
        filename = f"{base_name}_{completeness}.csv"

    out_path = out_dir / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main():
    for path in [RAW_PATH, LENIENT_PATH, STRICT_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"reference file not found: {path}")

    raw_ref = pd.read_csv(RAW_PATH)
    lenient_ref = pd.read_csv(LENIENT_PATH)
    strict_ref = pd.read_csv(STRICT_PATH)

    raw_ref = raw_ref.loc[:, ~raw_ref.columns.duplicated()].copy()
    lenient_ref = lenient_ref.loc[:, ~lenient_ref.columns.duplicated()].copy()
    strict_ref = strict_ref.loc[:, ~strict_ref.columns.duplicated()].copy()

    raw_ref["base_reference"] = "raw"
    lenient_ref["base_reference"] = "lenient"
    strict_ref["base_reference"] = "strict"

    raw_ref = add_required_features(raw_ref)
    lenient_ref = add_required_features(lenient_ref)
    strict_ref = add_required_features(strict_ref)

    references = {
        "raw": raw_ref,
        "lenient": lenient_ref,
        "strict": strict_ref,
    }

    summary_rows = []

    print("Done.")
    print(f"Raw reference rows: {len(raw_ref)}")
    print(f"Strict reference rows: {len(strict_ref)}")

    for base_name, base_df in references.items():
        print(f"\n===== Base reference: {base_name} =====")

        for policy in POLICIES:
            print(f"[policy] {policy}")

            if policy == "random":
                seeds = RANDOM_SEEDS
            else:
                seeds = [None]

            for completeness in COMPLETENESS_LEVELS:
                for seed in seeds:
                    degraded = make_degraded(
                        df=base_df,
                        base_name=base_name,
                        completeness=completeness,
                        policy=policy,
                        seed=seed,
                    )

                    out_path = save_reference(
                        degraded,
                        base_name=base_name,
                        completeness=completeness,
                        policy=policy,
                        seed=seed,
                    )

                    summary_rows.append({
                        "base_reference": base_name,
                        "policy": policy,
                        "completeness": completeness,
                        "seed": seed if seed is not None else -1,
                        "n_points": len(degraded),
                        "n_images": degraded["image_name"].nunique() if len(degraded) > 0 else 0,
                        "path": str(out_path),
                    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {OUT_SUMMARY}")
    print("\n===== summary head =====")
    print(summary.head(20))

    print("Done.")
    print(
        summary
        .groupby(["base_reference", "policy", "completeness"])["n_points"]
        .mean()
        .reset_index()
        .head(50)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()