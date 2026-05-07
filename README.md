# Reference-audit reproducibility package

This repository contains the code and precomputed intermediate files used to reproduce the reference-audit analyses and main figures reported in the paper.

The main analyses operate on standardized detector predictions and audit references. The provided files support direct reproduction of the reference-sensitivity analysis, annotation-retention stress tests, ranking-stability analysis, top-1 stability analysis, and main figure generation.

## Environment setup

We recommend using Python 3.10 or later.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the main scripts require standard scientific Python packages, including:

```text
numpy
pandas
scipy
matplotlib
seaborn
scikit-image
tifffile
```

## Repository structure

```text
scripts/data/
  Construct and refine the held-out FISH_spots audit split.

scripts/references/
  Generate rule-based candidate evidence.
  Construct Raw, Strict, and Lenient audit references.
  Generate degraded references for annotation-retention stress tests.

scripts/predictions/
  Convert detector outputs into a common standardized point-prediction format.

scripts/evaluation/
  Evaluate detector predictions under a unified point-matching protocol.
  Run reference-sensitivity, missingness-stress, ranking-stability, and top-1 stability analyses.

scripts/figures/
  Generate the main paper figures from the provided result files.

results/
  Precomputed references, standardized predictions, evaluation outputs, and figure-ready summaries.
```

## Analysis pipeline

The code follows the analysis pipeline described in the paper:

1. Construct the held-out FISH_spots audit split.
2. Generate rule-based candidate evidence.
3. Build Raw, Strict, and Lenient audit references.
4. Standardize detector outputs into a common point format.
5. Evaluate all detectors using unified one-to-one point matching.
6. Run annotation-retention stress tests under random and structured missingness.
7. Analyze ranking stability and top-1 model-selection stability.
8. Generate the main paper figures.

## Audit reference views

The repository includes three audit reference views:

- `Raw`: dataset-provided annotations.
- `Strict`: candidate-supported Raw annotations.
- `Lenient`: Raw annotations plus candidate-supported plausible additional spots.

These references are used as controlled audit views rather than corrected biological ground truth.

## Standardized detector predictions

The audit analyses are performed on standardized detector predictions. Each detector output is converted to a common point-prediction schema before evaluation.

The standardized prediction files are used as the common input for all reference-sensitivity, missingness-stress, ranking-stability, and top-1 stability analyses.

The common schema is:

```text
image_name, subset, method, x, y, score
```

This design separates detector-output standardization from the reference-audit analyses, ensuring that all detectors are evaluated under the same point-matching protocol.

## Note on rule-based candidates

Rule-based candidates are used as independent candidate evidence for constructing audit references. A rule-candidate detector is also evaluated as one baseline, but all detector outputs are evaluated under the same unified point-matching protocol.

## Main figure scripts

| Paper result | Script |
|---|---|
| Reference sensitivity / benchmark outcome analysis | `scripts/figures/fig2_3.py` |
| Missingness stress-test curves | `scripts/figures/fig4_A.py` |
| Missingness-induced F1 drop summary | `scripts/figures/fig4_B.py` |
| Ranking-stability analysis | `scripts/figures/fig5.py` |
| Top-1 stability / additional ranking analysis | `scripts/figures/fig6.py` |

## Reproducing the main analyses

Run the following scripts from the repository root.

```bash
# Reference-definition sensitivity analysis
python scripts/evaluation/evaluate_reference_sensitivity.py

# Generate degraded references for structured missingness stress tests
python scripts/references/make_degraded_references.py

# Structured missingness evaluation across all methods
python scripts/evaluation/evaluate_missingness_all_methods.py

# Ranking-stability and top-model-change analysis
python scripts/evaluation/analyze_ranking_stability.py
Then generate the main figures:

```bash
python scripts/figures/fig2_3.py
python scripts/figures/fig4_A.py
python scripts/figures/fig4_B.py
python scripts/figures/fig5.py
python scripts/figures/fig6.py
```

Generated figures are saved by the corresponding figure scripts under the configured output directory.

See `REPRODUCE.md` for a step-by-step reproduction guide.

## Notes

This repository is organized around audit-result reproducibility. The provided standardized predictions and audit references are the inputs used for the main reference-sensitivity, missingness-stress, ranking-stability, and figure-generation analyses.

## License

This code is provided for anonymous peer review. License information will be finalized upon publication.

## Optional detector-inference dependencies

The main audit analyses use the core dependencies listed in `requirements.txt`.

Optional detector-inference scripts, such as `scripts/predictions/predict_yolo_pointbox.py`, require additional packages listed in `requirements-optional.txt`:

```bash
pip install -r requirements-optional.txt
```
