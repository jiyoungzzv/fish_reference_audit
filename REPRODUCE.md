# Reproducing the reference-audit analyses

This document describes how to reproduce the main audit analyses and figures from the provided intermediate files.

## Overview

The main paper results are reproduced from:

- standardized detector predictions,
- Raw, Strict, and Lenient audit references,
- degraded references for annotation-retention stress tests,
- evaluation scripts,
- ranking-stability analysis scripts,
- figure-generation scripts.

The repository is organized around audit-result reproducibility.

## Step 1. Check the repository

From the repository root, run:

```bash
python -m compileall -q scripts
```

No output indicates that the Python files compile successfully.

## Step 2. Reference-sensitivity evaluation

Run:

```bash
python scripts/evaluation/evaluate_reference_sensitivity.py
```

This evaluates detector predictions under the Raw, Strict, and Lenient audit references using the unified point-matching protocol.

## Step 3. Missingness stress-test evaluation

Run:

```bash
python scripts/evaluation/evaluate_missingness_all_methods.py
```

This evaluates detector behavior under random and structured annotation-retention stress tests.

If Lenient missingness curves need to be regenerated separately, run:

```bash
python scripts/evaluation/add_lenient_missingness_curves.py
```

## Step 4. Ranking and top-1 stability analysis

Run:

```bash
python scripts/evaluation/analyze_ranking_stability.py
```

This script analyzes rank stability and top-1 model-selection stability across reference views, missingness policies, and retained-reference levels.

## Step 5. Generate main paper figures

Run:

```bash
python scripts/figures/fig2_3.py
python scripts/figures/fig4_A.py
python scripts/figures/fig4_B.py
python scripts/figures/fig5.py
python scripts/figures/fig6.py
```

## Paper-result mapping

| Paper result | Main script |
|---|---|
| Reference sensitivity and benchmark outcome changes | `scripts/figures/fig2_3.py` |
| Missingness stress-test curves | `scripts/figures/fig4_A.py` |
| Missingness-induced F1 drop summary | `scripts/figures/fig4_B.py` |
| Ranking-stability analysis | `scripts/figures/fig5.py` |
| Top-1 stability / additional ranking analysis | `scripts/figures/fig6.py` |

## Main data artifacts

The key provided artifacts are:

```text
results/references/
  Audit reference files.

results/predictions/standardized/
  Standardized detector predictions.

results/evaluation/
  Evaluation outputs used for analysis and figure generation.
```

## Notes

The audit references are controlled evaluation views, not corrected biological ground truth.

The Raw reference follows the dataset-provided annotations. The Strict reference retains candidate-supported Raw annotations. The Lenient reference includes Raw annotations plus candidate-supported plausible additional spots.

Rule-based candidates are used as independent candidate evidence for constructing audit references. A rule-candidate detector is also evaluated as one baseline, but all detector outputs are evaluated under the same unified point-matching protocol.
