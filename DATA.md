# Data documentation

## Raw data

This project uses the public FISH_spots fluorescence microscopy dataset. The raw microscopy images are not redistributed in this repository.

Users should download the raw dataset from the original public release and set the dataset path using:

export FISH_SPOTS_ROOT=/path/to/FISH_spots

The expected raw dataset structure is:

FISH_SPOTS_ROOT/
  2d/
    image/
    csv/

## Audit split

The audit experiments use held-out test images from MERFISH, RCA, and SeqFISH subsets. Split files and image-level metadata are provided in:

data_manifest/

## Derived assets

This repository provides code for constructing derived audit references:

- Raw: dataset-provided annotations
- Lenient: Raw plus candidate-supported plausible additional spots
- Strict: candidate-supported Raw annotations

The repository also provides scripts for generating degraded references used in annotation-retention stress tests.

## Existing assets and licenses

The paper cites the original FISH_spots dataset and the detector/model references used in the study. Users should follow the original licenses and terms of use of each dataset, software package, and model implementation.
