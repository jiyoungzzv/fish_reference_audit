# Third-Party Notices

This repository accompanies the NeurIPS 2026 paper **“How Complete Should a Reference Be? A Benchmark Audit for Fluorescence Spot Detection.”**

The following third-party data and software were used in the research. Their licenses belong to their respective copyright holders. This list describes upstream resources; it **does not assign a license** to the original code or derived artifacts in this repository.

| Resource | Upstream source | License / terms | Use in this work | Redistribution in this repository |
|---|---|---|---|---|
| **FISH_spots** | [GangCaoLab/FISH_spots](https://huggingface.co/datasets/GangCaoLab/FISH_spots) | [MIT (dataset-page designation)](https://huggingface.co/datasets/GangCaoLab/FISH_spots) | Source fluorescence microscopy images and point annotations | Original microscopy images are **not** redistributed. Derived audit references and evaluation metadata are provided; verify their distribution terms against the source dataset. |
| **deepBlink** | [BBQuercus/deepBlink](https://github.com/BBQuercus/deepBlink) | [MIT](https://github.com/BBQuercus/deepBlink/blob/master/LICENSE) (upstream source code) | Official deepBlink baseline and pretrained `smfish1` model | Confirm whether model weights or any upstream code files are included; check applicable weight terms separately. |
| **Ultralytics** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (upstream repository; alternative commercial terms may be available) | YOLO training and/or inference tooling | Confirm whether any Ultralytics source code, modified code, or model weights are redistributed, and comply with applicable obligations. |
| **U-Net** | [Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597) | **Implementation-specific; not established by the paper citation** | Architecture used for the heatmap baseline | Determine whether this repository contains an original implementation or third-party code and document the corresponding license if applicable. |

## Versions and redistribution checks

Before finalizing a public release, record the exact versions or commits used for FISH_spots, deepBlink (including the `smfish1` weights), and Ultralytics. See `environment.yml`, `requirements.txt`, and the reproducibility documentation for available environment details; these may not identify the exact dataset/model revision.

If third-party source files, model weights, or other copyrighted material are included, preserve their required copyright and license notices and verify the relevant redistribution terms. Merely listing a license here does not establish that all use or distribution obligations have been fulfilled.

## License for this repository

The licensing of the authors' original code and derived audit artifacts is **not specified by this notice**. Any repository-level license should be selected separately after confirming ownership, institutional requirements, and compatibility with redistributed third-party material.
