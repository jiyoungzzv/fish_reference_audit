# Third-Party Notices

This repository accompanies the paper
“How Complete Should a Reference Be? A Benchmark Audit for
Fluorescence Spot Detection.”

This document identifies the principal third-party data and
software used in the study and distinguishes their use in the
experiments from their redistribution in this repository.
Third-party assets remain subject to their respective licenses.

## FISH_spots dataset

- Source: https://huggingface.co/datasets/GangCaoLab/FISH_spots
- Dataset DOI: https://doi.org/10.57967/hf/6150
- Upstream dataset-card license designation: MIT.
- Use: fluorescence microscopy images and point annotations
  used for training and benchmark auditing.

The available local dataset checkout also declares
`license: mit` in its README. This records the license designation
of that checkout; it does not establish the exact dataset revision
used at experiment time or independently establish the terms of
every constituent source.

The public audit repository provides processed reference
coordinates and evaluation metadata. Original microscopy image
files are obtained from the upstream dataset rather than
distributed as a standalone image dataset in this repository.

The processed audit references are derived research artifacts,
not the original annotations released by GangCaoLab. Their
provenance and construction are described in the paper and
repository documentation. This notice does not relicense the
upstream data or grant additional rights to derived materials.

## deepBlink

- Source: https://github.com/BBQuercus/deepBlink
- Source-code license: MIT.
- Copyright: Copyright (c) 2020, Bastian Eichenberger.
- License text:
  https://github.com/BBQuercus/deepBlink/blob/master/LICENSE
- Use: spot predictions from the official pretrained `smfish1`
  model.

The inspected public repository includes code for standardizing
deepBlink prediction outputs. It does not bundle the upstream
deepBlink package source or the `smfish1` model weights.

The exact deepBlink package version and pretrained-weight
revision were not recoverable from the inspected experiment
records. The MIT designation above refers to the upstream
source code; this notice does not assert a separate license
for the pretrained weights.

## Ultralytics YOLO

- Source: https://github.com/ultralytics/ultralytics
- Version recorded in the trained checkpoint metadata: 8.4.21.
- License recorded in the checkpoint metadata: AGPL-3.0.
- License text for that version:
  https://github.com/ultralytics/ultralytics/blob/v8.4.21/LICENSE
- Use: YOLO training and inference, with bounding-box centers
  converted to point predictions.

The local training configuration identifies `yolov8n.pt` as
the initialization model with pretrained weights enabled.
The trained `best.pt` checkpoint records Ultralytics version
8.4.21.

The public repository includes an inference script that imports
the separately installed `ultralytics` package. It does not
bundle the upstream Ultralytics package source or model
checkpoint files.

Ultralytics components remain subject to their applicable
license terms. Their absence as bundled files does not, by
itself, resolve the licensing requirements of an integration.
This notice does not grant an exception to AGPL-3.0 or assign
a replacement license to Ultralytics components.

## U-Net architecture and local implementations

- Architectural reference:
  Ronneberger, Fischer, and Brox,
  “U-Net: Convolutional Networks for Biomedical Image Segmentation,”
  MICCAI 2015.
- Paper: https://arxiv.org/abs/1505.04597
- Use: U-Net heatmap and U-Net heatmap+POS detectors.

The inspected local training scripts define a U-Net-style
network using PyTorch layers. The public audit repository
contains prediction-processing code but does not include
these training implementations or their trained checkpoints.

The architectural citation is not a software license and does
not establish the authorship or licensing of a particular
implementation. No claim of exclusive implementation
authorship is made in this notice.

## Software dependencies

The environment and dependency files identify additional
packages used by the released scripts. These packages are
installed separately and retain their respective licenses
and third-party notices.

The supplied environment files describe dependencies but do
not constitute a complete, version-locked record of the
historical experiment environment.

## Redistribution and license scope

Required upstream copyright and license notices must accompany
redistributed third-party material where the applicable terms
require them. This document supplements, rather than replaces,
those notices and license texts.

This notice does not grant a license to the authors' original
code or released derived audit artifacts. Their licensing must
be stated separately, with the applicable files and scope
identified. Any such license does not override the rights or
conditions attached to third-party components.
