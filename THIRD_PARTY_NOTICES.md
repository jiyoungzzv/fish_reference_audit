# Third-party assets and license notices

This document records the sources and upstream license information
for third-party assets used in this study.

## FISH_spots dataset

- Source: https://huggingface.co/datasets/GangCaoLab/FISH_spots
- Dataset DOI: https://doi.org/10.57967/hf/6150
- License listed on the upstream dataset page: MIT.
- Use in this study: microscopy images and reference annotations
  for the benchmark audit.
- The public audit repository includes processed reference coordinates.
  Original microscopy images are obtained from the upstream source.

The upstream dataset page and the README in the local FISH_spots copy both identify the dataset license as MIT.
The exact dataset revision used in the experiments and any additional terms for constituent sources remain to be verified.
Upstream copyright and license notices must be preserved where required.

## deepBlink

- Source: https://github.com/BBQuercus/deepBlink
- Code license: MIT.
- License text:
  https://github.com/BBQuercus/deepBlink/blob/master/LICENSE
- Use in this study: spot predictions from the official pretrained
  `smfish1` model.

The code license is documented above. The exact software version
and any separate terms applicable to the pretrained model weights
remain to be verified.

## Ultralytics YOLO

- Source: https://github.com/ultralytics/ultralytics
- Upstream code license: AGPL-3.0.
- License text:
  https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Use in this study: object detection, with bounding-box centers
  converted to point predictions.

The exact software version, model provenance, and license obligations
applicable to the distributed inference integration remain to be verified.
No replacement license is asserted for Ultralytics components.

## U-Net implementations

The study evaluates U-Net heatmap and U-Net heatmap+POS detectors.
The provenance of their implementations, including any reused
third-party code and its license, remains to be documented.

## Scope and remaining checks

This is an initial inventory of the principal research assets.
It does not replace upstream license texts or grant additional rights.

Before finalizing this notice, document the versions or revisions used,
check pretrained-weight terms and any copied code, preserve required
copyright notices, and complete the inventory of other third-party
components.

Licensing for the authors' original code and released derived data
is addressed separately.
