# Path 4 — RGB video baseline on BdSLW60-SI (B2 upgrade)

**Goal**: the RGB row reviewers expect next to the skeleton models. ASL
Citizen's NeurIPS D&B 2023 baselines were exactly I3D (63.10 Top-1) vs
ST-GCN (59.52) under signer-independent evaluation; an SI benchmark without
an RGB baseline looks incomplete, and it doubles as the number that
justifies the pose-only design choice.

## Layout

```
path4_rgb_baseline/
├── README.md
├── rgb_dataset.py            BdSLW60 raw-video dataset; split + labels come
│                             from preprocessing.bdsl_signer_split and
│                             data/bdsl_si/classes.json (identical to pose NPYs)
├── train_rgb.py              entry point (torchvision video models or I3D)
├── pytorch_i3d.py            vendored I3D (piergiaj/pytorch-i3d, Apache-2.0;
│                             see LICENSE.pytorch_i3d.txt)
└── configs/
    ├── s3d_bdsl_si.yaml      the real run (Kinetics-400-pretrained S3D)
    └── s3d_bdsl_si_smoke.yaml ~3-min smoke (no downloads)
```

## Run

```bash
# smoke first (no weight download, 40 clips/split, 1 epoch)
python -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si_smoke.yaml

# real run — S3D (NLA-SLR's encoder family), Kinetics-400 weights
# auto-download via torchvision on first use
python -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si.yaml --seed 0
```

Results land in `results_final.csv` (`rgb_s3d_bdsl_si` = val rows,
`rgb_s3d_bdsl_si_testset` = final test row) like every other run.

## Architecture choices

| `model_name` | Source | Why |
|---|---|---|
| `s3d` (default) | torchvision, KINETICS400_V1 | S3D is NLA-SLR's encoder; official weights |
| `r2plus1d_18` | torchvision | cheaper CNN alternative |
| `mvit_v2_s` | torchvision (needs `num_frames: 16`) | the arXiv 2412.11553 SOTA-recipe family |
| `i3d` | vendored `pytorch_i3d.py` | exact ASL Citizen parity |

For `i3d` with Kinetics pretraining, download the checkpoint yourself (not
auto-fetched) and set `i3d_weights` in the config:

```bash
curl -L -o path4_rgb_baseline/weights/rgb_imagenet.pt \
  https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt
```

## Notes

- Video decode is the bottleneck at `num_worker: 0` (~Windows-safe default).
  cv2 decoding in workers is fine — raise `num_worker` on HPC.
- Horizontal flip is deliberately NOT used: BdSL hand dominance is
  class-informative (same reason `random_mirror: False` in the pose configs).
- Paper reporting: this row is *RGB, Kinetics-400-pretrained* — keep it in a
  separate comparison track from pose-only models (cf. NLA-SLR headline vs
  keypoint-only ablation, 61.05 vs 49.10 on WLASL-2000).
