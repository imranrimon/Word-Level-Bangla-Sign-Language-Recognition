# Path 2 — Auxiliary handshape head on BlockGCN (knowledge distillation)

**Goal**: improve BlockGCN classification on BdSLW60-SI by distilling
handshape knowledge from a Bangla-handshape image teacher (Path 1's
LoRA-adapted DINOv2). Modernised version of Koller et al.'s "deep-hand"
approach (CVPR 2016) — adapted for skeleton input + foundation-model teacher.

## Where it sits in the project

Path 2 composes with what we already have:

* **Teacher** = Path 1's Bangla-DINOv2 encoder (or Option B's stock DINOv2
  as a fallback before Path 1 finishes).
* **Hand crops per frame** = Option B's `preprocessing/extract_dinov2_features.py`
  output (V=3 regions: L hand, R hand, face). Path 2 uses only L+R.
* **Student** = BlockGCN (with a small projection head appended to the
  pre-classifier feature).
* **Pose data** = `data/bdsl_si/` from Stage 0.

So Path 2 introduces **only one new model** (`model.block_gcn_kd.Model`),
**one new feeder** (`feeders.kd_feeder.KDFeeder`), and **one new training
entry point** (`path2_handshape_kd.train_kd`). Everything else is reused.

## Layout

```
path2_handshape_kd/
├── README.md            (this file)
├── configs/
│   └── train_kd.yaml    KD hyperparameters (kd_weight, kd_loss kind, teacher_dim)
└── train_kd.py          training entry point
```

Sister files in the rest of the repo:

| File | Role |
|---|---|
| `model/block_gcn_kd.py`     | BlockGCN + projection head; forward returns (logits, student_proj) |
| `feeders/kd_feeder.py`      | yields (pose, teacher_pooled, label) per sample |
| `path2_handshape_kd/configs/train_kd.yaml` | hyperparameters; defaults to Path 1's encoder, falls back to stock DINOv2 |
| `tests/test_path2_smoke.py` | shape + KD-loss + grad-flow tests |

## Distillation design (one paragraph)

The student is BlockGCN with `return_features=True`; we read its pooled
pre-classifier feature `(N, feat_dim)` and pass it through a learnable linear
projection `W: feat_dim -> teacher_dim`. The teacher is the Path-1 (or
generic) DINOv2 encoder applied to *both hands* per frame; per-clip the
teacher features are pooled over time to a single `(2 * D_teacher,)` vector.
The KD loss is `1 - cos_sim(W(student_pool), teacher_pool).mean()`. Total
loss = `cross_entropy + kd_weight * kd_loss`. No teacher gradients; teacher
features are precomputed and frozen.

## Run commands

### Step P2.1 — extract teacher features (once)

If you haven't already produced Bangla-DINOv2 features via Path 1, you can
either:

(a) **Use Path 1's encoder (preferred)**: run `path1_bangla_dinov2/extract_features.py`
    to produce `data/bdsl_si_bdino/{train,val,test}_data.npy`.

(b) **Fall back to generic DINOv2**: run Option B's stock extractor — output
    will be at `data/bdsl_si_dino/`. Edit `configs/train_kd.yaml` to point
    `teacher_data_path` at `data/bdsl_si_dino/...` instead.

### Step P2.2 — KD training (× 3 seeds for the headline number)

```bash
for seed in 0 1 2; do
  python -m path2_handshape_kd.train_kd \
      --config path2_handshape_kd/configs/train_kd.yaml --seed $seed
done
```

Wall-clock estimate per seed on Quadro RTX 8000: **~1.5–2 h** (essentially
the same as plain BlockGCN — KD adds a single linear layer + cosine loss).

Output rows in `results_final.csv` will be tagged
`bdsl_block_gcn_kd_si_seed<N>`. The aggregator picks them up automatically:

```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/path2_table.md
```

### Step P2.3 — comparison row in the master table

Add to the headline Stage A table (or treat as a Stage E row):

| Model | SI Top-1 | Notes |
|---|---:|---|
| BlockGCN | 76.95% (pilot, seed 0) | from-scratch baseline |
| **BlockGCN + KD (generic DINOv2 teacher)** | TBD | sanity: does KD help with off-the-shelf teacher? |
| **BlockGCN + KD (Bangla-DINOv2 teacher, Path 1)** | TBD | the headline Path 2 result |

The interesting comparison is the gap between the two KD rows: it isolates
the contribution of *Bangla-domain adaptation of the teacher*, independent
of distillation per se.

## Why Path 2 strengthens the project's identity-shortcut story

Three orthogonal interventions on the SAME identity-shortcut problem:

| Track | How it counters the identity shortcut |
|---|---|
| Option B | swap the input representation (pose → DINOv2 features) |
| Path 1   | adapt the input encoder to the domain |
| **Path 2** | **add a handshape-aware regulariser to the student's representation** |
| Option C | pretrain the student on unlabeled Bangla pose |

If any subset of these win, that's a paper. If they stack, that's a stronger
paper.

## Hyperparameters worth ablating later

* `kd_weight ∈ {0.1, 0.5, 1.0, 2.0}` — strength vs CE
* `kd_loss ∈ {cosine, mse}` — cosine usually wins for representation matching
* `teacher_region_indices = [0, 1] vs [0, 1, 2]` — does adding face help?
* `teacher = generic DINOv2 vs Path 1 Bangla-DINOv2` — the paper-defining ablation
