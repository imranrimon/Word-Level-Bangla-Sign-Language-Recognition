# B1 — Foundation-model baselines on BdSLW60-SI

Goal: two rows that anchor the SI benchmark to 2025 SOTA and double as the
*supervised cross-lingual transfer* comparisons:

| Row | What it is | Transfer direction |
|---|---|---|
| Uni-Sign (pose-only) finetune | ICLR 2025 SOTA, CSL-News-pretrained (1,985 h), 69-keypoint GCN + mT5 | Chinese SL → BdSL |
| SHuBERT frozen-feature probe | ACL 2025 oral, 1,000 h ASL masked multi-stream SSL | ASL → BdSL |

Reporting rule: these are *large-scale-pretrained* rows — keep them in a
separate comparison track from the from-scratch pose models (same reason
NLA-SLR's RGB+pose headline is not a pose baseline). The interesting
number is the gap they open (or don't) over BlockGCN-from-scratch and over
our own SSL pretraining.

## Status

| Step | State |
|---|---|
| Uni-Sign repo cloned to `external/Uni-Sign` | ✅ done (2026-07-16) |
| BdSLW60 label files in Uni-Sign format (canonical SI split, 5748/655/1365) | ✅ generated at `external/Uni-Sign/data/BdSLW60/labels.{train,dev,test}` by `preprocessing/build_unisign_bdslw60_labels.py --nested` |
| Uni-Sign env + weights + pose extraction + 3-file patch + finetune | ⬜ steps below |
| SHuBERT probe | ⬜ steps below (second priority) |

## 1. Uni-Sign pose-only finetune (est. 1–2 GPU-days total)

### 1a. Environment (once)

`fine_tuning.py` uses the deepspeed launcher — run on Linux (HPC or WSL);
Windows deepspeed support is poor. Single GPU is fine
(`--include localhost:0`).

```bash
conda create --name Uni-Sign python=3.9 -y
conda activate Uni-Sign
cd external/Uni-Sign
pip install -r requirements.txt
# pose extraction extras (rtmlib is vendored in the repo)
pip install onnxruntime-gpu
pip install -e ./demo/rtmlib-main
```

### 1b. Downloads (once)

```bash
# mT5-base -> external/Uni-Sign/pretrained_weight/mt5-base  (~2.3 GB)
huggingface-cli download google/mt5-base --local-dir pretrained_weight/mt5-base

# Pose-only pretrained checkpoint (stage 1, CSL-News):
# browse https://huggingface.co/ZechengLi19/Uni-Sign and download the
# stage1 (pose-only) best_checkpoint.pth into out/stage1_pretraining/
huggingface-cli download ZechengLi19/Uni-Sign --local-dir hf_ckpts
```

### 1c. RTMPose extraction over BdSLW60 (~hours, GPU, once)

Their extractor processes a flat dir of .mp4; loop per class dir to keep
the `<class>/<file>` layout the label files use:

```bash
# from external/Uni-Sign, with BdSLW30 videos visible (copy or mount)
for d in /path/to/Word_level_Bangla_Sign_Language_Dataset/BdSLW30/*/; do
  cls=$(basename "$d")
  python ./demo/pose_extraction.py --src_dir "$d" \
      --tgt_dir "./dataset/BdSLW60/pose_format/$cls"
done
```

PowerShell equivalent:

```powershell
Get-ChildItem "F:\SLGTformer\Word_level_Bangla_Sign_Language_Dataset\BdSLW30" -Directory | ForEach-Object {
  python ./demo/pose_extraction.py --src_dir $_.FullName --tgt_dir ("./dataset/BdSLW60/pose_format/" + $_.Name)
}
```

Note: this is RTMPose COCO-WholeBody — a *second* pose modality, unrelated
to our MediaPipe-27 caches. Do not mix the two.

### 1d. Three-file patch to register the dataset

`config.py` — add to each dict:

```python
train_label_paths["BdSLW60"] = "./data/BdSLW60/labels.train"
dev_label_paths["BdSLW60"]   = "./data/BdSLW60/labels.dev"
test_label_paths["BdSLW60"]  = "./data/BdSLW60/labels.test"
rgb_dirs["BdSLW60"]  = "/path/to/Word_level_Bangla_Sign_Language_Dataset/BdSLW30"
pose_dirs["BdSLW60"] = "./dataset/BdSLW60/pose_format"
```

`datasets.py` — in the dataset-path resolution chain (~line 419), add before
the final `else: raise NotImplementedError`:

```python
elif "BdSLW60" in self.args.dataset:
    # labels carry <class>/<file> paths; both dirs are the flat roots
    self.pose_dir = pose_dirs[args.dataset]
    self.rgb_dir = rgb_dirs[args.dataset]
```

`utils.py` (~line 526) — add `"BdSLW60"` to the `--dataset` choices list.

### 1e. Finetune + eval (pose-only, ISLR)

```bash
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 8 --gradient-accumulation-steps 1 --epochs 20 \
  --opt AdamW --lr 3e-4 \
  --output_dir out/bdslw60_si_ft \
  --finetune <path to stage1 pose-only best_checkpoint.pth> \
  --dataset BdSLW60 --task ISLR --max_length 64
# (no --rgb_support => pose-only setting)
```

ISLR eval reports per-instance and per-class Top-1 (`islr_performance`) —
the same P-I/P-C convention as the WLASL literature. Log the resulting
numbers manually into `results_final.csv` as `unisign_pose_bdsl_si` /
`unisign_pose_bdsl_si_testset` rows (Uni-Sign does not write our CSV).

Also worth one extra cell each (cheap): `--eval` zero-shot (no finetune,
expected ≈0 — documents that vocabulary transfer requires finetuning) and a
from-scratch run (omit `--finetune`) to separate "architecture" from
"pretraining" credit.

## 2. SHuBERT frozen-feature probe (second priority)

Code: https://github.com/ShesterG/SHuBERT (project page
https://shubert.pals.ttic.edu/, demo space
https://huggingface.co/spaces/ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText —
checkpoint access is via the repo/space; confirm license before use).

Plan (matches the paper's own low-resource recommendation — frozen
layer-weighted features ≈ finetuned):

1. Run their 4-stream feature pipeline (MediaPipe crops → DINOv2 hands/face
   + 14-d body pose → SHuBERT encoder) over BdSLW60 clips. Reuses the same
   crop logic family as our `preprocessing/extract_dinov2_features.py`.
2. Freeze SHuBERT; train a linear head (and optionally a 2-layer
   transformer head) on train-split features under the canonical SI split;
   report val/test Top-1/Top-5 as `shubert_frozen_bdsl_si`.
3. This is the ASL→BdSL *SSL* transfer row — the direct published
   counterpart to our cross-lingual masked-pose-SSL claim, so it must be in
   the table for the SSL paper regardless of how it scores.

## Effort estimate

| Item | GPU time |
|---|---|
| RTMPose extraction (9,307 clips) | ~3–6 h |
| Uni-Sign finetune (20 epochs, batch 8) | ~6–12 h |
| Zero-shot + from-scratch cells | ~6–12 h |
| SHuBERT feature extraction + probes | ~8–16 h |
