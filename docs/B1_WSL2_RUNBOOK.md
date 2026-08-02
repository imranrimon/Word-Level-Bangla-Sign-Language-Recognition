# B1 on your local Windows machine, via WSL2 — copy-paste runbook

Goal: run the two mandatory foundation-model baselines (**Uni-Sign** pose finetune,
**SHuBERT** frozen probe) on **this Windows workstation**, inside WSL2 (Ubuntu),
because Uni-Sign's trainer uses **deepspeed**, which is unreliable on native
Windows. The same RTX 8000 is used through NVIDIA's CUDA-on-WSL2 driver — no
second machine, no cloud. For the science details (why these rows, the 3-file
patch, eval convention) see [`B1_FOUNDATION_BASELINES.md`](B1_FOUNDATION_BASELINES.md);
this file is the WSL2 operations layer.

> **GPU contention:** WSL2 shares the one physical GPU with the Windows training
> program. Run B1 **after** the SI sweep + Option C free the GPU (or when you can
> spare it). Check `nvidia-smi` first.

---

## 0. One-time: install WSL2 + Ubuntu (Windows side, ~15 min + reboot)

In an **elevated** PowerShell (Run as Administrator):

```powershell
wsl --install -d Ubuntu-22.04
# reboot when prompted, then set a UNIX username/password at first launch
```

Verify the GPU is visible inside WSL2 (uses the Windows driver — do NOT install a
Linux display driver inside WSL2):

```bash
# inside the Ubuntu (WSL2) shell:
nvidia-smi          # should list the Quadro RTX 8000
```

Your Windows `F:\SLGTformer` is visible at **`/mnt/f/SLGTformer`** from WSL2.
`results_final.csv` there is shared, so B1 rows you append land in the same file
the rest of the pipeline uses.

---

## 1. One-time: Miniconda + Uni-Sign env (inside WSL2)

```bash
# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/mc.sh
bash ~/mc.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash && exec bash

# Work on the WSL2 filesystem (fast I/O); copy the cloned repo in from Windows:
mkdir -p ~/b1 && cp -r /mnt/f/SLGTformer/external/Uni-Sign ~/b1/Uni-Sign
cd ~/b1/Uni-Sign

conda create --name Uni-Sign python=3.9 -y && conda activate Uni-Sign
pip install -r requirements.txt
pip install onnxruntime-gpu
pip install -e ./demo/rtmlib-main
# sanity: torch sees CUDA through WSL2
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. One-time: HF login + downloads (inside WSL2)

WSL2 has its own home, so authenticate here too (your Windows HF login does not
carry over). Paste your token when prompted — **and rotate it afterward** since it
was shared in chat:

```bash
huggingface-cli login          # paste HF token; do NOT hard-code it in any file

# mT5-base (~2.3 GB) and the Uni-Sign stage-1 (pose-only) checkpoint:
huggingface-cli download google/mt5-base --local-dir pretrained_weight/mt5-base
huggingface-cli download ZechengLi19/Uni-Sign --local-dir hf_ckpts
# locate the stage-1 pose-only best_checkpoint.pth inside hf_ckpts/ (browse the
# card at https://huggingface.co/ZechengLi19/Uni-Sign) and note its path.
```

## 3. Labels + RTMPose pose extraction (once, GPU, ~3–6 h)

The BdSLW60 SI label files are already generated in the repo, so just copy them in
(regenerate only if missing — see B1_FOUNDATION_BASELINES.md §Status):

```bash
mkdir -p data/BdSLW60
cp /mnt/f/SLGTformer/external/Uni-Sign/data/BdSLW60/labels.{train,dev,test} data/BdSLW60/
```

Extract RTMPose COCO-WholeBody keypoints (Uni-Sign's own pose modality — **not**
our MediaPipe-27 cache; keep them separate). Read videos straight from Windows:

```bash
BDSL=/mnt/f/SLGTformer/Word_level_Bangla_Sign_Language_Dataset/BdSLW30
for d in "$BDSL"/*/; do
  cls=$(basename "$d")
  python ./demo/pose_extraction.py --src_dir "$d" \
      --tgt_dir "./dataset/BdSLW60/pose_format/$cls"
done
```

## 4. Three-file patch to register the dataset

Apply the `config.py` / `datasets.py` / `utils.py` edits from
[`B1_FOUNDATION_BASELINES.md` §1d](B1_FOUNDATION_BASELINES.md). Set
`rgb_dirs["BdSLW60"] = "/mnt/f/SLGTformer/Word_level_Bangla_Sign_Language_Dataset/BdSLW30"`
and `pose_dirs["BdSLW60"] = "./dataset/BdSLW60/pose_format"`.

## 5. Finetune + the two extra cells (GPU, ~6–12 h + ~6–12 h)

```bash
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 8 --gradient-accumulation-steps 1 --epochs 20 \
  --opt AdamW --lr 3e-4 --output_dir out/bdslw60_si_ft \
  --finetune <path to stage1 pose-only best_checkpoint.pth> \
  --dataset BdSLW60 --task ISLR --max_length 64
```

For NeurIPS/ICLR credit-separation, also run (cheap, same command):
- `--eval` with `--finetune <ckpt>` but no training → **zero-shot** (documents that
  vocabulary transfer needs finetuning);
- omit `--finetune` → **from-scratch** (separates architecture from pretraining).

Record the ISLR per-instance/per-class Top-1 into the shared CSV
(`/mnt/f/SLGTformer/results_final.csv`) as rows `unisign_pose_bdsl_si` /
`unisign_pose_bdsl_si_testset` (Uni-Sign doesn't write our CSV).

## 6. SHuBERT frozen probe (second priority, ~8–16 h)

Follow [`B1_FOUNDATION_BASELINES.md` §2](B1_FOUNDATION_BASELINES.md): run their
4-stream feature pipeline over BdSLW60, freeze SHuBERT, train a linear head on the
canonical SI split, log `shubert_frozen_bdsl_si`. This is the direct ASL→BdSL SSL
transfer row — the published counterpart to our cross-lingual masked-pose-SSL
claim, so it must be in the table regardless of score.

---

## Why B1 is on the critical path for NeurIPS/ICLR
A pose-SLR paper at this tier will be rejected without a comparison to current pose
SOTA. Uni-Sign (ICLR'25) and SHuBERT (ACL'25) ARE that SOTA, and they double as the
**supervised** (CSL→BdSL) and **SSL** (ASL→BdSL) cross-lingual transfer baselines
that our cross-lingual masked-pose-SSL method must beat or match. See
[`TOPTIER_NEURIPS_ICLR_PLAN.md`](TOPTIER_NEURIPS_ICLR_PLAN.md).
