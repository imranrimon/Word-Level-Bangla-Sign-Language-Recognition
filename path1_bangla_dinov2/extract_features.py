"""Apply a LoRA-adapted DINOv2 encoder to BdSLW60 hand crops.

Mirrors `preprocessing/extract_dinov2_features.py` but loads the LoRA-tuned
backbone produced by `path1_bangla_dinov2/train.py`. Output NPYs share the
shape `(N, 384, T_max, V=3, M=1)` — drop-in replacement for
`data/bdsl_si_dino/` so downstream FlatTemporal training is unchanged.

Usage:
    python -m path1_bangla_dinov2.extract_features \
        --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
        --output-dir data/bdsl_si_bdino \
        --cache-dir data/bdsl_bdino_cache \
        --encoder-checkpoint work_dir/bdino_lora/encoder_epoch10.pt \
        --splits train val test --device cuda --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.dinov2_lora import build_dinov2_lora


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--encoder-checkpoint", required=True)
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test"])
    ap.add_argument("--timm-name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Build the same MultiHeadLoRADinov2 architecture used during training and
    # load the *backbone* state dict. We only need the encoder for feature
    # extraction; the multi-source heads are dropped.
    model = build_dinov2_lora(
        num_classes_per_source=[1],   # placeholder, head is unused
        timm_name=args.timm_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        pretrained=True,
    )
    state = torch.load(args.encoder_checkpoint, map_location="cpu")
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"loaded encoder from {args.encoder_checkpoint}")
    if missing:
        print(f"  missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    backbone = model.backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Reuse the upstream extractor's MediaPipe + crop pipeline by patching the
    # DINOv2 it loads with our backbone.
    import preprocessing.extract_dinov2_features as ext

    # Monkey-patch the loader to return our backbone instead of a fresh one.
    feat_dim = int(getattr(backbone, "num_features", 384))

    def _patched_loader(model_name, dev):
        return backbone.to(dev), feat_dim

    ext._load_dinov2 = _patched_loader

    # Now invoke the extractor's main() with our args reformatted.
    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    sys_argv_backup = sys.argv
    new_argv = [
        "extract_dinov2_features.py",
        "--dataset-root", args.dataset_root,
        "--output-dir", args.output_dir,
        "--splits", *args.splits,
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--max-frames", str(args.max_frames),
        "--model", args.timm_name,
    ]
    if args.cache_dir:
        new_argv += ["--cache-dir", args.cache_dir]
    if args.dry_run:
        new_argv += ["--dry-run"]
    sys.argv = new_argv
    try:
        ext.main()
    finally:
        sys.argv = sys_argv_backup


if __name__ == "__main__":
    main()
