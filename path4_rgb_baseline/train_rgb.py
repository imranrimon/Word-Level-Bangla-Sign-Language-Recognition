#!/usr/bin/env python
"""RGB video baseline on BdSLW60-SI (B2 upgrade).

Trains a Kinetics-400-pretrained RGB video classifier on the canonical
signer-independent split — the RGB row reviewers expect next to the
skeleton models (ASL Citizen's NeurIPS D&B baselines were I3D vs ST-GCN).

Architectures (torchvision, official Kinetics-400 weights downloaded on
first use from pytorch.org):
  * s3d          — S3D; NLA-SLR's encoder family. Default.
  * r2plus1d_18  — R(2+1)D-18.
  * mvit_v2_s    — MViTv2-S (the arXiv 2412.11553 recipe family; needs
                   num_frames == 16).
  * i3d          — vendored piergiaj implementation (Apache-2.0, exact ASL
                   Citizen parity). No auto-download: fetch the checkpoint
                   yourself and set `i3d_weights` in the config:
                     curl -L -o path4_rgb_baseline/weights/rgb_imagenet.pt \
                       https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt

Usage:
    python -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si.yaml
    python -m path4_rgb_baseline.train_rgb --config path4_rgb_baseline/configs/s3d_bdsl_si_smoke.yaml

Mirrors main.py conventions: work_dir/<Experiment_name>, resolved-config
copy, results_final.csv rows on every new best val Top-1, final best-val
checkpoint evaluated once on the test split as <Experiment_name>_testset.
"""

from __future__ import annotations

import argparse
import datetime
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main import append_result_row, TOP5_POLICY, init_seed  # noqa: E402
from main_cotrain import adjust_learning_rate, topk_from_scores  # noqa: E402
from path4_rgb_baseline.rgb_dataset import BdSLW60RGBDataset  # noqa: E402


def build_model(model_name, num_classes, pretrained, i3d_weights=None):
    if model_name == "s3d":
        from torchvision.models.video import s3d, S3D_Weights

        model = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None)
        model.classifier[1] = nn.Conv3d(1024, num_classes, kernel_size=1)
        return model, "kinetics"
    if model_name == "r2plus1d_18":
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

        model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, "kinetics"
    if model_name == "mvit_v2_s":
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

        model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None)
        model.head[1] = nn.Linear(model.head[1].in_features, num_classes)
        return model, "kinetics"
    if model_name == "i3d":
        from path4_rgb_baseline.pytorch_i3d import InceptionI3d

        model = InceptionI3d(400, in_channels=3)
        if pretrained:
            if not i3d_weights or not os.path.exists(i3d_weights):
                raise FileNotFoundError(
                    "i3d requires a local checkpoint; see the download command "
                    "in this file's docstring, then set `i3d_weights` in the config."
                )
            model.load_state_dict(torch.load(i3d_weights, map_location="cpu"))
        model.replace_logits(num_classes)
        return model, "pm1"
    raise ValueError(f"unknown model_name {model_name!r}")


@torch.no_grad()
def evaluate(model, loader, device, squeeze_time_logits):
    model.eval()
    scores, labels = [], []
    for clip, label, _ in loader:
        clip = clip.to(device, non_blocking=True)
        logits = model(clip)
        if squeeze_time_logits and logits.dim() > 2:  # vendored I3D: (N, C, T')
            logits = logits.mean(dim=2)
        scores.append(logits.float().cpu().numpy())
        labels.extend(label.numpy().tolist())
    scores = np.concatenate(scores, axis=0)
    labels = np.array(labels)
    return scores, labels, topk_from_scores(scores, labels, 1), topk_from_scores(scores, labels, 5)


def main():
    ap = argparse.ArgumentParser(description="RGB baseline on BdSLW60-SI")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_epoch", type=int, default=None)
    ap.add_argument("-Experiment_name", default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load <name>_model_best.pt and run the "
                         "val + test evaluations (recovers a missing _testset row)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_epoch is not None:
        cfg["num_epoch"] = args.num_epoch
    if args.Experiment_name:
        cfg["Experiment_name"] = args.Experiment_name

    name = cfg["Experiment_name"]
    seed = int(cfg.get("seed", 0))
    init_seed(seed)

    work_dir = os.path.join("work_dir", name)
    os.makedirs(os.path.join(work_dir, "eval_results"), exist_ok=True)
    with open(os.path.join(work_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    dev_ids = cfg.get("device", [0])
    if isinstance(dev_ids, int):
        dev_ids = [dev_ids]
    device = torch.device(f"cuda:{dev_ids[0]}" if torch.cuda.is_available() else "cpu")

    model_name = cfg.get("model_name", "s3d")
    pretrained = bool(cfg.get("pretrained", True))
    model, norm_mode = build_model(
        model_name, int(cfg.get("num_class", 60)), pretrained, cfg.get("i3d_weights")
    )
    model = model.to(device)
    squeeze_time = model_name == "i3d"

    ds_kwargs = dict(
        root=cfg["dataset_root"],
        classes_json=cfg.get("classes_json", "data/bdsl_si/classes.json"),
        num_frames=int(cfg.get("num_frames", 32)),
        size=int(cfg.get("frame_size", 224)),
        resize_short=int(cfg.get("resize_short", 256)),
        normalize=norm_mode,
        max_clips=cfg.get("max_clips"),
    )
    train_set = BdSLW60RGBDataset(split="train", train=True, **ds_kwargs)
    val_set = BdSLW60RGBDataset(split="val", train=False, **ds_kwargs)
    test_set = BdSLW60RGBDataset(split="test", train=False, **ds_kwargs)

    num_worker = int(cfg.get("num_worker", 0))
    batch_size = int(cfg.get("batch_size", 8))
    test_batch_size = int(cfg.get("test_batch_size", batch_size))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_worker, drop_last=True,
                              worker_init_fn=init_seed)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False,
                            num_workers=num_worker)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                             num_workers=num_worker)

    base_lr = float(cfg.get("base_lr", 0.01))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9,
        nesterov=bool(cfg.get("nesterov", True)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    num_epoch = int(cfg.get("num_epoch", 30))
    step = cfg.get("step", [15, 25])
    warm_up = int(cfg.get("warm_up_epoch", 2))
    csv_path = cfg.get("results_csv", "results_final.csv")
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[rgb] {name}: {model_name} ({n_param/1e6:.1f} M params, pretrained={pretrained}) "
          f"| device={device} amp={use_amp} | train={len(train_set)} val={len(val_set)} "
          f"test={len(test_set)} | frames={ds_kwargs['num_frames']}@{ds_kwargs['size']}px")

    best_val_top1, best_epoch = 0.0, -1
    model_path = os.path.join(work_dir, f"{name}_model_best.pt")

    if args.eval_only:
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, _, vtop1, vtop5 = evaluate(model, val_loader, device, squeeze_time)
        print(f"[rgb] eval-only | val top1 {vtop1:.4f} top5 {vtop5:.4f}")
        scores, labels, top1, top5 = evaluate(model, test_loader, device, squeeze_time)
        with open(os.path.join(work_dir, "eval_results", "testset_scores.pkl"), "wb") as f:
            pickle.dump(dict(zip(test_set.sample_name, scores)), f)
        append_result_row(csv_path, {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Experiment": f"{name}_testset",
            "Epoch": -1,
            "Top1_Acc": f"{top1:.4f}",
            "Top5_Acc": f"{top5:.4f}",
            "Top5_Policy": TOP5_POLICY,
            "WorkDir": work_dir,
        })
        print(f"[rgb] eval-only DONE | TEST top1 {top1:.4f} top5 {top5:.4f}")
        return

    for epoch in range(num_epoch):
        lr = adjust_learning_rate(optimizer, epoch, base_lr, step, warm_up)
        model.train()
        t0, losses = time.time(), []
        for clip, label, _ in train_loader:
            clip = clip.to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(clip)
                if squeeze_time and logits.dim() > 2:
                    logits = logits.mean(dim=2)
                loss = ce(logits, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        scores, labels, top1, top5 = evaluate(model, val_loader, device, squeeze_time)
        print(f"[rgb] epoch {epoch:3d} | lr {lr:.5f} | loss {np.mean(losses):.4f} | "
              f"val top1 {top1:.4f} top5 {top5:.4f} | {time.time()-t0:.1f}s")

        if top1 > best_val_top1:
            best_val_top1, best_epoch = top1, epoch
            torch.save(model.state_dict(), model_path)
            with open(os.path.join(work_dir, "eval_results", "best_acc.pkl"), "wb") as f:
                pickle.dump(dict(zip(val_set.sample_name, scores)), f)
            append_result_row(csv_path, {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Experiment": name,
                "Epoch": epoch,
                "Top1_Acc": f"{top1:.4f}",
                "Top5_Acc": f"{top5:.4f}",
                "Top5_Policy": TOP5_POLICY,
                "WorkDir": work_dir,
            })

    if best_epoch >= 0:
        model.load_state_dict(torch.load(model_path, map_location=device))
    scores, labels, top1, top5 = evaluate(model, test_loader, device, squeeze_time)
    with open(os.path.join(work_dir, "eval_results", "testset_scores.pkl"), "wb") as f:
        pickle.dump(dict(zip(test_set.sample_name, scores)), f)
    append_result_row(csv_path, {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Experiment": f"{name}_testset",
        "Epoch": best_epoch,
        "Top1_Acc": f"{top1:.4f}",
        "Top5_Acc": f"{top5:.4f}",
        "Top5_Policy": TOP5_POLICY,
        "WorkDir": work_dir,
    })
    print(f"[rgb] DONE. best val top1 {best_val_top1:.4f} @ epoch {best_epoch} | "
          f"TEST top1 {top1:.4f} top5 {top5:.4f} | checkpoint: {model_path}")


if __name__ == "__main__":
    main()
