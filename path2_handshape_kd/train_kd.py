"""Path 2 entry point — train BlockGCN with handshape-teacher KD.

Pose is loaded from `data/bdsl_si/` as usual. Teacher features are loaded
from a precomputed NPY (from Path 1's `extract_features.py` or Option B's
stock DINOv2 extractor). Distillation is **pooled cosine-similarity**: the
student's pre-classifier pooled feature is projected to the teacher's
dimensionality, and a `1 - cos_sim` loss is added to cross-entropy.

Usage:
    python -m path2_handshape_kd.train_kd \
        --config path2_handshape_kd/configs/train_kd.yaml --seed 0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from feeders.kd_feeder import KDFeeder
from model.block_gcn_kd import Model as KDModel

RESULT_COLUMNS = ["Timestamp", "Experiment", "Epoch", "Top1_Acc",
                  "Top5_Acc", "Top5_Policy", "WorkDir"]


def _init_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _kd_loss(student_proj, teacher_vec, kind="cosine"):
    """Pooled KD loss between (N, D) student and (N, D) teacher pooled features.

    Modes:
      cosine          - 1 - cos_sim(student, teacher); magnitude-invariant
                        (the default; safe regardless of teacher norm).
      mse             - raw MSE; SENSITIVE to per-dim scale differences between
                        student_proj output and teacher feature distribution.
                        Use only after verifying teacher feats are unit-norm.
      mse_normalized  - L2-normalize both then MSE; equivalent to a scaled
                        cosine and safer than raw MSE (audit fix #11).
    """
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(student_proj, teacher_vec, dim=-1).mean()
    if kind == "mse":
        return F.mse_loss(student_proj, teacher_vec)
    if kind == "mse_normalized":
        s = F.normalize(student_proj, dim=-1)
        t = F.normalize(teacher_vec, dim=-1)
        return F.mse_loss(s, t)
    raise ValueError(f"unknown KD loss kind: {kind!r} "
                     "(must be 'cosine', 'mse', or 'mse_normalized')")


def _append_csv_row(csv_path, row):
    new_file = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    for pose, _teacher, label in loader:
        pose = pose.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)
        logits, _ = model(pose)
        topk = logits.topk(5, dim=-1).indices
        total += label.size(0)
        correct1 += (topk[:, 0] == label).sum().item()
        correct5 += (topk == label.unsqueeze(-1)).any(dim=-1).sum().item()
    return correct1 / max(1, total), correct5 / max(1, total)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-csv", default="results_final.csv")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_exp = cfg.get("Experiment_name", "bdsl_block_gcn_kd_si")
    exp_name = f"{base_exp}_seed{args.seed}"
    work_dir = os.path.join(".", "work_dir", exp_name)
    os.makedirs(work_dir, exist_ok=True)
    _init_seed(args.seed)

    # ---- Feeders ----
    tf = cfg["train_feeder_args"]
    vf = cfg["test_feeder_args"]
    train_feeder = KDFeeder(
        pose_data_path=tf["pose_data_path"],
        pose_label_path=tf["pose_label_path"],
        teacher_data_path=tf["teacher_data_path"],
        teacher_label_path=tf.get("teacher_label_path"),
        random_choose=tf.get("random_choose", True),
        random_shift=tf.get("random_shift", True),
        window_size=int(tf.get("window_size", 120)),
        normalization=tf.get("normalization", True),
        random_mirror=tf.get("random_mirror", False),
        random_mirror_p=float(tf.get("random_mirror_p", 0.0)),
        is_vector=tf.get("is_vector", False),
        lap_pe=tf.get("lap_pe", False),
        teacher_region_indices=tuple(tf.get("teacher_region_indices", (0, 1))),
    )
    val_feeder = KDFeeder(
        pose_data_path=vf["pose_data_path"],
        pose_label_path=vf["pose_label_path"],
        teacher_data_path=vf["teacher_data_path"],
        teacher_label_path=vf.get("teacher_label_path"),
        random_choose=False, random_shift=False,
        window_size=int(vf.get("window_size", 120)),
        normalization=vf.get("normalization", True),
        random_mirror=False, is_vector=vf.get("is_vector", False),
        lap_pe=vf.get("lap_pe", False),
        teacher_region_indices=tuple(vf.get("teacher_region_indices", (0, 1))),
    )
    print(f"train feeder: {len(train_feeder)} samples;  val feeder: {len(val_feeder)} samples")

    train_loader = DataLoader(
        train_feeder, batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True, num_workers=int(cfg.get("num_worker", 0)),
        drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_feeder, batch_size=int(cfg.get("test_batch_size", 32)),
        shuffle=False, num_workers=int(cfg.get("num_worker", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Model ----
    margs = dict(cfg["model_args"])
    teacher_dim = int(margs.get("teacher_dim", 768))
    model = KDModel(**margs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"KD model params: {n_train/1e6:.2f}M trainable, teacher_dim={teacher_dim}")

    # ---- Optim ----
    base_lr = float(cfg.get("base_lr", 0.1))
    opt_name = cfg.get("optimizer", "SGD")
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=base_lr, momentum=0.9,
            nesterov=bool(cfg.get("nesterov", True)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr,
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )

    step_epochs = list(cfg.get("step", [60, 80]))
    warmup_epoch = int(cfg.get("warm_up_epoch", 5))
    num_epoch = int(cfg.get("num_epoch", 100))
    eval_interval = int(cfg.get("eval_interval", 5))
    save_interval = int(cfg.get("save_interval", 10))
    kd_weight = float(cfg.get("kd_weight", 1.0))
    kd_kind = cfg.get("kd_loss", "cosine")

    best_top1 = 0.0
    _printed_teacher_stats = False
    for epoch in range(num_epoch):
        # LR schedule (warmup -> step decay)
        if epoch < warmup_epoch:
            lr = base_lr * (epoch + 1) / max(1, warmup_epoch)
        else:
            decay = sum(1 for s in step_epochs if epoch >= s)
            lr = base_lr * (0.1 ** decay)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        t0 = time.time()
        running_ce = running_kd = 0.0
        n_batches = 0
        for step, (pose, teacher, label) in enumerate(train_loader):
            pose = pose.float().to(device, non_blocking=True)
            teacher = teacher.float().to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)
            # One-time diagnostic on the very first batch (audit fix #11):
            # if teacher feats are wildly out-of-scale, raw MSE would blow up.
            if not _printed_teacher_stats:
                with torch.no_grad():
                    t_norm = teacher.norm(dim=-1)
                    print(f"[diag] teacher feats: shape={tuple(teacher.shape)} "
                          f"per-sample L2-norm min={t_norm.min().item():.3f} "
                          f"max={t_norm.max().item():.3f} "
                          f"mean={t_norm.mean().item():.3f}; "
                          f"per-dim mean={teacher.mean().item():+.4f} "
                          f"std={teacher.std().item():.4f}. "
                          f"kd_loss kind={kd_kind!r} weight={kd_weight}.")
                _printed_teacher_stats = True
            logits, student_proj = model(pose)
            ce = F.cross_entropy(logits, label)
            kd = _kd_loss(student_proj, teacher, kind=kd_kind)
            loss = ce + kd_weight * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_ce += float(ce.item())
            running_kd += float(kd.item())
            n_batches += 1
        print(f"epoch {epoch+1}/{num_epoch}  lr={lr:.4f}  "
              f"ce={running_ce/max(1,n_batches):.4f}  "
              f"kd={running_kd/max(1,n_batches):.4f}  "
              f"({time.time()-t0:.1f}s)")

        do_eval = ((epoch + 1) % eval_interval == 0) or (epoch + 1 == num_epoch)
        if do_eval:
            top1, top5 = _evaluate(model, val_loader, device)
            print(f"  eval@{epoch+1}: Top1={top1*100:.2f}%  Top5={top5*100:.2f}%")
            if top1 > best_top1:
                best_top1 = top1
                ckpt = os.path.join(work_dir, f"{exp_name}_best.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"  saved best -> {ckpt}")
            _append_csv_row(args.results_csv, {
                "Timestamp": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Experiment": exp_name,
                "Epoch": epoch + 1,
                "Top1_Acc": f"{top1:.6f}",
                "Top5_Acc": f"{top5:.6f}",
                "Top5_Policy": "same_epoch_as_logged_top1",
                "WorkDir": work_dir,
            })

        if (epoch + 1) % save_interval == 0:
            ckpt = os.path.join(work_dir, f"{exp_name}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)


if __name__ == "__main__":
    main()
