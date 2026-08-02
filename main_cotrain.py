#!/usr/bin/env python
"""Joint co-training entry point: target (BdSLW60-SI) + auxiliary (BdSLW401) heads.

The B3 ablation companion to the BPT recipe. BPT trains sequentially
(supervised 401-way pretrain -> head swap -> low-LR 60-way finetune);
this script trains one shared backbone jointly with two classification
heads, following the Logos (EMNLP 2025) finding that multi-head joint
co-training can beat sequential transfer for low-resource targets.

Usage:
    python main_cotrain.py --config config/bdsl_block_gcn_cotrain_si.yaml
    python main_cotrain.py --config config/bdsl_block_gcn_cotrain_si_smoke.yaml

Required YAML keys (see config/bdsl_block_gcn_cotrain_si.yaml):

    Experiment_name:     bdsl_block_gcn_cotrain_si
    model:               model.block_gcn_multihead.Model
    model_args:          { num_class: 60, num_class_aux: 401, ... }
    target_train_feeder_args / target_val_feeder_args /
    target_test_feeder_args:   feeders.feeder.Feeder kwargs for BdSLW60-SI
    aux_train_feeder_args:     feeders.feeder.Feeder kwargs for BdSLW401
    aux_loss_weight:     1.0      # lambda on the auxiliary CE term
    aux_fraction:        1.0      # fraction of the aux train set sampled per epoch
    base_lr / step / weight_decay / batch_size / num_epoch / nesterov /
    warm_up_epoch / optimizer / device / seed / results_csv

Conventions mirrored from main.py so downstream tooling keeps working:
  * work_dir = ./work_dir/<Experiment_name>; resolved config copied there.
  * Every new best *val* Top-1 appends a row to results_final.csv
    (Experiment = <Experiment_name>, Top5 from the same epoch).
  * After training, the best-val checkpoint is evaluated once on the
    target *test* split and logged as Experiment = <Experiment_name>_testset.
  * Checkpoints: <work_dir>/<name>_model_best.pt (full two-head model) and
    <work_dir>/<name>_backbone_best.pt (backbone-only, loadable into
    model.block_gcn.Model via main.py --weights ... --ignore-weights fc).
"""

from __future__ import annotations

import argparse
import datetime
import os

# Same Windows MKL/OpenMP duplicate-runtime workaround the test suite uses;
# must be set before numpy/torch touch LAPACK (graph construction crashes
# otherwise under the base conda env).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from feeders.feeder import Feeder  # noqa: E402
from main import append_result_row, TOP5_POLICY, import_class, init_seed  # noqa: E402


class _DomainTagged(Dataset):
    """Wrap a Feeder so batches carry a domain id (0 = target, 1 = aux)."""

    def __init__(self, base, domain):
        self.base = base
        self.domain = domain

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        data, label, _ = self.base[index]
        return data, label, self.domain


class _CotrainConcat(Dataset):
    """Concat of target + aux with per-epoch resampling of the aux subset."""

    def __init__(self, target, aux, aux_fraction, seed):
        self.target = _DomainTagged(target, 0)
        self.aux = _DomainTagged(aux, 1)
        self.aux_fraction = float(aux_fraction)
        self._rng = np.random.RandomState(seed)
        self.resample()

    def resample(self):
        n_aux = int(round(self.aux_fraction * len(self.aux)))
        n_aux = max(0, min(n_aux, len(self.aux)))
        self.aux_indices = self._rng.choice(len(self.aux), size=n_aux, replace=False)

    def __len__(self):
        return len(self.target) + len(self.aux_indices)

    def __getitem__(self, index):
        if index < len(self.target):
            return self.target[index]
        return self.aux[int(self.aux_indices[index - len(self.target)])]


def get_parser():
    p = argparse.ArgumentParser(description="Joint co-training (target + aux heads)")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None, help="override YAML seed")
    p.add_argument("--num_epoch", type=int, default=None, help="override YAML num_epoch")
    p.add_argument("-Experiment_name", default=None, help="override YAML Experiment_name")
    return p


def adjust_learning_rate(optimizer, epoch, base_lr, step, warm_up_epoch):
    # Same schedule family as main.py: linear warm-up then step decay.
    if epoch < warm_up_epoch:
        lr = base_lr * (epoch + 1) / warm_up_epoch
    else:
        lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def topk_from_scores(scores, labels, k):
    rank = scores.argsort(axis=1)
    hit = [labels[i] in rank[i, -k:] for i in range(len(labels))]
    return sum(hit) / max(1, len(hit))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for data, label, _ in loader:
        data = data.float().to(device)
        logits_t, _ = model(data)
        all_scores.append(logits_t.cpu().numpy())
        all_labels.extend(label.numpy().tolist())
    scores = np.concatenate(all_scores, axis=0)
    labels = np.array(all_labels)
    return scores, labels, topk_from_scores(scores, labels, 1), topk_from_scores(scores, labels, 5)


def main():
    args = get_parser().parse_args()
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

    # ----- data ---------------------------------------------------------
    target_train = Feeder(**cfg["target_train_feeder_args"])
    target_val = Feeder(**cfg["target_val_feeder_args"])
    target_test = Feeder(**cfg["target_test_feeder_args"])
    aux_train = Feeder(**cfg["aux_train_feeder_args"])

    train_set = _CotrainConcat(
        target_train, aux_train,
        aux_fraction=cfg.get("aux_fraction", 1.0), seed=seed,
    )
    num_worker = int(cfg.get("num_worker", 0))
    batch_size = int(cfg.get("batch_size", 32))
    test_batch_size = int(cfg.get("test_batch_size", batch_size))

    def make_train_loader():
        return DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_worker, drop_last=True, worker_init_fn=init_seed,
        )

    val_loader = DataLoader(target_val, batch_size=test_batch_size, shuffle=False,
                            num_workers=num_worker)
    test_loader = DataLoader(target_test, batch_size=test_batch_size, shuffle=False,
                             num_workers=num_worker)

    # ----- model / optim -------------------------------------------------
    Model = import_class(cfg["model"])
    model = Model(**cfg.get("model_args", {})).to(device)

    base_lr = float(cfg.get("base_lr", 0.1))
    optimizer_name = str(cfg.get("optimizer", "SGD")).upper()
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=base_lr, momentum=0.9,
            nesterov=bool(cfg.get("nesterov", True)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
    elif optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=base_lr,
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    ce = nn.CrossEntropyLoss()
    aux_w = float(cfg.get("aux_loss_weight", 1.0))
    num_epoch = int(cfg.get("num_epoch", 100))
    step = cfg.get("step", [60, 80])
    warm_up = int(cfg.get("warm_up_epoch", 0))
    csv_path = cfg.get("results_csv", "results_final.csv")

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cotrain] {name}: {n_param/1e6:.2f} M params | device={device} | "
          f"target={len(target_train)} aux/epoch={len(train_set)-len(target_train)} "
          f"(fraction={cfg.get('aux_fraction', 1.0)}) | aux_loss_weight={aux_w}")

    best_val_top1 = 0.0
    best_epoch = -1
    model_path = os.path.join(work_dir, f"{name}_model_best.pt")
    backbone_path = os.path.join(work_dir, f"{name}_backbone_best.pt")

    for epoch in range(num_epoch):
        train_set.resample()
        loader = make_train_loader()
        lr = adjust_learning_rate(optimizer, epoch, base_lr, step, warm_up)

        model.train()
        t0 = time.time()
        losses, losses_t, losses_a = [], [], []
        for data, label, domain in loader:
            data = data.float().to(device)
            label = label.long().to(device)
            domain = domain.to(device)

            logits_t, logits_a = model(data)
            mask_t = domain == 0
            mask_a = domain == 1

            loss = data.new_zeros(())
            if mask_t.any():
                lt = ce(logits_t[mask_t], label[mask_t])
                loss = loss + lt
                losses_t.append(lt.item())
            if mask_a.any():
                la = ce(logits_a[mask_a], label[mask_a])
                loss = loss + aux_w * la
                losses_a.append(la.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scores, labels, top1, top5 = evaluate(model, val_loader, device)
        print(f"[cotrain] epoch {epoch:3d} | lr {lr:.5f} | "
              f"loss {np.mean(losses):.4f} (t {np.mean(losses_t) if losses_t else 0:.4f} / "
              f"a {np.mean(losses_a) if losses_a else 0:.4f}) | "
              f"val top1 {top1:.4f} top5 {top5:.4f} | {time.time()-t0:.1f}s")

        if top1 > best_val_top1:
            best_val_top1, best_epoch = top1, epoch
            torch.save(model.state_dict(), model_path)
            torch.save(model.backbone.state_dict(), backbone_path)
            with open(os.path.join(work_dir, "eval_results", "best_acc.pkl"), "wb") as f:
                pickle.dump(dict(zip(target_val.sample_name, scores)), f)
            append_result_row(csv_path, {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Experiment": name,
                "Epoch": epoch,
                "Top1_Acc": f"{top1:.4f}",
                "Top5_Acc": f"{top5:.4f}",
                "Top5_Policy": TOP5_POLICY,
                "WorkDir": work_dir,
            })

    # ----- final test-set evaluation with the best-val checkpoint --------
    if best_epoch >= 0:
        model.load_state_dict(torch.load(model_path, map_location=device))
    scores, labels, top1, top5 = evaluate(model, test_loader, device)
    with open(os.path.join(work_dir, "eval_results", "testset_scores.pkl"), "wb") as f:
        pickle.dump(dict(zip(target_test.sample_name, scores)), f)
    append_result_row(csv_path, {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Experiment": f"{name}_testset",
        "Epoch": best_epoch,
        "Top1_Acc": f"{top1:.4f}",
        "Top5_Acc": f"{top5:.4f}",
        "Top5_Policy": TOP5_POLICY,
        "WorkDir": work_dir,
    })
    print(f"[cotrain] DONE. best val top1 {best_val_top1:.4f} @ epoch {best_epoch} | "
          f"TEST top1 {top1:.4f} top5 {top5:.4f} | checkpoints: {model_path}")


if __name__ == "__main__":
    main()
