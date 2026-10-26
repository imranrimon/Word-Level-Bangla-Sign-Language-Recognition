#!/usr/bin/env python
"""SHuBERT-style masked self-supervised pretraining entry point.

Parallels main.py for the classification pipeline but:
  * loads a pose backbone (typically model.block_gcn.Model with
    return_features=True) and wraps it in ShubertPretrainer,
  * uses PretrainFeeder to serve (pose, cluster-ids) pairs,
  * saves *backbone-only* checkpoints so main.py can fine-tune them via
    --weights / --ignore-weights.

Usage:
    python main_pretrain.py --config config/bdsl_shubert_pretrain_xlingual.yaml

Required YAML keys:

    Experiment_name:          bdsl_shubert_pretrain
    backbone:                 model.block_gcn.Model
    backbone_args:            { ..., return_features: True }
    pretrain:
      manifest:               data/ssl_pool_manifest.json
      targets:                data/pretrain_kmeans_targets.npz
      num_codes:              64
      mask_ratio:             0.3
      window_size:            120
      feat_dim:               256
    base_lr:                  1.0e-4
    weight_decay:             1.0e-4
    batch_size:               32
    num_epoch:                30
    save_interval:            5
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from feeders.pretrain_feeder import PretrainFeeder  # noqa: E402
from model.shubert_pretrain import ShubertPretrainer  # noqa: E402


def _import_class(dotted):
    mod_name, _, cls_name = dotted.rpartition(".")
    return getattr(importlib.import_module(mod_name), cls_name)


class _TimeFeatureBackbone(nn.Module):
    """Adapter: make a pose Model emit per-time features (N, T, feat_dim).

    Two paths:
      * Native (SLGTFormer / model.twin_attention.Model only): its stage layers
        take a ``keep_prob`` arg and downsample T, so the generic loop below
        can't replay them. Built with ``return_features=True`` +
        ``stride_between_stages=False`` its own forward already yields per-frame
        features (N, T, feat_dim) with T_out==T_in, so we just call it.
      * Generic (BlockGCN / ST-GCN / adaptive_gnn): replay the backbone's
        data_bn + single-arg layer stack manually, pooling over V,M instead of
        V,M,T (their native return_features pools over T, losing the time axis).
        This mirrors tests/test_option_c_smoke.py.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = backbone.feat_dim
        # Only SLGTFormer needs its own forward (keep_prob arg + no single-arg
        # per-time return path); every other backbone uses the generic replay.
        self._native = type(backbone).__module__.endswith("twin_attention")

    def forward(self, x):
        if self._native:
            # keep_prob=1.0 -> no DropBlock/DropT during SSL feature extraction (deterministic).
            return self.backbone(x, 1.0)
        N, C, T, V, M = x.shape
        bb = self.backbone
        y = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        y = bb.data_bn(y)
        y = y.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for layer in bb.layers:
            y = layer(y)
        y = y.mean(dim=-1)                                    # (N*M, Cf, T')
        y = y.view(N, M, -1, y.size(-1)).mean(dim=1)           # (N, Cf, T')
        return y.transpose(1, 2).contiguous()                  # (N, T', Cf)


def _init_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_pretrainer(cfg, device):
    """Backbone + ShubertPretrainer exactly as configured (also used by the CPU smoke tests).

    Returns (backbone, pretrainer, use_lang, backbone_kwargs, pretrain_cfg)."""
    # Build backbone
    Backbone = _import_class(cfg["backbone"])
    backbone_kwargs = dict(cfg.get("backbone_args", {}))
    backbone_kwargs.setdefault("return_features", True)
    backbone = Backbone(**backbone_kwargs).to(device)

    # Wrap in SHuBERT pretrainer
    p = cfg["pretrain"]
    adapter = _TimeFeatureBackbone(backbone).to(device)
    # Cross-lingual mechanism knobs (default 0 -> pool-composition-only ablation).
    lambda_adv = float(p.get("lambda_adv", 0.0))
    lambda_contrast = float(p.get("lambda_contrast", 0.0))
    lambda_coral = float(p.get("lambda_coral", 0.0))
    # language ids are needed by: the GRL adversary, the distance gate (which keys
    # per-clip weights on lang), and CORAL (source vs target split). The contrastive
    # term keys on codes not lang, but the gate still needs lang when it is on.
    use_lang = (
        lambda_adv > 0.0
        or lambda_coral > 0.0
        or (p.get("source_distances", None) is not None and lambda_contrast > 0.0)
    )
    pretr = ShubertPretrainer(
        backbone=adapter,
        feat_dim=int(p.get("feat_dim", adapter.feat_dim)),
        num_codes=int(p["num_codes"]),
        mask_ratio=float(p.get("mask_ratio", 0.3)),
        lambda_mask=float(p.get("lambda_mask", 1.0)),
        lambda_latent=float(p.get("lambda_latent", 0.0)),
        in_channels=int(backbone_kwargs.get("in_channels", 3)),
        num_point=int(backbone_kwargs.get("num_point", 27)),
        num_languages=int(p.get("num_languages", 0)),
        lambda_adv=lambda_adv,
        lambda_contrast=lambda_contrast,
        grl_alpha=float(p.get("grl_alpha", 1.0)),
        proj_dim=int(p.get("proj_dim", 128)),
        contrast_temp=float(p.get("contrast_temp", 0.1)),
        code_sim_threshold=float(p.get("code_sim_threshold", 0.5)),
        source_distances=p.get("source_distances", None),
        align_tau=float(p.get("align_tau", 0.1)),
        gate_mode=str(p.get("gate_mode", "hardgate")),
        align_lambda_max=float(p.get("align_lambda_max", 1.0)),
        align_gamma=float(p.get("align_gamma", 1.0)),
        align_d0=(float(p["align_d0"]) if p.get("align_d0", None) is not None else None),
        lambda_coral=float(p.get("lambda_coral", 0.0)),
        target_lang=int(p.get("target_lang", 0)),
    ).to(device)
    return backbone, pretr, use_lang, backbone_kwargs, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=None,
                    help="override work_dir (defaults to ./work_dir/<Experiment_name>)")
    ap.add_argument("--init-weights", default=None,
                    help="external backbone .pt to initialise from (curriculum phase-2, e.g. the DG-XL backbone)")
    args, _unknown = ap.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("Experiment_name", "bdsl_shubert_pretrain")
    work_dir = args.work_dir or os.path.join(".", "work_dir", exp_name)
    os.makedirs(work_dir, exist_ok=True)

    _init_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone, pretr, use_lang, backbone_kwargs, p = build_pretrainer(cfg, device)
    lambda_adv = float(p.get("lambda_adv", 0.0))
    lambda_contrast = float(p.get("lambda_contrast", 0.0))

    # Curriculum init: warm-start the backbone from an external checkpoint (e.g. the
    # DG-XL backbone) so a phase-2 objective refines it. Auto-resume (train_state.pt)
    # takes precedence and overrides this later if a run is being continued.
    if args.init_weights and os.path.exists(args.init_weights):
        sd = torch.load(args.init_weights, map_location=device)
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        print(f"[init] warm-started backbone from {args.init_weights} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")

    if lambda_adv > 0 or lambda_contrast > 0 or float(p.get("lambda_coral", 0.0)) > 0:
        print(f"[x-lingual] adv={lambda_adv} (langs={p.get('num_languages')}) "
              f"contrast={lambda_contrast} coral={p.get('lambda_coral',0.0)} "
              f"gate={p.get('gate_mode','hardgate')} temp={p.get('contrast_temp',0.1)}")

    # Data
    feeder = PretrainFeeder(
        manifest_path=p["manifest"],
        targets_path=p["targets"],
        window_size=int(p.get("window_size", 120)),
        num_channels=int(backbone_kwargs.get("in_channels", 3)),
        num_joints=int(backbone_kwargs.get("num_point", 27)),
        num_person=int(backbone_kwargs.get("num_person", 1)),
        random_crop=True,
        max_clips=p.get("max_clips"),
        return_language=use_lang,
    )
    # Distance-weighted source selection (§29 distw): when the config requests it and
    # the manifest carries non-uniform per-clip weights, draw clips ~ w = exp(-gamma*d).
    # Falls back to plain shuffle for uniform weights, so all other pools are unaffected.
    sampler = None
    if bool(p.get("weighted_sampling", False)):
        import numpy as _np
        w = _np.asarray(feeder.sample_weights, dtype=_np.float64)
        if w.size and w.std() > 0:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(w), num_samples=len(feeder), replacement=True)
            print(f"[distw] WeightedRandomSampler: w in [{w.min():.4f}, {w.max():.4f}] "
                  f"mean {w.mean():.4f} (n={w.size})")
        else:
            print("[distw] weighted_sampling requested but weights are uniform -> shuffle")
    loader = DataLoader(
        feeder,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.get("num_worker", 0)),
        drop_last=True,
    )
    print(f"pretraining pool size: {len(feeder)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        pretr.parameters(),
        lr=float(cfg.get("base_lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    num_epoch = int(cfg.get("num_epoch", 30))
    save_interval = int(cfg.get("save_interval", 5))
    log_every = int(cfg.get("log_every", 50))

    # --- AUTO-RESUME: never restart from scratch on re-fire/requeue ---
    train_state = os.path.join(work_dir, "train_state.pt")
    start_epoch = 0
    if os.path.exists(train_state):
        st = torch.load(train_state, map_location=device)
        pretr.load_state_dict(st["pretr"])
        optimizer.load_state_dict(st["optim"])
        start_epoch = int(st["epoch"]) + 1
        print(f"[resume] loaded {train_state}: continuing from epoch {start_epoch}/{num_epoch}")
    if start_epoch >= num_epoch:
        print(f"[resume] already at epoch {start_epoch} >= {num_epoch}; nothing to do")
        return

    global_step = 0
    for epoch in range(start_epoch, num_epoch):
        pretr.train()
        t_epoch = time.time()
        losses = []
        for batch_idx, batch in enumerate(loader):
            if use_lang:
                x, codes, lang = batch
                lang = lang.to(device, non_blocking=True)
            else:
                x, codes = batch
                lang = None
            x = x.to(device, non_blocking=True)
            codes = codes.to(device, non_blocking=True)
            loss, aux = pretr(x, codes, lang)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            global_step += 1
            if (batch_idx + 1) % log_every == 0:
                comps = " ".join(f"{k}={aux[k]:.3f}" for k in ("mask_loss", "adv_loss", "contrast_loss") if k in aux)
                print(f"epoch {epoch} step {batch_idx+1}/{len(loader)} loss {np.mean(losses[-log_every:]):.4f}  [{comps}]")
        print(f"[epoch {epoch}] mean loss {np.mean(losses):.4f}  duration {time.time()-t_epoch:.1f}s")

        if (epoch + 1) % save_interval == 0 or epoch == num_epoch - 1:
            ckpt = os.path.join(work_dir, f"pretrained_epoch{epoch+1}.pt")
            torch.save(backbone.state_dict(), ckpt)             # backbone-only (for finetune --weights)
            # full training state for auto-resume (atomic write)
            torch.save({"epoch": epoch, "pretr": pretr.state_dict(), "optim": optimizer.state_dict()},
                       train_state + ".tmp")
            os.replace(train_state + ".tmp", train_state)
            print(f"saved {ckpt} + train_state@epoch{epoch}")


if __name__ == "__main__":
    main()
