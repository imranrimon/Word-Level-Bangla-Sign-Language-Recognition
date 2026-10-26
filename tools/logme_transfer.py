#!/usr/bin/env python
"""LogME (You et al. 2021) transferability of a source SSL backbone to a target
language: extract the source backbone's pooled features on the target TRAIN set and
score how well they linearly explain the target labels (Bayesian evidence). Higher
LogME => the source features transfer better => predicts a higher fine-tuning gain.

We compute LogME(src_s backbone features on target t, target t labels) for a
(source,target) pair and append it to data/logme_transfer.json, so it can be
correlated with the observed transfer gain alongside the pose-motion MMD
(tools/transferability_baselines.py, the transferability-estimator table).

    python tools/logme_transfer.py --config config/gsl_si_lr01_pairft.yaml \
        --backbone work_dir/bdsl_shubert_pretrain_src_bdsl/pretrained_epoch30.pt \
        --tag bdsl2gsl
CPU-only LogME; GPU used only for the forward pass. Model/feeder loaded like main.py.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def import_class(name):
    mod, _, cls = name.rpartition('.')
    m = __import__(mod, fromlist=[cls]); return getattr(m, cls)


def logme(F, y):
    """LogME score: F (n,d) features, y (n,) int labels. Higher is better."""
    F = np.asarray(F, np.float64); n, d = F.shape
    u, s, _ = np.linalg.svd(F, full_matrices=False)
    sigma = (s ** 2).reshape(-1, 1)                  # (d,1)
    K = int(y.max()) + 1
    evis = []
    for k in range(K):
        y_ = (y == k).astype(np.float64).reshape(-1, 1)
        x = u.T @ y_                                  # (d,1)
        x2 = x ** 2
        res_fixed = float((y_ ** 2).sum() - x2.sum())
        alpha, beta = 1.0, 1.0
        for _ in range(15):
            t = alpha / beta
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / (sigma + t) ** 2).sum())
            res = float((x2 * (t ** 2) / (sigma + t) ** 2).sum()) + res_fixed
            alpha = gamma / (m2 + 1e-8)
            beta = (n - gamma) / (res + 1e-8)
        evi = (d / 2 * np.log(alpha) + n / 2 * np.log(beta)
               - 0.5 * float(np.sum(np.log(alpha + beta * sigma)))
               - beta / 2 * res - alpha / 2 * m2 - n / 2 * np.log(2 * np.pi)) / n
        evis.append(evi)
    return float(np.mean(evis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--max-clips', type=int, default=4000)
    ap.add_argument('--out', default='data/logme_transfer.json')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ma = dict(cfg['model_args']); ma['return_features'] = True
    Model = import_class(cfg['model'])
    net = Model(**ma).cuda().eval()
    ck = torch.load(a.backbone, map_location='cpu')
    ck = ck.get('model', ck) if isinstance(ck, dict) else ck
    net.load_state_dict({k: v for k, v in ck.items() if not k.startswith('fc.')}, strict=False)

    fa = dict(cfg['train_feeder_args']); fa['random_choose'] = False
    fa['random_shift'] = False; fa['random_mirror'] = False
    Feeder = import_class(cfg['feeder'])
    ds = Feeder(**fa)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)

    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            f = net(x.float().cuda()).cpu().numpy()
            feats.append(f); labs.append(np.asarray(y))
            if sum(len(z) for z in labs) >= a.max_clips:
                break
    F = np.concatenate(feats)[:a.max_clips]
    Y = np.concatenate(labs)[:a.max_clips].astype(int)
    score = logme(F, Y)
    print(f"LogME[{a.tag}] = {score:.4f}  (n={len(Y)}, d={F.shape[1]}, K={int(Y.max())+1})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    db = json.load(open(a.out)) if os.path.exists(a.out) else {}
    db[a.tag] = score
    json.dump(db, open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
