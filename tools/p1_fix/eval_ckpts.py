#!/usr/bin/env python
"""Evaluate fine-tuned checkpoints on an arbitrary split WITHOUT touching results_final.csv.

Builds the model and the (deterministic) test feeder from a training config, overrides the
evaluation data/label paths, loads each checkpoint's `weights`, and writes Top-1/Top-5 per
checkpoint to a JSON file (one entry per checkpoint path). Resumable: entries already in the
output JSON are skipped. Any I/O error stops the run (never silently skipped).

  python tools/p1_fix/eval_ckpts.py --config config/karsl_si_lr01_f10.yaml \
      --data data/karsl_si/test_data.npy --labels data/karsl_si/test_label.pkl \
      --ckpts work_dir/karsl_si_lr01_f10_pool_seed0/karsl_si_lr01_f10_pool_seed0_model-best.pt ... \
      --out docs/p1_fix_2026-10/step3/karsl_u03_eval.json
"""
import argparse, importlib, json, os, sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def imp(s):
    m, c = s.rsplit(".", 1)
    return getattr(importlib.import_module(m), c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targs = dict(cfg["test_feeder_args"]); targs["data_path"] = a.data; targs["label_path"] = a.labels
    ds = imp(cfg["feeder"])(**targs)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=4)
    model = imp(cfg["model"])(**cfg["model_args"]).to(dev).eval()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for ck in a.ckpts:
        if ck in res:
            continue
        if not os.path.exists(ck):
            print(f"[missing] {ck}", flush=True); continue
        st = torch.load(ck, map_location="cpu")
        w = st["weights"] if "weights" in st else st
        model.load_state_dict({k.split("module.")[-1]: v for k, v in w.items()})
        scores = []
        with torch.no_grad():
            for x, y, idx in dl:
                scores.append(model(x.float().to(dev)).cpu().numpy())
        sc = np.concatenate(scores)
        lab = np.asarray(ds.label)
        rank = np.argsort(-sc, 1)
        t1 = float((rank[:, 0] == lab).mean()); t5 = float((rank[:, :5] == lab[:, None]).any(1).mean())
        res[ck] = {"top1": t1, "top5": t5, "n": int(len(lab)), "ckpt_epoch": int(st.get("epoch", -1)) if isinstance(st, dict) else -1,
                   "ckpt_val_acc": float(st.get("acc", float("nan"))) if isinstance(st, dict) else float("nan")}
        print(f"{ck}: top1 {t1:.4f} top5 {t5:.4f} (val-selected acc {res[ck]['ckpt_val_acc']:.4f} @ep{res[ck]['ckpt_epoch']})", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
