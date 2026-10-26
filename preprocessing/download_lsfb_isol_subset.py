#!/usr/bin/env python3
"""Download a law-comparable LSFB-ISOL subset (cleaned MediaPipe poses only).

Selects the top-K most frequent signs, caps clips/class, builds a signer-disjoint
(SI) split over the 99 LSFB signers, and downloads pose+left_hand+right_hand .npy
landmark files directly from the CC-BY UNamur server (no third-party package).

Server scheme (from lsfb-team/lsfb-dataset downloader.py, branch master):
  base = https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol
  poses/{pose,left_hand,right_hand}/{instance_id}.npy
  signer number = int(instance_id[13:16])
Writes data/lsfb_si_meta.csv (id,sign,signer,split) for the converter step.
"""
import os
import numpy as np, pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://lsfb.info.unamur.be/static/datasets/lsfb_v2/isol"
DEST = "data/lsfb_download"
K_SIGNS = 60          # match BdSLW60 scale
CAP_PER_CLASS = 220   # cap clips/class (sampled across signers) to keep download polite
LANDMARKS = ["pose", "left_hand", "right_hand"]   # skip face (27-kp skeleton = 7 body + 10/hand)
WORKERS = 12
SEED = 0

def main():
    df = pd.read_csv(os.path.join(DEST, "instances.csv"))
    df["snum"] = df["id"].str[13:16].astype(int)
    top = df["sign"].value_counts().head(K_SIGNS).index
    sub = df[df["sign"].isin(top)].copy()

    # signer-disjoint split: rank signers by clip count, hold out for val/test
    sig_order = sub["snum"].value_counts().index.tolist()   # busiest first
    rng = np.random.RandomState(SEED)
    signers = sorted(sub["snum"].unique())
    rng.shuffle(signers)
    n = len(signers)
    n_test = max(12, n // 6); n_val = max(6, n // 10)
    test_s = set(signers[:n_test]); val_s = set(signers[n_test:n_test + n_val])
    def split_of(s): return "test" if s in test_s else ("val" if s in val_s else "train")
    sub["split"] = sub["snum"].map(split_of)

    # cap clips/class within each split (keep signer diversity via groupby-sample)
    capped = []
    for (sign, split), g in sub.groupby(["sign", "split"]):
        cap = CAP_PER_CLASS if split == "train" else max(20, CAP_PER_CLASS // 3)
        capped.append(g.sample(n=min(cap, len(g)), random_state=SEED))
    sub = pd.concat(capped).reset_index(drop=True)

    sub[["id", "sign", "signer", "split"]].to_csv("data/lsfb_si_meta.csv", index=False)
    print(f"selected {len(sub)} clips | {sub['sign'].nunique()} signs | "
          f"train/val/test signers = {n-n_test-n_val}/{n_val}/{n_test} | "
          f"clips {dict(sub['split'].value_counts())}")

    origins = [f"poses/{lm}/{iid}.npy" for iid in sub["id"] for lm in LANDMARKS]
    print(f"downloading {len(origins)} pose files with {WORKERS} workers ...")

    sess = requests.Session()
    done = {"n": 0, "fail": 0}
    def fetch(origin):
        fp = os.path.join(DEST, origin)
        if os.path.isfile(fp) and os.path.getsize(fp) > 0:
            return True
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        try:
            r = sess.get(f"{BASE}/{origin}", timeout=60, allow_redirects=False)
            if r.status_code == 200 and r.headers.get("Content-Type") != "text/html" and r.content:
                with open(fp, "wb") as f: f.write(r.content)
                return True
        except Exception:
            pass
        return False
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, o): o for o in origins}
        for i, fut in enumerate(as_completed(futs), 1):
            ok = fut.result()
            done["n"] += 1
            if not ok: done["fail"] += 1
            if i % 1000 == 0 or i == len(origins):
                print(f"  {i}/{len(origins)}  fail={done['fail']}", flush=True)
    print(f"DONE: {done['n']-done['fail']}/{done['n']} ok, {done['fail']} failed")

if __name__ == "__main__":
    main()
