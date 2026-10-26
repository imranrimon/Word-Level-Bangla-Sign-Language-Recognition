#!/usr/bin/env python3
"""Prepare an NTU domain as an SSL pretraining source: explode its train bundle
into per-clip .npz (key 'data' = (C,T,V,M), trailing zeros stripped) and write an
SSL pool manifest JSON — the format feeders/pretrain_feeder.py +
preprocessing/compute_pretrain_targets.py expect.
Usage: python preprocessing/prep_ntu_ssl_source.py --domain cam2
"""
import os, json, argparse
import numpy as np

def valid_len(clip):  # clip (C,T,V,M); trailing zeros at the end (left-aligned)
    e = np.abs(clip).sum(axis=(0, 2, 3))
    nz = np.nonzero(e)[0]
    return int(nz[-1] + 1) if len(nz) else clip.shape[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--prefix", default="ntu", help="bundle dir prefix: data/<prefix>_<domain>/")
    a = ap.parse_args()
    pfx = a.prefix
    bundle = f"data/{pfx}_{a.domain}/train_data.npy"
    cache = f"data/{pfx}_{a.domain}_ssl_cache"
    os.makedirs(cache, exist_ok=True)
    arr = np.load(bundle, mmap_mode="r")
    print(f"[{pfx}_{a.domain}] {arr.shape[0]} clips -> {cache}")
    clips = []
    for i in range(arr.shape[0]):
        c = np.asarray(arr[i], dtype=np.float32)   # (C,T,V,M)
        T = valid_len(c)
        c = c[:, :max(T, 2)]
        p = f"{cache}/{a.domain}_{i:06d}.npz"
        np.savez_compressed(p, data=c)
        clips.append({"path": p, "source": f"{pfx}_{a.domain}"})
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{arr.shape[0]}", flush=True)
    man = f"data/ssl_pool_manifest_{pfx}_{a.domain}.json"
    json.dump({"clips": clips, "per_source_counts": {f"{pfx}_{a.domain}": len(clips)}},
              open(man, "w"))
    print(f"[wrote {man}: {len(clips)} clips]")

if __name__ == "__main__":
    main()
