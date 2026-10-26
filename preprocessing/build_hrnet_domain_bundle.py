#!/usr/bin/env python3
"""Build cross-domain bundles from PYSKL HRNet 2D pickles (17-joint COCO) for the
lab-vs-wild generality demo. Each DATASET is a domain: ntu_hrnet (lab, studio Kinect
video), hmdb51 / ucf101 (in-the-wild YouTube), gym (sport). Writes
data/hrnet_<domain>/{train,val,test}_{data.npy,label.pkl}, (N,C=3,T=300,V=17,M=1),
channels = (x, y, keypoint_score). Person 0 only. Left-aligned padding.
Usage: python preprocessing/build_hrnet_domain_bundle.py --domain hmdb51
"""
import os, pickle, argparse, json
import numpy as np

MAXF = 300; V = 17
PKL = {"ntu_hrnet": "data/hrnet_pyskl/ntu60_hrnet.pkl",
       "hmdb51": "data/hrnet_pyskl/hmdb51_hrnet.pkl",
       "ucf101": "data/hrnet_pyskl/ucf101_hrnet.pkl",
       "gym": "data/hrnet_pyskl/gym_hrnet.pkl"}
# (train_key, val_key, test_key) per dataset's native splits
SPLITS = {"ntu_hrnet": ("xsub_train", "xsub_val", "xsub_val"),
          "hmdb51": ("train1", "test1", "test1"),
          "ucf101": ("train1", "test1", "test1"),
          "gym": ("train", "val", "val")}

def to_clip(kp, score):
    kp = np.asarray(kp, np.float32); score = np.asarray(score, np.float32)   # (M,T,V,2),(M,T,V)
    m0 = kp[0]; s0 = score[0]                                                 # (T,V,2),(T,V)
    T = min(m0.shape[0], MAXF)
    x = m0[:T, :, 0].copy(); y = m0[:T, :, 1].copy(); sc = s0[:T].copy()      # (T,V)
    # SCALE-NORMALIZE: center + divide by per-clip pose bbox diagonal over VALID
    # (score>0) joints, so pixel-scale (5-6x across in-the-wild datasets) can't
    # confound cross-domain transfer or the MMD. Invalid joints stay 0.
    valid = sc > 0
    if valid.any():
        xs = x[valid]; ys = y[valid]
        cx, cy = float(xs.mean()), float(ys.mean())
        diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        diag = diag if diag > 1e-6 else 1.0
        x = np.where(valid, (x - cx) / diag, 0.0)
        y = np.where(valid, (y - cy) / diag, 0.0)
    else:
        x[:] = 0.0; y[:] = 0.0
    out = np.zeros((3, MAXF, V, 1), np.float32)
    out[0, :T, :, 0] = x; out[1, :T, :, 0] = y; out[2, :T, :, 0] = sc
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", required=True, choices=list(PKL))
    ap.add_argument("--cap-per-class", type=int, default=200); a = ap.parse_args()
    d = pickle.load(open(PKL[a.domain], "rb"))
    by_fd = {x["frame_dir"]: x for x in d["annotations"]}
    tr, va, te = SPLITS[a.domain]
    out_dir = f"data/hrnet_{a.domain}"; os.makedirs(out_dir, exist_ok=True)
    labels_sorted = sorted({x["label"] for x in d["annotations"]})
    remap = {c: i for i, c in enumerate(labels_sorted)}
    json.dump({"orig_labels": labels_sorted}, open(f"{out_dir}/classes.json", "w"))
    rng = np.random.RandomState(0)
    for sp, key in [("train", tr), ("val", va), ("test", te)]:
        fds = d["split"][key]
        # for datasets where val==test key, split the ids deterministically in half
        if sp in ("val", "test") and va == te:
            ids = sorted(fds); half = len(ids) // 2
            fds = ids[:half] if sp == "val" else ids[half:]
        items = [by_fd[f] for f in fds if f in by_fd]
        if sp == "train" and a.cap_per_class:
            byc = {}
            for x in items: byc.setdefault(x["label"], []).append(x)
            items = [lst[i] for lst in byc.values() for i in rng.permutation(len(lst))[:a.cap_per_class]]
        data, labs, names = [], [], []
        for x in items:
            data.append(to_clip(x["keypoint"], x["keypoint_score"])); labs.append(remap[x["label"]]); names.append(x["frame_dir"])
        arr = np.stack(data).astype(np.float32) if data else np.zeros((0,3,MAXF,V,1),np.float32)
        np.save(f"{out_dir}/{sp}_data.npy", arr)
        pickle.dump((names, labs), open(f"{out_dir}/{sp}_label.pkl", "wb"))
        print(f"  {a.domain}/{sp}: {arr.shape} classes={len(set(labs))}")

if __name__ == "__main__":
    main()
