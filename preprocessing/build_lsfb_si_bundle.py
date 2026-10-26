#!/usr/bin/env python3
"""Convert downloaded LSFB-ISOL cleaned poses -> 27-keypoint SI bundles.

Consumes data/lsfb_si_meta.csv (id,sign,signer,split) + data/lsfb_download/poses/
{pose,left_hand,right_hand}/{id}.npy and writes data/lsfb_si/{train,val,test}_
{data.npy,label.pkl} in the same (N, 3, T, 27, 1) float32 layout as the other SI
datasets (matches preprocessing/convert_wlasl_landmarks_to_pose_cache.py).

27-kp layout (identical convention to BdSL/WLASL):
  0-6   body        = MediaPipe pose [0,12,11,14,13,16,15]
  7-16  right hand  = hand-local [0,4,5,8,9,12,13,16,17,20]
  17-26 left hand   = same hand-local indices
Channels: x, y, binary-confidence (1 where landmark present).
"""
import os, pickle
import numpy as np, pandas as pd

DEST = "data/lsfb_download"
OUT = "data/lsfb_si"
MAX_FRAMES = 300
BODY = [0, 12, 11, 14, 13, 16, 15]
HAND_LOCAL = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

def load_lm(iid, kind):
    p = os.path.join(DEST, "poses", kind, f"{iid}.npy")
    try:
        if os.stat(p).st_size == 0:   # os.stat, not isfile(): an I/O error must not read as "missing"
            return None
    except FileNotFoundError:
        return None
    try:
        return np.load(p).astype(np.float32)   # (T, K, 3) float16 -> float32
    except OSError:  # unreadable storage (e.g. Lustre OST offline) must not silently drop the clip
        raise
    except Exception:
        return None

def convert(iid):
    """-> (3, T, 27, 1) float32 or None if any stream missing."""
    pose = load_lm(iid, "pose"); lh = load_lm(iid, "left_hand"); rh = load_lm(iid, "right_hand")
    if pose is None or lh is None or rh is None:
        return None
    T = min(pose.shape[0], lh.shape[0], rh.shape[0])
    if T < 2:
        return None
    body = pose[:T, BODY, :]          # (T,7,3)
    rhand = rh[:T, HAND_LOCAL, :]     # (T,10,3)
    lhand = lh[:T, HAND_LOCAL, :]     # (T,10,3)
    sel = np.concatenate([body, rhand, lhand], axis=1)   # (T,27,3)
    conf = (np.abs(sel).sum(-1) > 0).astype(np.float32)  # (T,27)
    out = np.zeros((3, T, 27, 1), dtype=np.float32)
    out[0, :, :, 0] = sel[:, :, 0]
    out[1, :, :, 0] = sel[:, :, 1]
    out[2, :, :, 0] = conf
    return out

def pad_crop(clip):
    """(3,T,27,1) -> (3,MAX_FRAMES,27,1). LEFT-align (real frames first, zeros at
    END) to match the feeder's valid_frame_length convention (non_zero[-1]+1,
    'zero padding usually at the end'); center-padding buries the ~17 real frames
    mid-buffer so the 120-crop lands on leading zeros -> near-chance accuracy."""
    T = clip.shape[1]
    if T >= MAX_FRAMES:
        return clip[:, :MAX_FRAMES]          # keep first MAX_FRAMES
    out = np.zeros((3, MAX_FRAMES, 27, 1), dtype=np.float32)
    out[:, :T] = clip                        # left-align; zeros trail at the end
    return out

def main():
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv("data/lsfb_si_meta.csv")
    signs = sorted(meta["sign"].unique())
    sign2idx = {s: i for i, s in enumerate(signs)}
    with open(os.path.join(OUT, "classes.json"), "w") as f:
        import json; json.dump(sign2idx, f, ensure_ascii=False, indent=2)
    print(f"{len(signs)} classes")
    for split in ["train", "val", "test"]:
        rows = meta[meta["split"] == split]
        data, names, labels, miss = [], [], [], 0
        for _, r in rows.iterrows():
            c = convert(r["id"])
            if c is None:
                miss += 1; continue
            data.append(pad_crop(c)); names.append(r["id"]); labels.append(sign2idx[r["sign"]])
        arr = np.stack(data).astype(np.float32) if data else np.zeros((0, 3, MAX_FRAMES, 27, 1), np.float32)
        np.save(os.path.join(OUT, f"{split}_data.npy"), arr)
        with open(os.path.join(OUT, f"{split}_label.pkl"), "wb") as f:
            pickle.dump((names, labels), f)
        print(f"{split}: {arr.shape}  (missing/bad {miss})  classes={len(set(labels))}")

if __name__ == "__main__":
    main()
