#!/usr/bin/env python
"""Do MediaPipe detect/lose jumps dominate the k-means pretraining targets? (P1 fix, step 5)

Samples clips proportionally from a pool manifest, builds per-frame features in `pose_motion`
(the original targets) and `pose_motion_masked` (motion only between frames where the joint is
detected in both), fits MiniBatchKMeans(K) with the same settings as the target script on each,
and reports:
  * share of frames that are TRANSITION frames (some joint changes detected state vs t-1),
  * share of squared motion-norm mass that sits in transition frames,
  * clusters whose members are >= 50% transition frames ("dropout clusters"), their count and the
    share of all frames they hold.
Pre-set decision rule (written before running): use `pose_motion_masked` for the v2 targets iff,
under `pose_motion`, dropout clusters hold >= 10% of frames OR are >= 10% of clusters.
"""
import argparse, json, os, sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from preprocessing.compute_pretrain_targets_v2 import masked_pose_motion  # noqa: E402
from preprocessing.compute_pretrain_targets import _clip_to_per_frame_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ssl_pool_manifest_bdsl_asl_v2.json")
    ap.add_argument("--n-clips", type=int, default=4000); ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clips = json.load(open(a.manifest))["clips"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(clips), size=min(a.n_clips, len(clips)), replace=False)
    F = {"pose_motion": [], "pose_motion_masked": []}; TR = []; SRC = []
    for i in idx:
        with np.load(clips[i]["path"]) as h:
            d = h["data"]
        valid = d.sum(axis=(0, 2, 3)) != 0
        if not valid.any():
            continue
        d = d[:, : int(np.where(valid)[0].max()) + 1]
        det = np.abs(d[:2, :, :, 0]).sum(0) > 0
        tr = np.zeros(det.shape[0], bool); tr[1:] = (det[1:] ^ det[:-1]).any(1)
        F["pose_motion"].append(_clip_to_per_frame_features(d, "pose_motion"))
        F["pose_motion_masked"].append(masked_pose_motion(d))
        TR.append(tr); SRC += [clips[i]["source"]] * det.shape[0]
    TR = np.concatenate(TR); SRC = np.array(SRC)
    res = {"frames": int(TR.size), "transition_frame_share": float(TR.mean()),
           "transition_frame_share_by_source": {s: float(TR[SRC == s].mean()) for s in sorted(set(SRC))}}
    for mode, L in F.items():
        X = np.concatenate(L)
        mot = X[:, X.shape[1] // 2:]
        m2 = (mot ** 2).sum(1)
        km = MiniBatchKMeans(n_clusters=a.K, random_state=0, batch_size=8192, n_init=3).fit(X)
        lab = km.labels_
        share_tr = np.array([TR[lab == k].mean() if (lab == k).any() else 0 for k in range(a.K)])
        occ = np.bincount(lab, minlength=a.K) / lab.size
        dc = share_tr >= 0.5
        res[mode] = {"motion_sq_mass_in_transition_frames": float(m2[TR].sum() / max(m2.sum(), 1e-12)),
                     "dropout_clusters": int(dc.sum()), "dropout_cluster_frame_share": float(occ[dc].sum()),
                     "transition_frames_in_dropout_clusters": float(TR[np.isin(lab, np.where(dc)[0])].sum() / max(TR.sum(), 1)),
                     "cluster_transition_share_sorted": sorted(np.round(share_tr, 3).tolist(), reverse=True)[:12]}
        print(mode, json.dumps(res[mode]), flush=True)
    pm = res["pose_motion"]
    res["decision_use_masked"] = bool(pm["dropout_cluster_frame_share"] >= 0.10 or pm["dropout_clusters"] >= 0.10 * a.K)
    print(json.dumps({k: v for k, v in res.items() if k not in F}, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
