#!/usr/bin/env python
"""Verify native frame rates of every P1 corpus (P1 fix program, step 2).

For corpora whose source videos are on the cluster, read container fps and frame counts with
OpenCV for a random sample of videos, and compare the frame counts with the valid-frame counts
of the pose arrays (all extractors process every frame -- no temporal subsampling -- so the two
should agree; the bundles cap T at 300). For LSFB (authors' poses, no videos) fps is derived from
instances.csv start/end (ms) vs pose-array length. GSL ships JPEG frame folders (no container fps)
and WLASL/BdSLW401/BdSLW102a have no videos on the cluster -> reported as documented values.

  python tools/p1_fix/probe_fps.py --out docs/p1_fix_2026-10/step2/fps_probe.json
"""
import argparse, glob, json, os, random
import numpy as np

D = "data"
VIDEOS = {
    "BdSL": f"{D}/bdslw60_clips/*/*.mp4",
    "BdSLW60_raw": f"{D}/bdslw60_raw/*/*.mp4",
    "LSA64": f"{D}/lsa64_raw/all/*.mp4",
    "AUTSL": f"{D}/autsl_download/train/*_color.mp4",
    "KArSL": f"{D}/karsl_extracted/0[1-3]/*/*.mp4",
    "LIBRAS": f"{D}/libras_download/Sinalizador0*/**/*.mp4",
    "BISINDO": f"{D}/bisindo_raw/*.mp4",
    "ASL-Citizen": f"{D}/aslcitizen_download/ASL_Citizen/videos/*.mp4",
}
BUNDLES = {"BdSL": "data/bdsl_si/train_data.npy", "LSA64": "data/lsa64_si/train_data.npy",
           "AUTSL": "data/autsl_si/train_data.npy", "KArSL": "data/karsl_si/train_data.npy",
           "GSL": "data/gsl_si/train_data.npy", "LSFB": "data/lsfb_si/train_data.npy",
           "LIBRAS": "data/libras_si/train_data.npy", "BISINDO": "data/bisindo_si/train_data.npy",
           "ASL-Citizen": "data/aslcitizen60_si/train_data.npy"}
CACHES = {"WLASL": "data/wlasl_pose_cache_v2", "BdSLW401": "data/bdslw401_pose_cache_front",
          "BdSLW102a": "data/bdslw102_a_pose_cache"}


def valid_T(a):
    a = np.asarray(a)
    if a.ndim == 5: a = a[0]
    nz = np.abs(a[:2, :, :, 0]).sum((0, 2)) > 0
    return int(np.where(nz)[0].max() + 1) if nz.any() else 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    import cv2
    rnd = random.Random(0); out = {}
    for name, pat in VIDEOS.items():
        fs = sorted(glob.glob(pat, recursive=True))
        if not fs:
            out[name] = {"videos": 0}; print(name, "no videos"); continue
        fps, nfr = [], []
        for f in rnd.sample(fs, min(a.n, len(fs))):
            c = cv2.VideoCapture(f)
            fps.append(float(c.get(cv2.CAP_PROP_FPS))); nfr.append(int(c.get(cv2.CAP_PROP_FRAME_COUNT))); c.release()
        out[name] = {"videos": len(fs), "fps_values": sorted(set(round(x, 2) for x in fps)),
                     "fps_median": float(np.median(fps)), "frames_median": float(np.median(nfr)),
                     "dur_s_median": float(np.median(np.array(nfr) / np.maximum(fps, 1e-6)))}
        print(name, out[name], flush=True)
    # GSL frame folders
    g = glob.glob(f"{D}/gsl_download/GSL_isolated_extracted/Greek_isolated/GSL_isol/*/glosses*")
    if g:
        cnt = [len(glob.glob(os.path.join(x, "*.jpg")) + glob.glob(os.path.join(x, "*.png"))) for x in rnd.sample(g, min(200, len(g)))]
        out["GSL"] = {"frame_dirs": len(g), "frames_median": float(np.median(cnt)), "note": "JPEG frame folders; no container fps"}
        print("GSL", out["GSL"], flush=True)
    # LSFB: fps = frames / duration
    import csv
    rows = list(csv.DictReader(open(f"{D}/lsfb_download/instances.csv")))
    est = []
    for r in rnd.sample(rows, 3000):
        p = f"{D}/lsfb_download/poses/pose/{r['id']}.npy"
        if os.path.exists(p):
            T = np.load(p, mmap_mode="r").shape[0]; dur = (int(r["end"]) - int(r["start"])) / 1000.0
            if dur > 0: est.append(T / dur)
        if len(est) >= 300: break
    out["LSFB"] = {"fps_est_median": float(np.median(est)), "fps_est_iqr": [float(np.percentile(est, 25)), float(np.percentile(est, 75))], "n": len(est)}
    print("LSFB", out["LSFB"], flush=True)
    # valid-frame counts in the pose arrays
    vt = {}
    for name, p in BUNDLES.items():
        x = np.load(p, mmap_mode="r"); idx = np.random.default_rng(0).choice(len(x), min(300, len(x)), replace=False)
        T = [valid_T(x[i]) for i in idx]; vt[name] = {"T_median": float(np.median(T)), "T_p90": float(np.percentile(T, 90)), "cap": int(x.shape[2])}
    for name, p in CACHES.items():
        fs = sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True)); T = []
        for f in rnd.sample(fs, min(300, len(fs))):
            with np.load(f) as h: T.append(valid_T(h["data"]))
        vt[name] = {"T_median": float(np.median(T)), "T_p90": float(np.percentile(T, 90))}
    out["pose_valid_T"] = vt
    print(json.dumps(vt, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
