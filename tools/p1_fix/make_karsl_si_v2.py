#!/usr/bin/env python
"""Build signer-disjoint KArSL bundles from the existing data/karsl_si arrays (no re-extraction).

The original data/karsl_si is train = U01+U02 (88%), val = random 12% of the SAME signers
(signer-DEPENDENT), test = U03. Sample names are '<part>_<signer>_<class>_...'.
Writes two NEW bundles (the original is untouched):
  data/karsl_si_v2a : train = all U01+U02 clips (train+val), val = U03, test = U03
                      (same training signers/size as the law protocol; val is now a held-out signer)
  data/karsl_si_v2b : train = U01, val = U02, test = U03   (fully 3-way signer-disjoint)
"""
import os, pickle
import numpy as np
SRC = "data/karsl_si"

def load(s):
    x = np.load(f"{SRC}/{s}_data.npy", mmap_mode="r")
    n, l = pickle.load(open(f"{SRC}/{s}_label.pkl", "rb"))
    return x, list(n), list(l)

def write(d, s, x, n, l):
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/{s}_data.npy", np.ascontiguousarray(x))
    pickle.dump((n, l), open(f"{d}/{s}_label.pkl", "wb"))
    print(f"{d}/{s}: {len(n)} clips, signers {sorted(set(a.split('_')[1] for a in n))}, classes {len(set(l))}")

xt, nt, lt = load("train"); xv, nv, lv = load("val"); xs, ns, ls = load("test")
X = np.concatenate([np.asarray(xt), np.asarray(xv)]); N = nt + nv; L = lt + lv
sig = np.array([a.split("_")[1] for a in N])
assert set(sig) == {"01", "02"} and set(a.split("_")[1] for a in ns) == {"03"}
xs = np.asarray(xs)
write("data/karsl_si_v2a", "train", X, N, L)
write("data/karsl_si_v2a", "val", xs, ns, ls)
write("data/karsl_si_v2a", "test", xs, ns, ls)
for d, keep in (("data/karsl_si_v2b", "01"),):
    m = sig == keep; idx = np.where(m)[0]; jdx = np.where(~m)[0]
    write(d, "train", X[idx], [N[i] for i in idx], [L[i] for i in idx])
    write(d, "val", X[jdx], [N[i] for i in jdx], [L[i] for i in jdx])
    write(d, "test", xs, ns, ls)
import shutil
for d in ("data/karsl_si_v2a", "data/karsl_si_v2b"):
    shutil.copy2(f"{SRC}/classes.json", f"{d}/classes.json")
