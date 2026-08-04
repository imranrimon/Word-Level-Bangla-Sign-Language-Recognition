"""Build stratified sub-sampled BdSLW60-SI train bundles + lr0.01 fraction configs
for the data-efficiency curve. Deterministic (seed 42), class-stratified.

    python tools/make_bdsl_fractions.py            # 10/25/50%
"""
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SI = os.path.join(ROOT, "data", "bdsl_si")
CFG = os.path.join(ROOT, "config")
FRACS = [0.10, 0.25, 0.50]


def main():
    data = np.load(os.path.join(SI, "train_data.npy"))  # full load on a compute node
    names, labels = pickle.load(open(os.path.join(SI, "train_label.pkl"), "rb"))
    labels = np.array(labels)
    rng = np.random.RandomState(42)
    for frac in FRACS:
        tag = int(frac * 100)
        idx = []
        for c in np.unique(labels):
            ci = np.where(labels == c)[0]
            k = max(1, int(round(len(ci) * frac)))
            idx.extend(rng.choice(ci, k, replace=False).tolist())
        idx = sorted(idx)
        np.save(os.path.join(SI, "train_data_f{}.npy".format(tag)), data[idx])
        with open(os.path.join(SI, "train_label_f{}.pkl".format(tag)), "wb") as fh:
            pickle.dump(([names[i] for i in idx], [int(labels[i]) for i in idx]), fh)
        print("f{}: {} clips, {} classes".format(tag, len(idx), len(set(labels[idx]))), flush=True)

    tmpl = open(os.path.join(CFG, "bdsl_block_gcn_si_lr01.yaml")).read()
    for f in [10, 25, 50]:
        out = tmpl.replace("train_data.npy", "train_data_f{}.npy".format(f)) \
                  .replace("train_label.pkl", "train_label_f{}.pkl".format(f))
        open(os.path.join(CFG, "bdsl_block_gcn_si_lr01_f{}.yaml".format(f)), "w").write(out)
    ok = all(os.path.exists(os.path.join(CFG, "bdsl_block_gcn_si_lr01_f{}.yaml".format(f))) for f in [10, 25, 50])
    print("configs written:", ok, flush=True)


if __name__ == "__main__":
    main()
