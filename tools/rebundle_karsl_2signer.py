"""Re-bundle KArSL to a 2-signer train (U01+U02), test U03, so the low-data
cross-lingual test isn't starved (1-signer @10% was ~5 clips/class -> near chance).
val = stratified 12% slice of the combined train (selection only; test U03 is the
signer-independent metric). Overwrites train/val in data/karsl_si; test unchanged."""
import pickle
import numpy as np


def load(p):
    return np.load(f"data/karsl_si/{p}_data.npy"), pickle.load(open(f"data/karsl_si/{p}_label.pkl", "rb"))


def main():
    (d1, (n1, l1)) = load("train")   # U01
    (d2, (n2, l2)) = load("val")     # U02
    data = np.concatenate([d1, d2], 0)
    names = list(n1) + list(n2)
    labels = np.array(list(l1) + list(l2))
    rng = np.random.default_rng(42)
    val_idx = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * 0.12)))
        val_idx += list(idx[:k])
    val_mask = np.zeros(len(labels), bool); val_mask[val_idx] = True
    tr = ~val_mask
    np.save("data/karsl_si/train_data.npy", data[tr])
    pickle.dump(([names[i] for i in np.where(tr)[0]], labels[tr].tolist()),
                open("data/karsl_si/train_label.pkl", "wb"))
    np.save("data/karsl_si/val_data.npy", data[val_mask])
    pickle.dump(([names[i] for i in val_idx], labels[val_mask].tolist()),
                open("data/karsl_si/val_label.pkl", "wb"))
    print(f"2-signer rebundle: train {tr.sum()} val {val_mask.sum()} "
          f"(classes {len(np.unique(labels))}, ~{tr.sum()/len(np.unique(labels)):.1f}/class) | test U03 unchanged")


if __name__ == "__main__":
    main()
