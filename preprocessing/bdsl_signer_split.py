"""Signer-independent split definition for BdSLW60.

The 9,307 clips in BdSLW60 come from 18 signers (U01..U18). A random
clip-level split (the default in `preprocess_bdsl.py`) mixes the same
signer's repetitions of the same word across train/val, which lets a
model learn signer identity as a shortcut. This module defines a fixed
signer-disjoint assignment that all downstream preprocessing can share.

Design decisions:

* **Only "full" signers go in train/val/test.** 11 of the 18 signers
  recorded nearly every word in the 60-class vocabulary. The other 7
  recorded 9-38 classes. Placing partial/tiny signers in train would
  unbalance the class distribution; placing them in val/test would mean
  evaluating on an under-sampled subset of classes. They are routed to
  a separate "pretrain" pool so Option-C style SSL can use them as
  domain-matched unlabeled data.

* **Small, publicly-motivated val / test sets.** One val signer (U15) is
  enough for model selection; two test signers (U02, U13) give ~1,400
  clips with full 60-class coverage, which is the standard
  signer-independent protocol used in recent SLR literature (EMNLP
  Findings 2025; BdSLW401 arXiv 2503.02360).

Filename format: `U<signer>W<word>F[_<session>]_trial_<N>_[LR].mp4`.
The optional `_<session>` infix appears only for 30 files in the
`dengue` class.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict


_FILENAME_RE = re.compile(
    r"^U(\d+)W(\d+)([A-Z]*)(?:_(\d+))?_trial_(\d+)_([LR])\.mp4$",
    re.IGNORECASE,
)


SIGNER_SPLIT = {
    "train":    (1, 4, 5, 6, 8, 9, 11, 12),
    "val":      (15,),
    "test":     (2, 13),
    "pretrain": (3, 7, 10, 14, 16, 17, 18),
}


LABELED_SPLITS = ("train", "val", "test")
ALL_SPLITS = ("train", "val", "test", "pretrain")


def _build_lookup():
    lookup = {}
    for split_name, signers in SIGNER_SPLIT.items():
        for s in signers:
            if s in lookup:
                raise ValueError(
                    "signer U{:02d} appears in both '{}' and '{}'".format(
                        s, lookup[s], split_name
                    )
                )
            lookup[s] = split_name
    return lookup


_SIGNER_TO_SPLIT = _build_lookup()


def parse_filename(name):
    """Parse `U<s>W<w>F[_<session>]_trial_<n>_[LR].mp4`.

    Returns a tuple (signer:int, word:int, trial:int, hand:str) or None if
    the filename does not match the expected pattern.
    """
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    signer = int(m.group(1))
    word = int(m.group(2))
    trial = int(m.group(5))
    hand = m.group(6).upper()
    return signer, word, trial, hand


def classify_signer(signer_id):
    """Map a signer ID (1..18) to its split name. Raises on unknown IDs."""
    try:
        return _SIGNER_TO_SPLIT[int(signer_id)]
    except KeyError as exc:
        raise ValueError("unknown signer id: {}".format(signer_id)) from exc


def all_signers():
    """Flat, sorted list of every signer id mentioned in SIGNER_SPLIT."""
    return sorted(_SIGNER_TO_SPLIT.keys())


def scan_dataset(root):
    """Yield (filepath, class_name, signer, split_name) for every parseable clip.

    Clips whose filename does not match `_FILENAME_RE` are silently skipped;
    use `find_unparseable` to audit them.
    """
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fn in sorted(os.listdir(class_dir)):
            parsed = parse_filename(fn)
            if parsed is None:
                continue
            signer = parsed[0]
            if signer not in _SIGNER_TO_SPLIT:
                continue
            yield (
                os.path.join(class_dir, fn),
                class_name,
                signer,
                _SIGNER_TO_SPLIT[signer],
            )


def find_unparseable(root):
    """Return a list of filepaths that end in .mp4 but do not match the pattern."""
    out = []
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fn in sorted(os.listdir(class_dir)):
            if not fn.lower().endswith(".mp4"):
                continue
            if parse_filename(fn) is None:
                out.append(os.path.join(class_dir, fn))
    return out


def summarize_splits(root):
    """Return a dict: split_name -> {'clips', 'classes' (set), 'signers' (set)}."""
    summary = defaultdict(lambda: {"clips": 0, "classes": set(), "signers": set()})
    for _, class_name, signer, split_name in scan_dataset(root):
        bucket = summary[split_name]
        bucket["clips"] += 1
        bucket["classes"].add(class_name)
        bucket["signers"].add(signer)
    return dict(summary)


def write_split_json(out_path):
    """Dump the canonical signer assignment to a JSON file for reproducibility."""
    payload = {
        "schema": "bdslw60-signer-independent-v1",
        "description": (
            "Fixed signer-disjoint split of BdSLW60 for reproducible benchmarking. "
            "Train/val/test contain only signers who recorded the full 60-word "
            "vocabulary; partial-coverage signers are routed to a pretraining pool."
        ),
        "filename_pattern": _FILENAME_RE.pattern,
        "splits": {k: list(v) for k, v in SIGNER_SPLIT.items()},
    }
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="path to the BdSLW30/BdSLW60 folder containing class subdirs",
    )
    parser.add_argument(
        "--write-json",
        default=None,
        help="optional output path for the canonical split JSON",
    )
    args = parser.parse_args()

    summary = summarize_splits(args.root)
    print("{:<10} {:>6} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        bucket = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>6} {:>8} {:>8}".format(
            name, bucket["clips"], len(bucket["classes"]), len(bucket["signers"])
        ))

    unparseable = find_unparseable(args.root)
    print("unparseable: {} (examples: {})".format(len(unparseable), unparseable[:3]))

    if args.write_json:
        write_split_json(args.write_json)
        print("wrote {}".format(args.write_json))


if __name__ == "__main__":
    _main()
