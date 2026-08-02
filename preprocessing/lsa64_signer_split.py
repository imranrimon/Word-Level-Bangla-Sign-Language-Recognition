"""Signer-independent split definition for LSA64 (Argentine Sign Language).

LSA64 has 3,200 clips: 64 signs x 10 signers x 5 repetitions, recorded as a
flat set of `.mp4` files (typically under an `all/` subdir) named
`<sign>_<signer>_<repetition>.mp4` (each field zero-padded to 3 digits, e.g.
`001_001_001.mp4`). Unlike BdSLW60 the clips are NOT organised into per-class
subdirectories — the class (sign id) is encoded in the filename.

This module mirrors `bdsl_signer_split.py` so the shared NPY generator can
reuse the same interface. It defines a fixed signer-disjoint assignment used
for the cross-lingual transfer study (2nd target language, plan R9).

Design decisions:

* **All 10 signers cover all 64 signs** (balanced, non-expert subjects), so
  there is no "partial signer" problem — every signer is eligible for any
  split. We mirror the BdSLW60 ratio (majority train, one val, two test) to
  keep the signer-independent protocol comparable across target languages.

* **train = signers 1-7, val = signer 8, test = signers 9-10.** Signer-disjoint;
  test gives 2 x 64 x 5 = 640 clips with full 64-class coverage.

Filename format: `<sign:3>_<signer:3>_<rep:3>.mp4`.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict


_FILENAME_RE = re.compile(r"^(\d{3})_(\d{3})_(\d{3})\.mp4$", re.IGNORECASE)


SIGNER_SPLIT = {
    "train": (1, 2, 3, 4, 5, 6, 7),
    "val":   (8,),
    "test":  (9, 10),
}


LABELED_SPLITS = ("train", "val", "test")
ALL_SPLITS = ("train", "val", "test")


def _build_lookup():
    lookup = {}
    for split_name, signers in SIGNER_SPLIT.items():
        for s in signers:
            if s in lookup:
                raise ValueError(
                    "signer {:03d} appears in both '{}' and '{}'".format(
                        s, lookup[s], split_name
                    )
                )
            lookup[s] = split_name
    return lookup


_SIGNER_TO_SPLIT = _build_lookup()


def parse_filename(name):
    """Parse `<sign>_<signer>_<rep>.mp4`.

    Returns (signer:int, sign:int, rep:int) or None if the name does not match.
    """
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    sign = int(m.group(1))
    signer = int(m.group(2))
    rep = int(m.group(3))
    return signer, sign, rep


def class_name_for(sign_id):
    """Canonical, deterministic class-name string for a sign id (e.g. 1 -> 'sign_001')."""
    return "sign_{:03d}".format(int(sign_id))


def classify_signer(signer_id):
    """Map a signer id (1..10) to its split name. Raises on unknown ids."""
    try:
        return _SIGNER_TO_SPLIT[int(signer_id)]
    except KeyError as exc:
        raise ValueError("unknown signer id: {}".format(signer_id)) from exc


def all_signers():
    return sorted(_SIGNER_TO_SPLIT.keys())


def _iter_mp4(root):
    """Yield every `.mp4` under `root` (recursively — handles the `all/` subdir)."""
    yield from glob.iglob(os.path.join(root, "**", "*.mp4"), recursive=True)


def scan_dataset(root):
    """Yield (filepath, class_name, signer, split_name) for every parseable clip."""
    for file_path in sorted(_iter_mp4(root)):
        parsed = parse_filename(os.path.basename(file_path))
        if parsed is None:
            continue
        signer, sign, _rep = parsed
        if signer not in _SIGNER_TO_SPLIT:
            continue
        yield (
            file_path,
            class_name_for(sign),
            signer,
            _SIGNER_TO_SPLIT[signer],
        )


def find_unparseable(root):
    """Return `.mp4` paths that do not match the expected naming pattern."""
    out = []
    for file_path in sorted(_iter_mp4(root)):
        if parse_filename(os.path.basename(file_path)) is None:
            out.append(file_path)
    return out


def summarize_splits(root):
    """Return split_name -> {'clips', 'classes' (set), 'signers' (set)}."""
    summary = defaultdict(lambda: {"clips": 0, "classes": set(), "signers": set()})
    for _, class_name, signer, split_name in scan_dataset(root):
        bucket = summary[split_name]
        bucket["clips"] += 1
        bucket["classes"].add(class_name)
        bucket["signers"].add(signer)
    return dict(summary)


def write_split_json(out_path):
    """Dump the canonical signer assignment to JSON for reproducibility."""
    payload = {
        "schema": "lsa64-signer-independent-v1",
        "description": (
            "Fixed signer-disjoint split of LSA64 (Argentine Sign Language) for "
            "cross-lingual transfer evaluation. 64 signs x 10 signers x 5 reps; "
            "train=signers 1-7, val=8, test=9-10."
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
    parser.add_argument("--root", required=True, help="dir containing LSA64 .mp4 clips")
    parser.add_argument("--write-json", default=None, help="optional split JSON output path")
    args = parser.parse_args()

    summary = summarize_splits(args.root)
    print("{:<10} {:>6} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        b = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>6} {:>8} {:>8}".format(
            name, b["clips"], len(b["classes"]), len(b["signers"])
        ))
    print("unparseable: {}".format(len(find_unparseable(args.root))))
    if args.write_json:
        write_split_json(args.write_json)
        print("wrote {}".format(args.write_json))


if __name__ == "__main__":
    _main()
