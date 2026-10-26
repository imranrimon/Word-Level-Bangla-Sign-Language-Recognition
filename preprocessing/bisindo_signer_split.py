"""Signer-independent split definition for WL-BISINDO (Indonesian Sign Language).

WL-BISINDO (Word-Level Bahasa Isyarat Indonesia, Banten regional variant;
Kindy, Leonali & Lucky, Procedia CS 2025, doi:10.1016/j.procs.2025.08.277) is a
flat set of 1,600 `.mp4` clips: 32 isolated signs performed by 5 signers.
Clips are NOT organised into per-class subdirectories; the class (label id) and
the signer id are encoded in the filename:

    <signer:int>_<label:int>_<sample:int>.mp4     e.g. `signer3_label17_sample2.mp4`

This module mirrors `lsa64_signer_split.py` so the shared NPY generator
(`generate_bisindo_split_npy.py`) can reuse the same interface. It defines a
fixed signer-disjoint assignment used for the cross-lingual transfer study
(6th target language, P1 distance-gating law).

Design decisions (verified from the released file listing):

* **Signers 1, 2, 3, 4 each cover ALL 32 signs** (420/420/320/320 clips resp.);
  signer 0 is a PARTIAL signer (120 clips, only 12 of 32 classes) and so is
  EXCLUDED from every labeled split — a partial signer would give an unbalanced
  val/test set and break the SI protocol. Excluding it keeps every split at full
  32-class coverage, exactly like LSA64/GSL.

* **train = signers 1,2 / val = signer 3 / test = signer 4.** Signer-disjoint;
  mirrors the BdSLW60/LSA64/GSL protocol (majority train, one val, one test).
  Gives train ~840, val ~320, test ~320 clips, all 32 classes each.

Filename format: `signer<int>_label<int>_sample<int>.mp4` (fields not zero-padded).
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict


_FILENAME_RE = re.compile(
    r"^signer(\d+)_label(\d+)_sample(\d+)\.mp4$", re.IGNORECASE
)


# Signers 1-4 cover all 32 classes; signer 0 is partial (12 classes) -> excluded.
SIGNER_SPLIT = {
    "train": (1, 2),
    "val":   (3,),
    "test":  (4,),
}


LABELED_SPLITS = ("train", "val", "test")
ALL_SPLITS = ("train", "val", "test")


def _build_lookup():
    lookup = {}
    for split_name, signers in SIGNER_SPLIT.items():
        for s in signers:
            if s in lookup:
                raise ValueError(
                    "signer {} appears in both '{}' and '{}'".format(
                        s, lookup[s], split_name
                    )
                )
            lookup[s] = split_name
    return lookup


_SIGNER_TO_SPLIT = _build_lookup()


def parse_filename(name):
    """Parse `signer<S>_label<L>_sample<K>.mp4`.

    Returns (signer:int, label:int, sample:int) or None if the name does not match.
    """
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    signer = int(m.group(1))
    label = int(m.group(2))
    sample = int(m.group(3))
    return signer, label, sample


def class_name_for(label_id):
    """Canonical, deterministic class-name string for a label id (e.g. 5 -> 'label_05')."""
    return "label_{:02d}".format(int(label_id))


def classify_signer(signer_id):
    """Map a signer id to its split name. Raises on ids not in a labeled split
    (e.g. the excluded partial signer 0)."""
    try:
        return _SIGNER_TO_SPLIT[int(signer_id)]
    except KeyError as exc:
        raise ValueError("signer id not in any labeled split: {}".format(signer_id)) from exc


def all_signers():
    return sorted(_SIGNER_TO_SPLIT.keys())


def _iter_mp4(root):
    """Yield every `.mp4` under `root` (recursively)."""
    yield from glob.iglob(os.path.join(root, "**", "*.mp4"), recursive=True)


def scan_dataset(root):
    """Yield (filepath, class_name, signer, split_name) for every parseable clip
    whose signer belongs to a labeled split (partial signer 0 is skipped)."""
    for file_path in sorted(_iter_mp4(root)):
        parsed = parse_filename(os.path.basename(file_path))
        if parsed is None:
            continue
        signer, label, _sample = parsed
        if signer not in _SIGNER_TO_SPLIT:
            continue
        yield (
            file_path,
            class_name_for(label),
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
        "schema": "bisindo-signer-independent-v1",
        "description": (
            "Fixed signer-disjoint split of WL-BISINDO (Indonesian Sign Language, "
            "Banten variant) for cross-lingual transfer evaluation (P1, 6th target "
            "language). 32 signs x 5 signers x ~10 samples = 1600 clips. Signers 1-4 "
            "cover all 32 classes; partial signer 0 (12 classes) is EXCLUDED. "
            "train=signers 1-2, val=3, test=4 (signer-disjoint)."
        ),
        "filename_pattern": _FILENAME_RE.pattern,
        "excluded_signers": [0],
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
    parser.add_argument("--root", required=True, help="dir containing WL-BISINDO .mp4 clips")
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
