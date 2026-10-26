"""Signer-independent split definition for GSL isol. (Greek Sign Language, isolated).

GSL isol. is the isolated-gloss subset of the Greek Sign Language dataset
(Adaloglou et al., "A Comprehensive Study on Sign Language Recognition Methods").
It ships as pre-cut per-gloss RGB frame folders inside `GSL_isolated.tar.gz`:

    <scenario><N>_signer<S>_rep<R>_glosses/glosses<GGGG>/frame_<FFFF>.jpg

with 310 unique glosses, 7 signers, 3 scenarios (health/kep/police), 5
scenario-blocks each. The official iso split CSVs
(GSL_split/GSL_isolated/{train,val,test}_greek_iso.csv, rows
`<scenario><N>_signer<S>_rep<R>_glosses/glosses<GGGG>|<GLOSS>`) are NOT
signer-disjoint (the same 7 signers recur across train/val/test).

For the cross-lingual transfer study (P1, 6th target language) we IGNORE the
official iso split and build our OWN fixed **signer-disjoint** 5/1/1 partition
over the 7 signers, mirroring `lsa64_signer_split.py`. The class label is the
Greek gloss string (field 2 of the iso CSVs); the canonical 310-class vocabulary
is `GSL_isolated/iso_classes.csv`.

Design decisions:

* **All 7 signers cover (nearly) all 310 glosses** across the 3 scenarios, so
  there is no "partial signer" problem — every signer is eligible for any split.
  We mirror the BdSLW60/LSA64 ratio (majority train, one val, one/one test) to
  keep the signer-independent protocol comparable across target languages.

* **train = signers 1-5, val = signer 6, test = signer 7.** Signer-disjoint.

Directory-name field:  `<scenario><N>_signer<S>_rep<R>_glosses`
Instance path in CSV:  `<dir>/glosses<GGGG>|<GLOSS>`
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict


# Matches the top-level directory field of an iso-CSV path, e.g.
# "police1_signer1_rep1_glosses" -> scenario=police, block=1, signer=1, rep=1
_DIR_RE = re.compile(
    r"^(?P<scenario>[a-z]+)(?P<block>\d+)_signer(?P<signer>\d+)_rep(?P<rep>\d+)_glosses$",
    re.IGNORECASE,
)


SIGNER_SPLIT = {
    "train": (1, 2, 3, 4, 5),
    "val":   (6,),
    "test":  (7,),
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


def parse_dir(dir_name):
    """Parse an iso-CSV top-level dir field.

    Returns dict(scenario, block, signer, rep) or None if it does not match.
    """
    m = _DIR_RE.match(dir_name)
    if m is None:
        return None
    return {
        "scenario": m.group("scenario").lower(),
        "block": int(m.group("block")),
        "signer": int(m.group("signer")),
        "rep": int(m.group("rep")),
    }


def classify_signer(signer_id):
    """Map a signer id (1..7) to its split name. Raises on unknown ids."""
    try:
        return _SIGNER_TO_SPLIT[int(signer_id)]
    except KeyError as exc:
        raise ValueError("unknown signer id: {}".format(signer_id)) from exc


def all_signers():
    return sorted(_SIGNER_TO_SPLIT.keys())


def read_iso_csv(csv_path):
    """Read an iso CSV (`<dir>/glosses<GGGG>|<GLOSS>`).

    Yields (rel_path, gloss) for each row. `rel_path` is the clip directory
    relative to the isolated dataset root (e.g.
    'police1_signer1_rep1_glosses/glosses0000'); `gloss` is the class string.
    """
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line or "|" not in line:
                continue
            rel_path, gloss = line.split("|", 1)
            yield rel_path.strip(), gloss.strip()


def read_classes(iso_classes_path):
    """Read the canonical 310-gloss vocabulary from iso_classes.csv (one per line)."""
    classes = []
    with open(iso_classes_path, encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                classes.append(g)
    return classes


def scan_dataset(iso_root, csv_paths):
    """Yield (clip_dir, gloss, signer, split_name) for every clip present on disk.

    `iso_root` is the extracted isolated dataset root (contains the
    `<...>_glosses/` directories). `csv_paths` is an iterable of iso-CSV paths
    (train/val/test_greek_iso.csv) that enumerate the clips and their glosses;
    we UNION them and re-split by our signer-disjoint assignment (the official
    CSV split is discarded).
    """
    seen = set()
    for csv_path in csv_paths:
        if not csv_path or not os.path.exists(csv_path):
            continue
        for rel_path, gloss in read_iso_csv(csv_path):
            if rel_path in seen:
                continue
            seen.add(rel_path)
            top = rel_path.split("/", 1)[0]
            parsed = parse_dir(top)
            if parsed is None:
                continue
            signer = parsed["signer"]
            if signer not in _SIGNER_TO_SPLIT:
                continue
            clip_dir = os.path.join(iso_root, rel_path)
            yield (clip_dir, gloss, signer, _SIGNER_TO_SPLIT[signer])


def summarize_splits(iso_root, csv_paths, require_exists=False):
    """Return split_name -> {'clips', 'classes' (set), 'signers' (set)}."""
    summary = defaultdict(lambda: {"clips": 0, "classes": set(), "signers": set()})
    for clip_dir, gloss, signer, split_name in scan_dataset(iso_root, csv_paths):
        if require_exists and not os.path.isdir(clip_dir):
            continue
        b = summary[split_name]
        b["clips"] += 1
        b["classes"].add(gloss)
        b["signers"].add(signer)
    return dict(summary)


def write_split_json(out_path):
    payload = {
        "schema": "gsl-isol-signer-independent-v1",
        "description": (
            "Fixed signer-disjoint split of GSL isol. (Greek Sign Language, isolated) "
            "for cross-lingual transfer evaluation (P1, 6th target language). "
            "310 glosses, 7 signers; train=signers 1-5, val=6, test=7. The official "
            "iso split CSVs are NOT signer-disjoint and are unioned then re-split here."
        ),
        "dir_pattern": _DIR_RE.pattern,
        "splits": {k: list(v) for k, v in SIGNER_SPLIT.items()},
    }
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso-root", required=True,
                        help="extracted GSL isolated dataset root (has *_glosses/ dirs)")
    parser.add_argument("--csv-dir", required=True,
                        help="dir with {train,val,test}_greek_iso.csv (GSL_split/GSL_isolated)")
    parser.add_argument("--require-exists", action="store_true",
                        help="only count clips whose frame dir exists on disk")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()

    csvs = [os.path.join(args.csv_dir, "{}_greek_iso.csv".format(s))
            for s in ("train", "val", "test")]
    summary = summarize_splits(args.iso_root, csvs, require_exists=args.require_exists)
    print("{:<10} {:>7} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        b = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>7} {:>8} {:>8}".format(
            name, b["clips"], len(b["classes"]), len(b["signers"])
        ))
    if args.write_json:
        write_split_json(args.write_json)
        print("wrote {}".format(args.write_json))


if __name__ == "__main__":
    _main()
