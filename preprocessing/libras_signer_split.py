"""Signer-independent split definition for MINDS-Libras (Brazilian Sign
Language, isolated glosses).

MINDS-Libras (Rezende et al., 2021) is an isolated Brazilian Sign Language
(LIBRAS) dataset: 20 signs performed 5 times each by 12 signers (~1195 clips;
a few signer/sign combinations are missing). On Zenodo (record 2667329) it
ships as ONE .zip PER SIGNER — `Sinalizador01.zip` .. `Sinalizador12.zip`
("Sinalizador" = signer in Portuguese) — so the per-clip signer label is the
TOP-LEVEL directory, making a signer-disjoint split unambiguous.

This is the 6th cross-lingual TARGET language for the distance-gating law (P1),
replacing INCLUDE-50 (dropped: its official split is random, not signer-disjoint).
LIBRAS is a genuinely distinct language (not in the ASL/BdSL/Turkish/Arabic/Greek
pretrain pool or existing targets), spreading the source->target distance axis.

We build our OWN fixed signer-disjoint partition over the 12 signers, mirroring
`lsa64_signer_split.py` / `gsl_signer_split.py` but keeping the majority-train /
held-out-val / held-out-test ratio comparable across languages:

* **train = signers 1-8, val = signers 9-10, test = signers 11-12.**
  Signer-disjoint; test covers 2 signers x 20 signs x 5 reps ~= 200 clips.

Internal clip layout (confirmed by inspecting Sinalizador07.zip):
    Sinalizador<SS>/<Camera>/<NN><SignName>Sinalizador<SS>-<rep>.mp4
e.g. `Sinalizador07/Canon/01AcontecerSinalizador07-1.mp4`. The zips also carry a
macOS `__MACOSX/.../._<clip>.mp4` AppleDouble resource-fork sibling per file —
those are NOT real videos and are excluded (any path under `__MACOSX` or with a
`._` basename prefix). Both the signer AND the class are encoded in the filename:
`<NN><SignName>Sinalizador<SS>-<rep>.mp4`, so the class is `<NN><SignName>` (e.g.
`01Acontecer`), the signer is `<SS>`, the repetition is `<rep>`. The class
vocabulary is the union of `<NN><SignName>` labels observed on disk (sorted for
determinism) so the emitted bundle needs no external class list.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict


# Top-level per-signer directory, e.g. "Sinalizador07" -> signer 7.
_SIGNER_DIR_RE = re.compile(r"^Sinalizador0*(\d+)$", re.IGNORECASE)

# Clip basename, e.g. "01AcontecerSinalizador07-1.mp4":
#   group(1) = class number ("01"), group(2) = sign name ("Acontecer"),
#   group(3) = signer id ("07"),   group(4) = repetition ("1").
_CLIP_RE = re.compile(
    r"^(\d+)([A-Za-zÀ-ÿ]+)Sinalizador0*(\d+)-(\d+)\.(?:mp4|avi|mov|mkv)$",
    re.IGNORECASE,
)

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


SIGNER_SPLIT = {
    "train": (1, 2, 3, 4, 5, 6, 7, 8),
    "val":   (9, 10),
    "test":  (11, 12),
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


def classify_signer(signer_id):
    """Map a signer id (1..12) to its split name. Raises on unknown ids."""
    try:
        return _SIGNER_TO_SPLIT[int(signer_id)]
    except KeyError as exc:
        raise ValueError("unknown signer id: {}".format(signer_id)) from exc


def all_signers():
    return sorted(_SIGNER_TO_SPLIT.keys())


def _is_junk(path):
    """True for macOS AppleDouble resource forks / metadata that ride along in
    the Zenodo zips (`__MACOSX/.../._foo.mp4`, or any `._`-prefixed basename)."""
    norm = path.replace("\\", "/")
    if "/__MACOSX/" in norm or norm.startswith("__MACOSX/"):
        return True
    return os.path.basename(norm).startswith("._")


def parse_clip(basename):
    """Parse `<NN><SignName>Sinalizador<SS>-<rep>.mp4`.

    Returns dict(class_name, sign, signer, rep) or None if it does not match.
    class_name is the canonical `<NN><SignName>` (e.g. '01Acontecer').
    """
    m = _CLIP_RE.match(basename)
    if m is None:
        return None
    num, name, signer, rep = m.group(1), m.group(2), m.group(3), m.group(4)
    return {
        "class_name": "{}{}".format(num, name),
        "sign": name,
        "signer": int(signer),
        "rep": int(rep),
    }


def _iter_videos(signer_dir):
    for ext in _VIDEO_EXTS:
        for p in glob.iglob(os.path.join(signer_dir, "**", "*" + ext), recursive=True):
            if not _is_junk(p):
                yield p
        for p in glob.iglob(os.path.join(signer_dir, "**", "*" + ext.upper()), recursive=True):
            if not _is_junk(p):
                yield p


def scan_dataset(root):
    """Yield (video_path, class_name, signer, split_name) for every real clip.

    `root` is the extracted MINDS-Libras root containing `Sinalizador<SS>/` dirs.
    The signer/class come from the *filename* (authoritative); the split from the
    signer id. Junk (AppleDouble) files and unparseable names are skipped.
    """
    signer_dirs = sorted(
        d for d in glob.glob(os.path.join(root, "Sinalizador*"))
        if os.path.isdir(d) and not _is_junk(d)
    )
    seen = set()
    for signer_dir in signer_dirs:
        for vp in sorted(set(_iter_videos(signer_dir))):
            if vp in seen:
                continue
            seen.add(vp)
            info = parse_clip(os.path.basename(vp))
            if info is None:
                continue
            signer = info["signer"]
            if signer not in _SIGNER_TO_SPLIT:
                continue
            yield (vp, info["class_name"], signer, _SIGNER_TO_SPLIT[signer])


def discover_classes(root):
    """Sorted list of sign-class names observed across all signers on disk."""
    classes = set()
    for _vp, cls, _signer, _split in scan_dataset(root):
        classes.add(cls)
    return sorted(classes)


def summarize_splits(root):
    """Return split_name -> {'clips', 'classes' (set), 'signers' (set)}."""
    summary = defaultdict(lambda: {"clips": 0, "classes": set(), "signers": set()})
    for _vp, cls, signer, split_name in scan_dataset(root):
        b = summary[split_name]
        b["clips"] += 1
        b["classes"].add(cls)
        b["signers"].add(signer)
    return dict(summary)


def write_split_json(out_path):
    payload = {
        "schema": "minds-libras-signer-independent-v1",
        "description": (
            "Fixed signer-disjoint split of MINDS-Libras (Brazilian Sign Language, "
            "isolated glosses) for cross-lingual transfer evaluation (P1, 6th target "
            "language, replacing INCLUDE-50). 20 signs, 12 signers; "
            "train=signers 1-8, val=9-10, test=11-12. Signer id is the top-level "
            "Sinalizador<SS> directory (Zenodo record 2667329, one zip per signer)."
        ),
        "signer_dir_pattern": _SIGNER_DIR_RE.pattern,
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
    parser.add_argument("--root", required=True,
                        help="extracted MINDS-Libras root (has Sinalizador<SS>/ dirs)")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()

    summary = summarize_splits(args.root)
    print("{:<10} {:>6} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        b = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>6} {:>8} {:>8}".format(
            name, b["clips"], len(b["classes"]), len(b["signers"])
        ))
    print("total classes discovered: {}".format(len(discover_classes(args.root))))
    if args.write_json:
        write_split_json(args.write_json)
        print("wrote {}".format(args.write_json))


if __name__ == "__main__":
    _main()
