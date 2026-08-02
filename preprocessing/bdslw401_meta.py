"""BdSLW401 filename parser and dataset scanner.

BdSLW401 (arXiv:2503.02360, Kaggle `hasanssl/bdslw401`) contains ~102k
multi-view Bangla Sign Language clips covering 401 words. Directory layout
(from the Kaggle release, verified 2026-04):

    <root>/
        Front/Front/<split>/W<word>S<signer>F_<trial>.mp4
        [likely] Side/...     (multi-view; to be confirmed after unzip)

  * `W<word>` : three-digit word id (1..401)
  * `S<signer>`: signer id
  * `F`       : view flag (matches `Front/` prefix)
  * `_<trial>`: two-digit zero-padded trial index

For the Option-C pretraining stage we don't need the authors' train/val/
test split - all clips become a single unlabeled pool feeding the
SHuBERT-style masked code-prediction task. This module therefore parses
filenames and enumerates clips, but does not define a labeled split of
its own.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict


_FILENAME_RE = re.compile(
    r"^W(\d+)S(\d+)([A-Z]*)_(\d+)\.mp4$",
    re.IGNORECASE,
)


def parse_filename(name):
    """Return (word:int, signer:int, view:str, trial:int) or None.

    `view` is the letter suffix after the signer id ('F' for frontal in
    the released data; left as the raw string so future view labels
    don't silently disappear).
    """
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3).upper(), int(m.group(4))


def scan_dataset(root):
    """Walk `root` recursively and yield (filepath, word, signer, view, author_split).

    `author_split` is the directory name one level above the clip
    (typically 'train', 'val', or 'test' in the Kaggle release), or the
    empty string if we can't determine one. It is informational only -
    the SSL pretraining pool uses every clip regardless.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".mp4"):
                continue
            parsed = parse_filename(fn)
            if parsed is None:
                continue
            word, signer, view, trial = parsed
            parent = os.path.basename(os.path.normpath(dirpath)).lower()
            author_split = parent if parent in ("train", "val", "test") else ""
            yield (
                os.path.join(dirpath, fn),
                word,
                signer,
                view,
                author_split,
            )


def summarize(root):
    """Aggregate counts over the whole dataset.

    Returns:
      {'clips': int, 'words': set[int], 'signers': set[int],
       'views': set[str], 'author_splits': {name: int}}
    """
    result = {
        "clips": 0,
        "words": set(),
        "signers": set(),
        "views": set(),
        "author_splits": defaultdict(int),
    }
    for _fp, w, s, v, asplit in scan_dataset(root):
        result["clips"] += 1
        result["words"].add(w)
        result["signers"].add(s)
        result["views"].add(v or "<none>")
        result["author_splits"][asplit or "<none>"] += 1
    return result


def _main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="BdSLW401 root directory")
    args = ap.parse_args()

    s = summarize(args.root)
    print("clips:   {}".format(s["clips"]))
    print("words:   {} (range {}..{})".format(
        len(s["words"]),
        min(s["words"]) if s["words"] else "-",
        max(s["words"]) if s["words"] else "-",
    ))
    print("signers: {} (range {}..{})".format(
        len(s["signers"]),
        min(s["signers"]) if s["signers"] else "-",
        max(s["signers"]) if s["signers"] else "-",
    ))
    print("views:   {}".format(sorted(s["views"])))
    print("author_splits: {}".format(dict(s["author_splits"])))


if __name__ == "__main__":
    _main()
