"""Build a vocabulary alignment table across Bangla SLR datasets.

Reads each dataset's class list (a list of word names — typically the
folder names under the dataset root) and produces a JSON alignment table
that pairs equivalent words across BdSLW60, BdSLW401, and BdSLW102_A.

Three matching strategies (cumulative):
  1. Exact string match (case-insensitive, whitespace-normalized).
  2. Bangla transliteration match — if class names are mixed Bangla /
     ASCII transliteration, normalize via lower-case + strip diacritics.
  3. Manual curator pass — output stub rows where source and target are
     ambiguous matches, sorted by Levenshtein distance for review.

The output JSON has three sections per source-target pair:
  * `exact`        : confirmed matches (auto-accepted)
  * `candidates`   : near-matches sorted by edit distance (review needed)
  * `unmatched`    : source classes with no candidate above threshold

Usage:

    python preprocessing/build_bangla_vocab_alignment.py \\
        --classes bdslw60=data/bdsl_si/classes.json \\
        --classes bdslw401=data/bdslw401_classes.json \\
        --classes bdslw102a=data/bdslw102a_classes.json \\
        --output  data/bangla_vocab_alignment.json

If you don't yet have the per-dataset classes.json files, run with
--scan to derive them from the raw data root (one class per top-level
directory):

    python preprocessing/build_bangla_vocab_alignment.py \\
        --scan bdslw401=data/bdslw401_raw/Front/Front/train \\
        --scan bdslw102a=data/bdslw102_a_raw/...

The script also computes pairwise Jaccard overlap and prints a coverage
report so you can spot-check before launching cross-dataset training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize_for_exact(s):
    """Lower-case + strip + collapse whitespace + NFC for Bangla composition."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def _is_bangla(s):
    """Heuristic: any char in the Bangla Unicode block (U+0980..U+09FF)?"""
    return any(0x0980 <= ord(c) <= 0x09FF for c in s)


def _strip_ascii_diacritics(s):
    """For transliterated ASCII names: lower-case + strip diacritics."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return _normalize_for_exact(s)


def _levenshtein(a, b):
    """Compute Levenshtein distance — small inputs only (word names)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        curr = [i]
        for j, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Class-list loading
# ---------------------------------------------------------------------------

def _load_classes(spec):
    """Spec is 'name=path' where path is .json with {classes: [...]} or .txt
    with one class per line. Returns (name, [class_names])."""
    if "=" not in spec:
        raise ValueError(f"--classes spec must be name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes") if isinstance(data, dict) else data
    else:
        with open(path, encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError(f"no classes loaded from {path}")
    return name.strip(), [str(c) for c in classes]


def _scan_class_root(spec):
    """Spec is 'name=root_dir'; class names = top-level subdirectory names."""
    if "=" not in spec:
        raise ValueError(f"--scan spec must be name=root, got {spec!r}")
    name, root = spec.split("=", 1)
    root = root.strip()
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    return name.strip(), classes


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _candidates(src_norm, tgt_norms, max_dist=3, top_k=3):
    """Find top-k tgt entries closest to src by edit distance, up to max_dist."""
    distances = [(name, _levenshtein(src_norm, n)) for name, n in tgt_norms.items()]
    distances = [(n, d) for n, d in distances if d <= max_dist]
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]


def _norm_variants(orig):
    """Return all normalized surface forms for an entry, including slash-
    separated alternatives. 'Baba/Abba' -> ['baba/abba', 'baba', 'abba'].
    """
    base = _normalize_for_exact(orig) if _is_bangla(orig) else _strip_ascii_diacritics(orig)
    variants = {base}
    if "/" in base:
        for part in base.split("/"):
            part = part.strip()
            if part:
                variants.add(part)
    return variants


def _align_pair(src_name, src_classes, tgt_name, tgt_classes,
                edit_threshold=2):
    """Build alignment from src to tgt. Returns dict with three sections.

    Both source and target vocabularies are expanded over slash-joined
    aliases (e.g. 'Baba/Abba' produces lookup entries 'baba', 'abba', and
    'baba/abba'), so a source 'baba' exact-matches the target 'Baba/Abba'.
    """
    # Normalize source list (just keep one normalized form per entry).
    src_norm_to_orig = {}
    for orig in src_classes:
        for v in _norm_variants(orig):
            src_norm_to_orig.setdefault(v, orig)

    # Normalize target with slash-variant expansion in the lookup.
    tgt_norm_to_orig = {}
    for orig in tgt_classes:
        for v in _norm_variants(orig):
            tgt_norm_to_orig.setdefault(v, orig)

    exact = {}
    candidates = {}
    unmatched = []

    for src_norm, src_orig in src_norm_to_orig.items():
        if src_norm in tgt_norm_to_orig:
            exact[src_orig] = tgt_norm_to_orig[src_norm]
            continue
        near = _candidates(src_norm, tgt_norm_to_orig, max_dist=edit_threshold)
        if near:
            candidates[src_orig] = [
                {"target": tgt_norm_to_orig[n], "edit_distance": d}
                for n, d in near
            ]
        else:
            unmatched.append(src_orig)

    return {
        "source": src_name,
        "target": tgt_name,
        "n_source": len(src_classes),
        "n_target": len(tgt_classes),
        "n_exact": len(exact),
        "n_candidates": len(candidates),
        "n_unmatched": len(unmatched),
        "jaccard_lower_bound": len(exact) / max(1, len(src_classes) + len(tgt_classes) - len(exact)),
        "exact": exact,
        "candidates": candidates,
        "unmatched": unmatched,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--classes", action="append", default=[],
        help="name=path/to/classes.json (or .txt). Repeatable.",
    )
    ap.add_argument(
        "--scan", action="append", default=[],
        help="name=path/to/class-folder-root for auto-discovery. Repeatable.",
    )
    ap.add_argument("--output", required=True,
                    help="JSON path for the alignment table.")
    ap.add_argument(
        "--edit-threshold", type=int, default=2,
        help="max edit distance for candidate matches (default 2)",
    )
    args = ap.parse_args()

    datasets = {}
    for spec in args.classes:
        name, classes = _load_classes(spec)
        datasets[name] = classes
    for spec in args.scan:
        name, classes = _scan_class_root(spec)
        datasets[name] = classes

    if len(datasets) < 2:
        raise SystemExit("provide at least 2 datasets via --classes or --scan")

    print(f"loaded {len(datasets)} datasets:")
    for name in sorted(datasets):
        print(f"  {name}: {len(datasets[name])} classes")

    # Pairwise alignment.
    alignments = {}
    for src, tgt in combinations(sorted(datasets), 2):
        key = f"{src}__to__{tgt}"
        alignments[key] = _align_pair(
            src, datasets[src], tgt, datasets[tgt],
            edit_threshold=args.edit_threshold,
        )
        # Also compute reverse.
        rkey = f"{tgt}__to__{src}"
        alignments[rkey] = _align_pair(
            tgt, datasets[tgt], src, datasets[src],
            edit_threshold=args.edit_threshold,
        )

    out = {
        "datasets": {name: classes for name, classes in datasets.items()},
        "alignments": alignments,
        "edit_threshold": args.edit_threshold,
        "instructions": (
            "Review the 'candidates' section of each alignment pair and either "
            "promote a candidate to 'exact' (if you confirm semantic equivalence) "
            "or move the source class to 'unmatched'. Re-running this script will "
            "preserve your decisions if you save them into a separate "
            "<output>_curated.json file that downstream tools consume."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {args.output}")
    print("\n=== Pairwise alignment summary ===")
    for key, info in alignments.items():
        print(
            f"  {key}:  exact={info['n_exact']}, "
            f"candidates={info['n_candidates']}, "
            f"unmatched={info['n_unmatched']}, "
            f"jaccard >= {info['jaccard_lower_bound']:.3f}"
        )


if __name__ == "__main__":
    main()
