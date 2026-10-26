"""Build the SSL pretraining pool manifest: a flat list of pose-cache clip paths.

Takes one or more pose-cache directories and writes a single JSON manifest that
the SSL feeder iterates during SHuBERT-style masked pretraining.

Usage (run after each cache-producing preprocessing step):

    python preprocessing/build_ssl_pool_manifest.py \
        --cache-dirs \
            data/bdsl_cache/pretrain \
            data/bdslw102_a_pose_cache \
            data/bdslw401_pose_cache_front \
        --output data/ssl_pool_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


_SIGNER_RE = re.compile(r"U(\d+)W", re.IGNORECASE)


def _walk_npz(root):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".npz"):
                yield os.path.join(dirpath, fn).replace("\\", "/")


def _should_exclude(path, excluded_signers):
    """For BdSLW60-format paths (U<signer>W<word>...), exclude if signer is held-out."""
    if not excluded_signers:
        return False
    m = _SIGNER_RE.search(os.path.basename(path))
    if m is None:
        return False            # path does not encode a signer ID (BdSLW401 / BdSLW102_A format)
    return int(m.group(1)) in excluded_signers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--min-frames",
        type=int,
        default=8,
        help="skip clips whose pose cache has fewer frames than this",
    )
    ap.add_argument(
        "--exclude-si-val-test",
        action="store_true",
        default=True,
        help="drop clips whose signer ID is in the BdSLW60-SI val or test split "
             "(prevents SSL encoder from memorising held-out signers' pose "
             "distribution). On by default.",
    )
    ap.add_argument(
        "--no-exclude-si-val-test", dest="exclude_si_val_test", action="store_false",
        help="disable the signer exclusion filter (not recommended for SSL pretraining)",
    )
    args = ap.parse_args()

    excluded = set()
    if args.exclude_si_val_test:
        try:
            from preprocessing.bdsl_signer_split import SIGNER_SPLIT
        except Exception as exc:
            print(f"[WARN] cannot import SIGNER_SPLIT ({exc}); skipping exclusion")
        else:
            excluded = set(SIGNER_SPLIT["val"]) | set(SIGNER_SPLIT["test"])
            print(f"excluding BdSLW60 signers: {sorted(excluded)}")

    entries = []
    per_source = {}
    per_source_excluded = {}
    for root in args.cache_dirs:
        if not os.path.exists(root):
            print(f"skip missing: {root}")
            continue
        raw_paths = list(_walk_npz(root))
        kept = 0
        dropped = 0
        for p in raw_paths:
            if _should_exclude(p, excluded):
                dropped += 1
                continue
            entries.append({"path": p, "source": root})
            kept += 1
        per_source[root] = kept
        if dropped:
            per_source_excluded[root] = dropped
    print(f"total clips: {len(entries)}")
    for k, v in per_source.items():
        extra = f" (+{per_source_excluded[k]} excluded)" if k in per_source_excluded else ""
        print(f"  {k}: {v}{extra}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "clips": entries,
            "per_source_counts": per_source,
            "per_source_excluded": per_source_excluded,
            "excluded_signers": sorted(excluded),
        }, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
