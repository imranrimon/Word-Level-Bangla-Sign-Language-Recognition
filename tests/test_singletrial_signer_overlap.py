"""Audit fix #7: characterize BdSL60-SingleTrial signers vs the SI split.

The cross-recording-robustness claim (T6) hinges on the eval set containing
signers the model has NOT seen during training. If SingleTrial also contains
signers from the SI train set {1,4,5,6,8,9,11,12}, then T6 is conflating two
distinct factors:

  * cross-recording robustness (what we want to measure), and
  * seen-vs-unseen signer (what Stage A already measures).

This test characterizes the actual overlap so the paper can either (a) filter
SingleTrial to held-out signers only before reporting T6, or (b) explicitly
report two T6 numbers — one per regime — for an honest comparison.

Skips quietly when the eval bundle hasn't been built yet.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABEL_PKL = ROOT / "data" / "bdsl60_singletrial_eval" / "eval_label.pkl"
SIGNER_RE = re.compile(r"U(\d+)W", re.IGNORECASE)


def _parse_signer(fn):
    m = SIGNER_RE.search(fn)
    return int(m.group(1)) if m else None


@pytest.mark.skipif(not LABEL_PKL.exists(), reason="SingleTrial eval bundle not built")
def test_singletrial_signers_documented():
    """Print per-bucket signer breakdown and assert SingleTrial is non-trivial."""
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT

    si_train = set(SIGNER_SPLIT["train"])
    si_val_test = set(SIGNER_SPLIT["val"]) | set(SIGNER_SPLIT["test"])
    si_pretrain = set(SIGNER_SPLIT["pretrain"])

    with open(LABEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    # Bundle is (filenames, labels) — both are lists of equal length.
    if isinstance(bundle, tuple) and len(bundle) == 2:
        names = bundle[0]
    elif isinstance(bundle, dict) and "filenames" in bundle:
        names = bundle["filenames"]
    else:
        raise RuntimeError(f"unexpected label.pkl shape: {type(bundle)}")

    per_signer = {}
    for n in names:
        s = _parse_signer(str(n))
        if s is None:
            continue
        per_signer[s] = per_signer.get(s, 0) + 1

    signers = set(per_signer.keys())
    overlap_train = signers & si_train
    overlap_val_test = signers & si_val_test
    overlap_pretrain = signers & si_pretrain
    unknown = signers - si_train - si_val_test - si_pretrain

    print(f"\nBdSL60-SingleTrial signer breakdown:")
    print(f"  total clips: {sum(per_signer.values())}")
    print(f"  unique signers: {sorted(signers)}")
    print(f"  in SI train     {sorted(si_train)}: {sorted(overlap_train)} "
          f"({sum(per_signer[s] for s in overlap_train)} clips)")
    print(f"  in SI val/test  {sorted(si_val_test)}: {sorted(overlap_val_test)} "
          f"({sum(per_signer[s] for s in overlap_val_test)} clips)")
    print(f"  in SI pretrain  {sorted(si_pretrain)}: {sorted(overlap_pretrain)} "
          f"({sum(per_signer[s] for s in overlap_pretrain)} clips)")
    if unknown:
        print(f"  UNKNOWN: {sorted(unknown)} "
              f"({sum(per_signer[s] for s in unknown)} clips)")

    if overlap_train:
        print(
            "\n[METHODOLOGY] T6 measured on the full SingleTrial set will conflate "
            "cross-recording robustness with cross-signer generalization. For a "
            "clean cross-recording-only measurement, filter eval clips to "
            f"signers NOT in SI train {sorted(si_train)}."
        )

    # Hard floor: SingleTrial must span at least 3 distinct signers to be a
    # meaningful cross-recording test set.
    assert len(signers) >= 3, (
        f"SingleTrial spans only {len(signers)} signers — too few for a "
        f"defensible cross-recording eval"
    )


@pytest.mark.skipif(not LABEL_PKL.exists(), reason="SingleTrial eval bundle not built")
def test_singletrial_held_out_subset_is_usable():
    """The subset of SingleTrial clips whose signers are NOT in SI train must
    be non-empty and span enough classes to compute Top-1 meaningfully.

    This is the subset T6 should ACTUALLY be reported on. If it's tiny, the
    paper either uses the full set (with caveat) or skips T6.
    """
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT

    si_train = set(SIGNER_SPLIT["train"])

    with open(LABEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, tuple) and len(bundle) == 2:
        names, labels = bundle
    elif isinstance(bundle, dict):
        names, labels = bundle["filenames"], bundle["labels"]
    else:
        raise RuntimeError(f"unexpected label.pkl shape: {type(bundle)}")

    held_out_clips = []
    held_out_classes = set()
    for name, label in zip(names, labels):
        s = _parse_signer(str(name))
        if s is not None and s not in si_train:
            held_out_clips.append(name)
            held_out_classes.add(int(label))

    print(f"\nHeld-out subset of SingleTrial (signer NOT in SI train):")
    print(f"  clips: {len(held_out_clips)}")
    print(f"  classes covered: {len(held_out_classes)} of 60")

    assert len(held_out_clips) >= 50, (
        f"only {len(held_out_clips)} clips in the held-out subset — "
        f"T6 numbers would be too noisy"
    )
    assert len(held_out_classes) >= 30, (
        f"held-out subset covers only {len(held_out_classes)}/60 classes — "
        f"insufficient coverage for a per-class Top-1 claim"
    )
