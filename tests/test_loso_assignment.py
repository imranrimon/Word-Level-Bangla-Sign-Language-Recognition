"""Audit fix #2: smoke tests for the LOSO assignment helper.

Verifies that build_loso_assignment correctly partitions the 11 full-vocab
signers (train union val union test in the canonical SIGNER_SPLIT) into
train/val/test for one fold, and that the pretrain pool is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_loso_assignment_round_trip_every_full_signer():
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT
    from preprocessing.generate_loso_split_bundle import build_loso_assignment

    full = (set(SIGNER_SPLIT["train"])
            | set(SIGNER_SPLIT["val"])
            | set(SIGNER_SPLIT["test"]))
    pretrain = set(SIGNER_SPLIT["pretrain"])

    # For every choice of test signer, assignment must split into train + val + test
    # covering exactly the 11 full signers (plus pretrain unchanged).
    for ts in sorted(full):
        vs = next(iter(sorted(full - {ts})))
        assignment = build_loso_assignment(ts, vs)
        train = {s for s, sp in assignment.items() if sp == "train"}
        val   = {s for s, sp in assignment.items() if sp == "val"}
        test  = {s for s, sp in assignment.items() if sp == "test"}
        pre   = {s for s, sp in assignment.items() if sp == "pretrain"}
        assert test == {ts}, f"ts={ts}: test={test}"
        assert val == {vs}, f"ts={ts}: val={val}"
        assert train == full - {ts, vs}, f"ts={ts}: train={train}"
        assert pre == pretrain
        # No overlap, complete coverage of all 18 signers.
        all_assigned = train | val | test | pre
        assert all_assigned == full | pretrain


def test_loso_assignment_rejects_pretrain_pool_signer():
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT
    from preprocessing.generate_loso_split_bundle import build_loso_assignment

    pretrain = sorted(SIGNER_SPLIT["pretrain"])
    full = list(SIGNER_SPLIT["train"])

    with pytest.raises(ValueError, match="not in the 11 full-vocabulary"):
        build_loso_assignment(pretrain[0], full[0])

    with pytest.raises(ValueError, match="not in the 11 full-vocabulary"):
        build_loso_assignment(full[0], pretrain[0])


def test_loso_assignment_rejects_equal_test_and_val():
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT
    from preprocessing.generate_loso_split_bundle import build_loso_assignment

    s = sorted(SIGNER_SPLIT["train"])[0]
    with pytest.raises(ValueError, match="must differ"):
        build_loso_assignment(s, s)
