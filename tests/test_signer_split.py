"""Tests for the signer-independent split helper.

Unit tests run without the dataset present. Integration tests run against
the real BdSLW60 directory if it is available, otherwise skip.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from preprocessing.bdsl_signer_split import (
    ALL_SPLITS,
    LABELED_SPLITS,
    SIGNER_SPLIT,
    all_signers,
    classify_signer,
    find_unparseable,
    parse_filename,
    scan_dataset,
    summarize_splits,
    write_split_json,
)


DATASET_ROOT = ROOT / "Word_level_Bangla_Sign_Language_Dataset" / "BdSLW30"
SPLITS_JSON = ROOT / "splits" / "signer_independent.json"

needs_dataset = pytest.mark.skipif(
    not DATASET_ROOT.exists(),
    reason="BdSLW60 dataset folder not present",
)


# ----- unit: parsing --------------------------------------------------------

def test_parse_standard_filename():
    assert parse_filename("U10W37F_trial_0_L.mp4") == (10, 37, 0, "L")
    assert parse_filename("U1W357F_trial_13_R.mp4") == (1, 357, 13, "R")


def test_parse_accepts_optional_session_infix():
    # 30 'dengue' clips carry an extra _<session>_ component
    assert parse_filename("U1W357F_1_trial_0_L.mp4") == (1, 357, 0, "L")
    assert parse_filename("U1W357F_2_trial_5_R.mp4") == (1, 357, 5, "R")


def test_parse_accepts_fl_suffix_variant():
    # 10 clips use 'FL' instead of plain 'F'
    assert parse_filename("U10W37FL_trial_0_L.mp4") == (10, 37, 0, "L")


def test_parse_rejects_malformed():
    for name in [
        "not-a-clip.mp4",
        "U10W37F_0_L.mp4",         # missing 'trial'
        "U10W37F_trial_X_L.mp4",   # non-numeric trial
        "UaaW37F_trial_0_L.mp4",   # non-numeric signer
        "U10W37F_trial_0_L.mov",   # wrong extension
    ]:
        assert parse_filename(name) is None, name


# ----- unit: split invariants ----------------------------------------------

def test_all_18_signers_assigned_exactly_once():
    signers = all_signers()
    assert signers == list(range(1, 19)), signers


def test_splits_are_pairwise_disjoint():
    seen = {}
    for split_name, signers in SIGNER_SPLIT.items():
        for s in signers:
            assert s not in seen, "U{:02d} appears in {} and {}".format(
                s, seen[s], split_name
            )
            seen[s] = split_name


def test_labeled_splits_do_not_overlap_pretrain():
    labeled = set()
    for s in LABELED_SPLITS:
        labeled.update(SIGNER_SPLIT[s])
    pretrain = set(SIGNER_SPLIT["pretrain"])
    assert labeled.isdisjoint(pretrain)


def test_classify_signer_for_every_id():
    for s in all_signers():
        split = classify_signer(s)
        assert split in ALL_SPLITS
    with pytest.raises(ValueError):
        classify_signer(99)


# ----- unit: JSON artifact round-trip ---------------------------------------

def test_committed_split_json_matches_code():
    assert SPLITS_JSON.exists(), str(SPLITS_JSON)
    with open(SPLITS_JSON) as f:
        payload = json.load(f)
    for name, expected in SIGNER_SPLIT.items():
        assert payload["splits"][name] == list(expected)


def test_write_split_json_round_trip(tmp_path):
    out = tmp_path / "splits.json"
    write_split_json(str(out))
    with open(out) as f:
        payload = json.load(f)
    assert payload["schema"] == "bdslw60-signer-independent-v1"
    for name, expected in SIGNER_SPLIT.items():
        assert payload["splits"][name] == list(expected)


# ----- integration: real dataset -------------------------------------------

@needs_dataset
def test_scan_dataset_counts_match_expected():
    summary = summarize_splits(str(DATASET_ROOT))

    # All labeled splits must cover every one of the 60 classes.
    for name in LABELED_SPLITS:
        assert len(summary[name]["classes"]) == 60, (
            "split '{}' covers {} classes, expected 60".format(
                name, len(summary[name]["classes"])
            )
        )

    # Exact clip counts (measured from the committed dataset snapshot).
    expected_clips = {"train": 5748, "val": 655, "test": 1365, "pretrain": 1539}
    for name, expected in expected_clips.items():
        assert summary[name]["clips"] == expected, (
            "split '{}' has {} clips, expected {}".format(
                name, summary[name]["clips"], expected
            )
        )
    # The four splits must partition the 9307 parseable clips exactly.
    assert sum(summary[n]["clips"] for n in ALL_SPLITS) == 9307


@needs_dataset
def test_unparseable_files_are_the_known_dengue_variants():
    unparseable = find_unparseable(str(DATASET_ROOT))
    # The 30 known edge cases all live in the 'dengue' class.
    assert len(unparseable) == 0, (
        "parser should now accept all BdSLW60 filenames; "
        "unexpected unparseable files: {}".format(unparseable[:5])
    )


@needs_dataset
def test_scan_splits_partition_the_dataset():
    by_clip = {}
    for filepath, class_name, signer, split in scan_dataset(str(DATASET_ROOT)):
        key = filepath
        assert key not in by_clip, "clip appeared in scan twice: {}".format(key)
        by_clip[key] = (class_name, signer, split)
    # Every scanned clip is in exactly one of ALL_SPLITS.
    for _, _, split in by_clip.values():
        assert split in ALL_SPLITS
