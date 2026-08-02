"""Regression test for the BdSLW401 / BdSLW102_A bundlers.

We don't run the full bundler against the real cache (slow), but exercise
the filename parser and the pose-bundling primitive against tiny synthetic
fixtures so the file-format contract doesn't silently break.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# BdSLW401 — filename parser
# ---------------------------------------------------------------------------

def test_bdslw401_filename_parser_canonical():
    from preprocessing.bundle_bdslw401_pose_to_npy import _parse_npz

    assert _parse_npz("W001S04F_02.npz") == (1, 4, "F", 2)
    assert _parse_npz("W401S25F_99.npz") == (401, 25, "F", 99)
    # Case insensitive.
    assert _parse_npz("w042s07f_05.NPZ") == (42, 7, "F", 5)


def test_bdslw401_filename_parser_rejects_garbage():
    from preprocessing.bundle_bdslw401_pose_to_npy import _parse_npz

    assert _parse_npz("not_a_clip.mp4") is None
    assert _parse_npz("W01.npz") is None
    assert _parse_npz("U10W37F_trial_0_L.npz") is None    # BdSLW60 format


def test_bdslw401_bundler_end_to_end_on_tmp_fixture(tmp_path):
    from preprocessing.bundle_bdslw401_pose_to_npy import _bundle_split

    # Synthetic mini-cache: 3 clips for 2 classes, all train.
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for name in ("W001S01F_01.npz", "W001S02F_01.npz", "W002S01F_01.npz"):
        np.savez_compressed(train_dir / name,
                            data=np.random.randn(3, 30, 27, 1).astype(np.float32))

    classes_to_idx = {"W001": 0, "W002": 1}
    large, (names, labels) = _bundle_split(train_dir, "train", max_frames=60,
                                            classes_to_idx=classes_to_idx)
    assert large.shape == (3, 3, 60, 27, 1)
    assert names == ["W001S01F_01.mp4", "W001S02F_01.mp4", "W002S01F_01.mp4"]
    assert labels == [0, 0, 1]
    # First-frame values copied through (rest zero-padded).
    assert (large[:, :, 30:, :, :] == 0).all()


# ---------------------------------------------------------------------------
# BdSLW102_A — sentence-cache filename parser
# ---------------------------------------------------------------------------

def test_bdslw102a_sentence_filename_parser():
    from preprocessing.bundle_bdslw102a_sentence_pose_to_npy import _parse_npz

    assert _parse_npz("0_sentence1_withBg.npz") == (0, 1, "withbg")
    assert _parse_npz("12_sentence5_withoutBg.npz") == (12, 5, "withoutbg")
    assert _parse_npz("3_sentence7_raw.npz") == (3, 7, "raw")
    # Missing variant suffix -> 'raw'.
    assert _parse_npz("4_sentence2.npz") == (4, 2, "raw")
    assert _parse_npz("bogus.npz") is None


# ---------------------------------------------------------------------------
# BdSLW401 word-name extractor
# ---------------------------------------------------------------------------

def test_extract_split_columns_canonical():
    from preprocessing.extract_bdslw401_word_names import _split_columns

    # Bangla token + 1 romanized + multi-token english.
    row = _split_columns("W007 চাচী Chachi Wife of Paternal Uncle")
    assert row["id"] == "W007"
    assert "চাচী" in row["bangla"]
    assert row["romanized"] == "Chachi"
    assert row["english"] == "Wife of Paternal Uncle"


def test_extract_split_columns_slash_joined_romanized():
    from preprocessing.extract_bdslw401_word_names import _split_columns

    row = _split_columns("W001 বাবা /আব্বা Baba/Abba Father")
    assert row["id"] == "W001"
    assert row["romanized"] == "Baba/Abba"
    assert row["english"] == "Father"


def test_extract_normalise_id():
    from preprocessing.extract_bdslw401_word_names import _normalise_id

    assert _normalise_id("W1") == "W001"
    assert _normalise_id("w007") == "W007"
    assert _normalise_id("W401") == "W401"
