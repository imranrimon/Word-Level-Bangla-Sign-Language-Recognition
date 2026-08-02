"""Unit tests for preprocessing.bdslw401_meta.

All tests here are filesystem-agnostic - we construct tiny synthetic
directory trees in tmp_path because the real BdSLW401 download may or
may not be present when CI runs.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from preprocessing.bdslw401_meta import (
    parse_filename,
    scan_dataset,
    summarize,
)


def test_parse_standard_filename():
    assert parse_filename("W001S04F_02.mp4") == (1, 4, "F", 2)
    assert parse_filename("W401S57F_11.mp4") == (401, 57, "F", 11)


def test_parse_handles_missing_view_letter():
    assert parse_filename("W12S3_05.mp4") == (12, 3, "", 5)


def test_parse_rejects_malformed():
    for name in [
        "not-a-clip.mp4",
        "W001S04F.mp4",                # no trial
        "W001S04F_trial_02.mp4",       # BdSLW60 style - different dataset
        "U1W12F_trial_0_L.mp4",        # BdSLW60 filename
        "W001S04F_02.mov",             # wrong extension
    ]:
        assert parse_filename(name) is None, name


def _make_fake_dataset(tmp_path):
    """Make a small fixture matching BdSLW401's Front/Front/<split> layout."""
    layout = {
        "Front/Front/train": [
            "W001S01F_01.mp4",
            "W001S01F_02.mp4",
            "W002S01F_01.mp4",
            "W001S02F_01.mp4",
            "not_a_clip.mp4",          # should be skipped
        ],
        "Front/Front/val": [
            "W001S03F_01.mp4",
        ],
        "Front/Front/test": [
            "W002S04F_01.mp4",
            "W002S04F_02.mp4",
        ],
    }
    for rel_dir, files in layout.items():
        d = tmp_path / rel_dir
        d.mkdir(parents=True, exist_ok=True)
        for fn in files:
            (d / fn).write_bytes(b"")
    return tmp_path


def test_scan_dataset_yields_parseable_clips_only(tmp_path):
    root = _make_fake_dataset(tmp_path)
    clips = list(scan_dataset(str(root)))
    # 4 train parseable + 1 val + 2 test = 7. The 'not_a_clip.mp4' is skipped.
    assert len(clips) == 7
    for filepath, _word, _signer, _view, _asplit in clips:
        assert os.path.exists(filepath)


def test_scan_dataset_reports_author_split(tmp_path):
    root = _make_fake_dataset(tmp_path)
    clips = list(scan_dataset(str(root)))
    author_splits = {s for *_, s in clips}
    assert author_splits == {"train", "val", "test"}


def test_summarize_counts_are_correct(tmp_path):
    root = _make_fake_dataset(tmp_path)
    s = summarize(str(root))
    assert s["clips"] == 7
    assert s["words"] == {1, 2}
    assert s["signers"] == {1, 2, 3, 4}
    assert s["views"] == {"F"}
    assert dict(s["author_splits"]) == {"train": 4, "val": 1, "test": 2}
