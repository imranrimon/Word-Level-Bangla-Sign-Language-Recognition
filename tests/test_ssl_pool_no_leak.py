"""Audit fix #4: SSL pool must not include BdSLW60-SI val or test signers.

If a bug in `_should_exclude` or `_SIGNER_RE` ever silently lets val/test
signers' clips into the pretraining pool, the SHuBERT encoder would learn
their pose distribution unsupervised and inflate downstream Top-1 on the
SI val/test sets. This test fails immediately if that ever happens.

Skips quietly when the manifest hasn't been built yet (CI on a fresh checkout).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data" / "ssl_pool_manifest.json"
SIGNER_RE = re.compile(r"U(\d+)W", re.IGNORECASE)


@pytest.mark.skipif(not MANIFEST.exists(), reason="SSL pool manifest not built")
def test_ssl_pool_excludes_si_val_test_signers():
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT

    held_out = set(SIGNER_SPLIT["val"]) | set(SIGNER_SPLIT["test"])

    with open(MANIFEST) as f:
        manifest = json.load(f)

    leaks = []
    for entry in manifest["clips"]:
        m = SIGNER_RE.search(os.path.basename(entry["path"]))
        if m and int(m.group(1)) in held_out:
            leaks.append(entry["path"])

    assert not leaks, (
        f"SSL pool leaks {len(leaks)} clips from BdSLW60-SI held-out signers "
        f"{sorted(held_out)}. First 5: {leaks[:5]}"
    )


@pytest.mark.skipif(not MANIFEST.exists(), reason="SSL pool manifest not built")
def test_ssl_pool_per_source_counts_match_actual():
    """The manifest's recorded per-source counts must equal a fresh recount.

    Catches any drift between the cached counts (written at build time) and
    the actual clip list (which could grow if someone appended manually).
    """
    with open(MANIFEST) as f:
        manifest = json.load(f)
    recorded = manifest.get("per_source_counts", {})
    actual = {}
    for entry in manifest["clips"]:
        actual[entry["source"]] = actual.get(entry["source"], 0) + 1
    for source, n in recorded.items():
        assert actual.get(source, 0) == n, (
            f"{source}: per_source_counts={n} but actual={actual.get(source, 0)}"
        )


@pytest.mark.skipif(not MANIFEST.exists(), reason="SSL pool manifest not built")
def test_ssl_pool_manifest_excluded_signers_field_is_consistent():
    """If the manifest claims it excluded signers, those exact signers must
    not appear anywhere in the clip list — defends against the case where
    the user re-ran build_ssl_pool_manifest.py with --no-exclude-si-val-test
    but forgot to also update the comment / metadata field.
    """
    with open(MANIFEST) as f:
        manifest = json.load(f)
    excluded = set(manifest.get("excluded_signers", []))
    if not excluded:
        return  # nothing claimed; the other test covers this case
    for entry in manifest["clips"]:
        m = SIGNER_RE.search(os.path.basename(entry["path"]))
        if m:
            sid = int(m.group(1))
            assert sid not in excluded, (
                f"clip {entry['path']} has signer U{sid}, which is listed in "
                f"manifest['excluded_signers']={sorted(excluded)}"
            )
