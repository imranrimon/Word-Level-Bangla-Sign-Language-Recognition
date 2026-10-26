"""The SSL pretraining pools must not contain BdSLW60 validation or test signers.

If a clip from a held-out BdSLW60 signer entered a pool, the encoder would see that signer's
pose distribution during pretraining and the BdSL fine-tune numbers would be optimistic.
The P1 pools are ``data/ssl_pool_manifest_bdsl_asl.json`` (v1) and
``data/ssl_pool_manifest_bdsl_asl_v2.json`` (v2, geometry-checked WLASL). Each test skips when
the manifest has not been built (fresh checkout, CI).
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

MANIFESTS = [ROOT / "data" / n for n in ("ssl_pool_manifest_bdsl_asl.json", "ssl_pool_manifest_bdsl_asl_v2.json")]
SIGNER_RE = re.compile(r"U(\d+)W", re.IGNORECASE)


def _bdslw60_entries(manifest):
    """Clips that come from BdSLW60 (file names U<signer>W<word>...)."""
    for entry in manifest["clips"]:
        src = (entry.get("source", "") + entry["path"]).lower()
        if "bdsl_si" in src or "bdsl_cache" in src or "bdslw60" in src:
            yield entry


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_pool_excludes_bdslw60_val_test_signers(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT

    held_out = set(SIGNER_SPLIT["val"]) | set(SIGNER_SPLIT["test"])
    manifest = json.load(open(path))
    leaks = []
    for e in _bdslw60_entries(manifest):
        m = SIGNER_RE.search(os.path.basename(e["path"]))
        if m and int(m.group(1)) in held_out:
            leaks.append(e["path"])
    assert not leaks, f"{path.name} leaks {len(leaks)} clips of held-out signers {sorted(held_out)}: {leaks[:5]}"


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.name)
def test_pool_per_source_counts_match_clip_list(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    manifest = json.load(open(path))
    actual = {}
    for entry in manifest["clips"]:
        actual[entry["source"]] = actual.get(entry["source"], 0) + 1
    for source, n in manifest.get("per_source_counts", {}).items():
        assert actual.get(source, 0) == n, f"{source}: per_source_counts={n} but clip list has {actual.get(source, 0)}"
