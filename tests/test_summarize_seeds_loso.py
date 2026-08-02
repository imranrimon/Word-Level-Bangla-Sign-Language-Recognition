"""Smoke test that summarize_seeds.py groups LOSO runs across both
seed and test-signer axes (audit fix #2 dependency).

A run named 'foo_loso_test02_seed1' should collapse to base 'foo' so the
aggregator's mean/std spans both axes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_base_name_strips_loso_and_seed():
    from tools.summarize_seeds import _base_name, _seed_id, _loso_test_signer

    cases = {
        "bdsl_block_gcn_si":                        ("bdsl_block_gcn_si", None, None),
        "bdsl_block_gcn_si_seed1":                   ("bdsl_block_gcn_si", 1, None),
        "bdsl_block_gcn_si_loso_test02":             ("bdsl_block_gcn_si", None, 2),
        "bdsl_block_gcn_si_loso_test13_seed0":       ("bdsl_block_gcn_si", 0, 13),
        "bdsl_block_gcn_shubert_bdsl_asl_seed2":     ("bdsl_block_gcn_shubert_bdsl_asl", 2, None),
        "bhc_lora_bdsl47_digits_seed0":              ("bhc_lora_bdsl47_digits", 0, None),
    }
    for name, (expect_base, expect_seed, expect_loso) in cases.items():
        assert _base_name(name) == expect_base, \
            f"{name!r}: base_name -> {_base_name(name)!r} != {expect_base!r}"
        assert _seed_id(name) == expect_seed, \
            f"{name!r}: seed -> {_seed_id(name)} != {expect_seed}"
        assert _loso_test_signer(name) == expect_loso, \
            f"{name!r}: loso -> {_loso_test_signer(name)} != {expect_loso}"


def test_base_name_none_safe():
    from tools.summarize_seeds import _base_name, _seed_id, _loso_test_signer

    assert _base_name(None) is None
    assert _seed_id(None) is None
    assert _loso_test_signer(None) is None
