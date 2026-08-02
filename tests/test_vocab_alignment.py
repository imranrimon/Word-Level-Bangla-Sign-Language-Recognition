"""Smoke tests for preprocessing/build_bangla_vocab_alignment.py.

These don't run the CLI — they exercise the alignment primitives so the
edit-distance + Bangla-Unicode + transliteration normalisation behaves
the way the cross-dataset paper assumes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_normalize_for_exact_handles_bangla_and_ascii():
    from preprocessing.build_bangla_vocab_alignment import _normalize_for_exact

    assert _normalize_for_exact("ABBU") == "abbu"
    assert _normalize_for_exact("  amma  ") == "amma"
    assert _normalize_for_exact("baba") == "baba"
    # NFC composition: pre-composed vs decomposed should match.
    assert _normalize_for_exact("আ") == _normalize_for_exact("আ")


def test_levenshtein_basic():
    from preprocessing.build_bangla_vocab_alignment import _levenshtein

    assert _levenshtein("abbu", "abbu") == 0
    assert _levenshtein("abbu", "abba") == 1
    assert _levenshtein("amma", "amama") == 1
    assert _levenshtein("", "abbu") == 4
    # Optimal alignment: "_amma__" -> "fath_er" anchors on the matching 'a',
    # giving 5 ops (insert f, sub m->t, sub m->h, sub a->e, insert r).
    assert _levenshtein("amma", "father") == 5


def test_align_pair_exact_and_candidates():
    from preprocessing.build_bangla_vocab_alignment import _align_pair

    src_classes = ["amma", "abbu", "book", "water"]
    tgt_classes = ["amma", "abba", "table", "watter"]
    result = _align_pair("src", src_classes, "tgt", tgt_classes,
                         edit_threshold=2)
    assert result["n_source"] == 4
    assert result["n_target"] == 4
    # 'amma' is an exact match.
    assert "amma" in result["exact"]
    assert result["exact"]["amma"] == "amma"
    # 'abbu' has 'abba' as a candidate (distance 1).
    assert "abbu" in result["candidates"]
    assert any(c["target"] == "abba" for c in result["candidates"]["abbu"])
    # 'water' has 'watter' as a candidate (distance 1).
    assert "water" in result["candidates"]
    # 'book' has no near candidate at edit_threshold=2.
    assert "book" in result["unmatched"]


def test_align_pair_handles_slash_joined_target_aliases():
    """BdSLW401 has slash-joined entries like 'Baba/Abba'. Source 'baba'
    must exact-match such a target, not fall to candidates.
    """
    from preprocessing.build_bangla_vocab_alignment import _align_pair

    src_classes = ["baba", "dada", "dadi"]
    tgt_classes = ["Baba/Abba", "Dada/Nana", "Dadi/Nani", "Other"]
    result = _align_pair("src", src_classes, "tgt", tgt_classes,
                         edit_threshold=2)
    assert result["n_exact"] == 3, (
        f"expected all 3 slash-joined matches to be exact, "
        f"got {result['n_exact']}; exact={result['exact']}"
    )
    # The exact match's RHS should be the canonical (slash-joined) form,
    # not a variant.
    assert result["exact"]["baba"] == "Baba/Abba"
    assert result["exact"]["dada"] == "Dada/Nana"
    assert result["exact"]["dadi"] == "Dadi/Nani"


def test_align_pair_jaccard_lower_bound_monotonic():
    from preprocessing.build_bangla_vocab_alignment import _align_pair

    # Two identical lists should produce 100% exact overlap and Jaccard==1.0.
    cls = ["a", "b", "c", "d"]
    r = _align_pair("s", cls, "t", cls, edit_threshold=1)
    assert r["n_exact"] == 4
    assert r["jaccard_lower_bound"] == 1.0

    # Disjoint vocabularies that ARE within edit distance 1 — should produce
    # 0 exact but ALL candidates.
    src = ["aa", "bb", "cc"]
    tgt = ["aab", "bbc", "ccd"]
    r = _align_pair("s", src, "t", tgt, edit_threshold=1)
    assert r["n_exact"] == 0
    assert r["n_candidates"] == 3
    assert r["jaccard_lower_bound"] == 0.0
