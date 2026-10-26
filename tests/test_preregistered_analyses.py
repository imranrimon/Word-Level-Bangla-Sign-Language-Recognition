"""The pre-registered analyses reproduce from the archived run logs (no data, no GPU).

* v2 rerun (tools/p1_fix/refit_v2.py): the 24 cells rebuilt from results/results_p1fix_v2.csv equal the
  archived cells, and the primary fit gives the binding verdict NOT SUPPORTED with the archived
  posterior (b = -4.59 [-11.56, +2.61], P(b<0) = 0.905, LOLO 6/8).
* 7-seed refit (tools/p1_refit/refit.py): the 49 cells rebuilt from results/results_final.csv equal the
  archived cells and pooled gains; the archived decision is REFRAME (P(b<0) = 0.739, LOLO 2/7).
  The full refit (minutes of CPU) runs in `python tools/reproduce.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "p1_fix"))
sys.path.insert(0, str(ROOT / "tools" / "p1_refit"))

import refit_v2 as V2  # noqa: E402

V2_JSON = json.load(open(ROOT / "docs/p1_fix_2026-10/results/refit_v2.json"))
V1_JSON = json.load(open(ROOT / "docs/p1_refit_2026-10/refit_results.json"))


def _v2_cells():
    val, test = V2.load_csv(str(ROOT / "results" / "results_p1fix_v2.csv"))
    runs, log = V2.gather(val, test, V2.LANGS, [0, 1, 2], [0, 1, 2, 3, 4], V2.base_v2)
    return V2.cells(runs, V2.LANGS, [0, 1, 2]), log


def test_v2_cells_and_exclusions_match_archive():
    C, log = _v2_cells()
    assert len(C) == 24 and sum(c["nx"] + c["np"] for c in C) == 239
    assert len(log) == 1 and "gross failure" in log[0][4] and log[0][:3] == ("KArSL", 0, "pool")
    ref = {(c["t"], c["p"]): c for c in V2_JSON["cells"]}
    for c in C:
        r = ref[(c["t"], c["p"])]
        assert c["G"] == pytest.approx(r["G"], abs=1e-12) and c["s2"] == pytest.approx(r["s2"], abs=1e-12)


def test_v2_primary_verdict_reproduces():
    C, _ = _v2_cells()
    dist = V2_JSON["distances_primary"]
    P = V2.analyse(C, dist, np.random.default_rng(20261022))
    A = V2_JSON["primary"]
    assert P["verdict"] == A["verdict"] == "NOT SUPPORTED"
    assert P["lolo"]["correct"] == 6 and P["lolo"]["n"] == 8
    assert all(v["pred"] < 0 for v in P["lolo"]["per_lang"].values())        # C2 carries no distance information
    np.testing.assert_allclose(P["b"], A["b"], atol=1e-9)
    assert P["P_b_neg"] == pytest.approx(A["P_b_neg"], abs=1e-12)
    assert [round(x, 2) for x in P["b"]] == [-11.56, -4.59, 2.61] and round(P["P_b_neg"], 3) == 0.905


def test_v1_cells_match_archive():
    import refit as V1
    val, test = V1.A.load(str(ROOT / "results" / "results_final.csv"))
    pseeds = list(range(7))
    runs, log = V1.gather(val, test, pseeds)
    C = V1.cells(runs, pseeds)
    assert len(C) == 49 and not log
    ref = {(c["t"], c["p"]): c for c in V1_JSON["cells"]}
    for c in C:
        r = ref[(c["t"], c["p"])]
        assert float(c["G"]) == pytest.approx(r["G"], abs=1e-12) and float(c["s2"]) == pytest.approx(r["s2"], abs=1e-12)
    pooled = json.load(open(ROOT / "docs/p1_refit_2026-10/pooled_gains.json"))
    G = np.full((7, 7), np.nan)
    for c in C:
        G[c["pi"], c["li"]] = c["G"]
    np.testing.assert_allclose(G, np.array(pooled["G"]), atol=1e-9)


def test_v1_archived_decision():
    D = V1_JSON["decision"]
    assert D["verdict"] == "REFRAME" and D["D2_lolo_correct"] == 2 and round(D["D1_P_b_neg"], 3) == 0.739
    assert not D["D1_pass"] and not D["D2_pass"]
