#!/usr/bin/env python
"""Reproduce the paper's numbers from the archived results (no data, no GPU).

Every step re-runs an analysis script on the files in results/ and docs/ and compares its
output with the archived output that the paper quotes:

  v2      tools/p1_fix/refit_v2.py                 -> docs/p1_fix_2026-10/results/refit_v2.json (byte-identical)
  v1      tools/p1_refit/refit.py --quick          -> decision, primary posterior and LOLO of
                                                      docs/p1_refit_2026-10/refit_results.json (exact)
  report  tools/p1_refit/report.py                 -> docs/p1_refit_2026-10/RESULTS.md (generated part)
  figs    tools/paper/make_figs.py                 -> Figs. 1-4; asserts every caption number against the JSON
  null    tools/paper/null_sim.py                  -> results/provenance/null_sim.txt
  reml    tools/paper/exact_reml_v1.py             -> results/provenance/v1_exact_reml_2026-10-23.txt
  hrnet   tools/hrnet_gate.py                      -> results/provenance/hrnet_gate_2026-10-22.txt
  audit   scripts/hpc/audit_reruns_2026-10/collect.py -> docs/audit_reruns_2026-10/r{1,2}_*.md (tables)

  python tools/reproduce.py              # all steps (about 5 minutes on 2 CPU cores)
  python tools/reproduce.py --only v2 figs
  python tools/reproduce.py --full       # v1 with every sensitivity analysis (about 10-20 minutes)

Exit status 0 iff every step matches.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
HEADLINE = {  # quoted in the paper's abstract / Table 2 / Sec. 7.4
    "v1": dict(P_b_neg=0.739, lolo=2, verdict="REFRAME"),
    "v2": dict(b=(-11.56, -4.59, 2.61), P_b_neg=0.905, lolo=6, verdict="NOT SUPPORTED"),
}


def run(cmd, out=None, env=None):
    e = dict(os.environ, MPLBACKEND="Agg", **(env or {}))
    r = subprocess.run(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout[-3000:]}")
    if out:
        open(out, "w").write(r.stdout)
    return r.stdout


def same_text(a, b, prefix=False, skip_lines=()):
    A = open(a).read().splitlines()
    B = open(b).read().splitlines()
    if prefix:
        B = B[:len(A)]
    A = [x for i, x in enumerate(A) if i not in skip_lines]
    B = [x for i, x in enumerate(B) if i not in skip_lines]
    if A == B:
        return True, ""
    for i, (x, y) in enumerate(zip(A, B)):
        if x != y:
            return False, f"first difference at line {i + 1}:\n  got      {x}\n  archived {y}"
    return False, f"length differs ({len(A)} vs {len(B)} lines)"


def close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


# ----------------------------------------------------------------------------------- steps
def step_v2(tmp, full):
    out = os.path.join(tmp, "v2")
    log = run([PY, "tools/p1_fix/refit_v2.py", "--out", out])
    got = os.path.join(out, "refit_v2.json")
    ref = os.path.join(ROOT, "docs/p1_fix_2026-10/results/refit_v2.json")
    same = open(got, "rb").read() == open(ref, "rb").read()
    P = json.load(open(got))["primary"]
    h = HEADLINE["v2"]
    head = (P["verdict"] == h["verdict"] and round(P["P_b_neg"], 3) == h["P_b_neg"] and P["lolo"]["correct"] == h["lolo"]
            and tuple(round(x, 2) for x in P["b"]) == h["b"])
    msg = (f"verdict {P['verdict']}; b {P['b'][1]:+.2f} [{P['b'][0]:+.2f}, {P['b'][2]:+.2f}]; P(b<0) {P['P_b_neg']:.3f}; "
           f"LOLO {P['lolo']['correct']}/{P['lolo']['n']}; JSON {'byte-identical' if same else 'DIFFERS'}")
    return same and head, msg, log


def step_v1(tmp, full):
    out = os.path.join(tmp, "v1")
    cmd = [PY, "tools/p1_refit/refit.py", "--pseeds", "0-6", "--out", out]
    cmd += ["--alt-dist", "docs/p1_refit_2026-10/realpool_distances.json"] if full else ["--quick"]
    log = run(cmd)
    got = json.load(open(os.path.join(out, "refit_results.json")))
    ref = json.load(open(os.path.join(ROOT, "docs/p1_refit_2026-10/refit_results.json")))
    keys = list(got) if full else ["prereg_sha256", "pseeds", "targets", "dist", "exclusions", "cells",
                                   "pseeds_complete_all7", "data_sufficient", "primary", "lolo", "decision"]
    bad = [k for k in keys if not close(got.get(k), ref.get(k))]
    D = got["decision"]
    h = HEADLINE["v1"]
    head = D["verdict"] == h["verdict"] and round(D["D1_P_b_neg"], 3) == h["P_b_neg"] and D["D2_lolo_correct"] == h["lolo"]
    b = got["primary"]["b"]
    msg = (f"verdict {D['verdict']}; P(b<0) {D['D1_P_b_neg']:.3f}; b {b[1]:+.2f} [{b[0]:+.2f}, {b[2]:+.2f}]; "
           f"LOLO {D['D2_lolo_correct']}/7; {len(keys) - len(bad)}/{len(keys)} fields equal the archived JSON"
           + (f" (differ: {bad})" if bad else ""))
    return not bad and head, msg, log


def step_report(tmp, full):
    out = os.path.join(tmp, "RESULTS.md")
    log = run([PY, "tools/p1_refit/report.py", "--md", out])
    # line 4 states the hash of the distributed PREREG.md (a documented redacted copy; PROVENANCE.md)
    ok, why = same_text(out, os.path.join(ROOT, "docs/p1_refit_2026-10/RESULTS.md"), prefix=True, skip_lines=(3,))
    return ok, "generated part of docs/p1_refit_2026-10/RESULTS.md " + ("identical" if ok else why), log


def step_figs(tmp, full):
    out = os.path.join(tmp, "figures")
    log = run([PY, "tools/paper/make_figs.py", "--out-dir", out])
    figs = sorted(f for f in os.listdir(out) if f.endswith(".pdf"))
    ok = figs == ["fig_distance_steps.pdf", "fig_seedlaws.pdf", "fig_v2law.pdf", "fig_variance.pdf"]
    return ok, f"{len(figs)} figures rendered, caption numbers asserted against the JSON", log


def _provenance(cmd, ref, tmp, name):
    out = os.path.join(tmp, name)
    log = run(cmd, out=out)
    ok, why = same_text(out, os.path.join(ROOT, ref))
    return ok, f"{ref} " + ("identical" if ok else why), log


def step_null(tmp, full):
    return _provenance([PY, "tools/paper/null_sim.py"], "results/provenance/null_sim.txt", tmp, "null.txt")


def step_reml(tmp, full):
    return _provenance([PY, "tools/paper/exact_reml_v1.py"], "results/provenance/v1_exact_reml_2026-10-23.txt", tmp, "reml.txt")


def step_hrnet(tmp, full):
    return _provenance([PY, "tools/hrnet_gate.py"], "results/provenance/hrnet_gate_2026-10-22.txt", tmp, "hrnet.txt")


def step_audit(tmp, full):
    out = os.path.join(tmp, "audit")
    os.makedirs(out, exist_ok=True)
    log = run([PY, "scripts/hpc/audit_reruns_2026-10/collect.py", "--out", out, "--monotr-seeds", "0,1,2"])
    res = []
    for f in ("r1_autsl_leakfree.md", "r2_true_mono.md"):
        ok, why = same_text(os.path.join(out, f), os.path.join(ROOT, "docs/audit_reruns_2026-10", f), prefix=True)
        res.append((ok, f if ok else f"{f}: {why}"))
    return all(o for o, _ in res), "tables of " + ", ".join(m for _, m in res) + " identical" * all(o for o, _ in res), log


STEPS = dict(v2=step_v2, v1=step_v1, report=step_report, figs=step_figs, null=step_null, reml=step_reml,
             hrnet=step_hrnet, audit=step_audit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(STEPS), help="run these steps only")
    ap.add_argument("--full", action="store_true", help="v1: run every sensitivity analysis and compare the whole JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each step's output")
    a = ap.parse_args()
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in a.only or list(STEPS):
            t0 = time.time()
            try:
                ok, msg, log = STEPS[name](tmp, a.full)
            except Exception as e:  # noqa: BLE001 - report and continue with the other steps
                ok, msg, log = False, f"ERROR {e}", ""
            fails += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] {name:6s} {msg}  ({time.time() - t0:.0f} s)", flush=True)
            if a.verbose and log:
                print(log)
    print("all steps reproduce the archived results" if not fails else f"{fails} step(s) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
