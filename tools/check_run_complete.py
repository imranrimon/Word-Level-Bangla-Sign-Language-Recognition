#!/usr/bin/env python
"""Completeness check for main.py training runs, and a sweep of every fine-tune the P1 analyses read.

A main.py training run is COMPLETE when its log reached the final epoch of its own saved
config (work_dir/<run>/config.yaml num_epoch): main.py always evaluates at the last epoch,
so `max("Eval epoch: N") == num_epoch` (fallback: the final-epoch checkpoint
`<run>_model-<num_epoch-1>.pt` exists). Checkpoints are pruned to best/last in this repo,
so checkpoint presence alone is not a completeness signal.

A `--phase test` run (TEST-split re-score of a training checkpoint) is traced to its training
run through the `Load weights from <ckpt>.` line of its log; it is flagged when that training
run is incomplete, when the scored checkpoint is not `<run>_model-best.pt` (best-on-val
selection not applied), or when the score dump is older than the checkpoint (stale).

Read errors are fatal (no silent skips).

  python tools/check_run_complete.py work_dir/gsl_si_f10_v2_pool_ps0_seed0 [...]   # one-line status each
  python tools/check_run_complete.py --p1-sweep [--md completeness.md]            # every P1 fine-tune

Library use: `from check_run_complete import train_status`; `train_status(wd)["complete"]`.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, "work_dir")
EVAL_RE = re.compile(r"\] Eval epoch: (\d+)")
TRAIN_RE = re.compile(r"\] Training epoch: (\d+)")
LOAD_RE = re.compile(r"Load weights from (.+?)\.?\s*$", re.M)
NAN_RE = re.compile(r"Loss: nan", re.I)
EVAL_TOP1_RE = re.compile(r"\] Eval epoch: (\d+)\n(?:.*\n){0,3}?.*\tTop1: ([\d.]+)%")


def val_curve(log):
    """{epoch (1-based): val Top-1 %}; a resumed run's later evals override earlier ones."""
    return {int(e): float(a) for e, a in EVAL_TOP1_RE.findall(log)}


def best_val_epoch(curve):
    """main.py keeps the FIRST epoch reaching the max (strict >)."""
    if not curve:
        return None
    m = max(curve.values())
    return min(e for e, a in curve.items() if a == m)


def _read(path):
    with open(path, errors="replace") as f:   # OSError (EIO) propagates on purpose
        return f.read()


def local_dir(path):
    """Map a logged checkpoint path (Windows / old /users clone / relative) to work_dir/<run>."""
    p = path.replace("\\", "/").rstrip("/")
    parts = p.split("/")
    if "work_dir" in parts:
        i = len(parts) - 1 - parts[::-1].index("work_dir")
        if i + 1 < len(parts):
            return os.path.join(WD, parts[i + 1])
    return os.path.join(WD, os.path.basename(os.path.dirname(p)))


def ckpts(wd):
    out = {}
    for p in glob.glob(os.path.join(wd, "*_model-*.pt")):
        m = re.search(r"_model-(\d+)\.pt$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def train_status(wd):
    """Status dict for one main.py work_dir (train or test phase)."""
    wd = os.path.abspath(wd)
    st = dict(run=os.path.basename(wd), wd=wd, exists=os.path.isdir(wd))
    if not st["exists"]:
        return dict(st, status="MISSING", complete=False)
    cfgp = os.path.join(wd, "config.yaml")
    if not os.path.isfile(cfgp):
        # score-only dir written by an external evaluator (eval_results/best_acc.pkl, no config/log):
        # judge the training run of the same name without the _test suffix
        m = re.match(r"^(.*_seed\d+)_test$", st["run"])
        if m and os.path.isfile(os.path.join(wd, "eval_results", "best_acc.pkl")):
            cands = [m.group(1), re.sub(r"_seed(\d+)$", r"_scratch_seed\1", m.group(1))]  # lr01 Protocol A: *_lr01_scratch_seed<S>
            par = next((c for c in cands if os.path.isdir(os.path.join(os.path.dirname(wd), c))), cands[0])
            ps = train_status(os.path.join(os.path.dirname(wd), par))
            return dict(st, phase="test", status="SCORE_ONLY(parent " + ps["status"] + ")", parent=par,
                        parent_status=ps, complete=ps.get("complete"), flags=[])
        return dict(st, status="NO_CONFIG", complete=None)
    cfg = yaml.safe_load(_read(cfgp)) or {}
    st["phase"] = cfg.get("phase", "train")
    st["num_epoch"] = int(cfg.get("num_epoch", 0) or 0)
    logp = os.path.join(wd, "log.txt")
    log = _read(logp) if os.path.isfile(logp) else ""
    st["log_mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(logp))) if log else ""
    loads = LOAD_RE.findall(log)
    if st["phase"] == "test" and loads and os.path.normpath(local_dir(loads[-1].strip())) == os.path.normpath(wd):
        # a --phase test re-score run in place over a training dir (config.yaml overwritten): judge the training log
        st["phase"] = "train"
        st["note"] = "config.yaml overwritten by an in-place --phase test run"
    if st["phase"] == "test":
        st["weights"] = loads[-1].strip() if loads else None
        pk = os.path.join(wd, "eval_results", "best_acc.pkl")
        st["has_score"] = os.path.isfile(pk)
        if not st["weights"]:
            return dict(st, status="TEST_NO_WEIGHTS_LINE", complete=None)
        parent = local_dir(st["weights"])
        st["parent"] = os.path.basename(parent)
        st["uses_best"] = st["weights"].replace("\\", "/").endswith("_model-best.pt")
        ps = train_status(parent)
        st["parent_status"] = ps
        lw = os.path.join(parent, os.path.basename(st["weights"].replace("\\", "/")))
        st["stale"] = bool(st["has_score"] and os.path.isfile(lw)
                           and os.path.getmtime(pk) < os.path.getmtime(lw))
        flags = []
        if ps.get("complete") is False:
            flags.append(f"PARENT_{ps['status']}")
        if not st["uses_best"]:
            flags.append("NOT_BEST_CKPT")
        if st["stale"]:
            flags.append("STALE")
        if not st["has_score"]:
            flags.append("NO_SCORE")
        m = re.search(r"_model-(\d+)\.pt$", st["weights"])
        curve = val_curve(_read(os.path.join(parent, "log.txt"))) if os.path.isfile(os.path.join(parent, "log.txt")) else {}
        if not st["uses_best"] and m and curve:
            ep = int(m.group(1)) + 1
            be = best_val_epoch(curve)
            st["scored_epoch"], st["best_epoch"] = ep, be
            st["scored_val"], st["best_val"] = curve.get(ep), curve[be]
            if curve.get(ep) == curve[be]:
                flags.remove("NOT_BEST_CKPT")   # the retained per-epoch checkpoint IS the best-val epoch
                st["uses_best"] = True
            else:
                flags[flags.index("NOT_BEST_CKPT")] = (f"NOT_BEST_CKPT(scored ep{ep} val {curve.get(ep)}; "
                                                       f"best ep{be} val {curve[be]})")
        st["flags"] = flags
        st["complete"] = ps.get("complete")
        return dict(st, status="TEST_OK" if not flags else "TEST_" + "+".join(flags))
    ev = [int(x) for x in EVAL_RE.findall(log)]
    tr = [int(x) for x in TRAIN_RE.findall(log)]
    ck = ckpts(wd)
    st.update(last_eval=max(ev) if ev else 0, last_train=max(tr) if tr else 0,
              n_eval=len(ev), ckpt_epochs=sorted(ck), last_ckpt=max(ck) if ck else None,
              has_best=os.path.isfile(os.path.join(wd, f"{st['run']}_model-best.pt")),
              nan=bool(NAN_RE.search(log)))
    n = st["num_epoch"]
    st["complete"] = bool(n) and (st["last_eval"] >= n or (n - 1) in ck)
    if st["complete"]:
        st["status"] = "COMPLETE"
    elif not log and not ck:
        st["status"] = "EMPTY"
    else:
        st["status"] = "INCOMPLETE"
        if log and time.time() - os.path.getmtime(logp) < 1800:
            st["status"] = "IN_PROGRESS"   # log written in the last 30 min: a job is (probably) training it
    # resume point (tools/run_resumable_seed.py semantics): latest checkpoint + 1, else fresh
    st["resume_from"] = (max(ck) + 1) if ck else 0
    return st


def short(st):
    if st.get("phase") == "test":
        ps = st.get("parent_status", {})
        return (f"{st['run']}: {st['status']}  weights={os.path.basename(str(st.get('weights')))}"
                f"  parent={st.get('parent')} [{ps.get('status')} {ps.get('last_eval', '?')}/{ps.get('num_epoch', '?')}]")
    if st["status"] in ("MISSING", "NO_CONFIG"):
        return f"{st['run']}: {st['status']}"
    return (f"{st['run']}: {st['status']}  eval {st['last_eval']}/{st['num_epoch']}  train-ep {st['last_train']}"
            f"  ckpts {st['ckpt_epochs']}  best.pt {'y' if st['has_best'] else 'n'}"
            f"{'  NaN-loss' if st['nan'] else ''}")


# =============================================================================
# P1 enumeration: every fine-tune the two pre-registered analyses read
# =============================================================================
V1_STEM = {"BdSL": "bdsl_block_gcn_si", "LSA64": "lsa64_si", "KArSL": "karsl_si", "AUTSL": "autsl_si",
           "GSL": "gsl_si", "LSFB": "lsfb_si", "LIBRAS": "libras_si"}
V2_STEM = dict(V1_STEM, BISINDO="bisindo_si")


def p1_runs():
    """[(group, run name)]: 7-seed refit (v1 backbones) and the corrected v2 rerun."""
    out = []
    for t, stem in V1_STEM.items():
        ft = [0, 1, 2, 3, 4] if t in ("GSL", "LSFB", "LIBRAS") else [0, 1, 2]
        for p in range(7):
            for arm in ("pool", "xling"):
                for f in ft:
                    out.append(("v1 7-pseed refit", f"{stem}_lr01_f10_{arm}{'' if p == 0 else f'_ps{p}'}_seed{f}"))
    for p in range(3):   # BdSL headline arms (mono over pseeds 0-2; scratch has no pretraining seed)
        for f in (0, 1, 2):
            out.append(("v1 BdSL headline", f"bdsl_block_gcn_si_lr01_f10_mono{'' if p == 0 else f'_ps{p}'}_seed{f}"))
    for f in (0, 1, 2):
        out.append(("v1 BdSL headline", f"bdsl_block_gcn_si_lr01_f10_scratch_seed{f}"))
    for t, stem in V2_STEM.items():
        for p in range(3):
            for arm in ("pool", "xling"):
                for f in range(5):
                    out.append(("v2 rerun", f"{stem}_f10_v2_{arm}_ps{p}_seed{f}"))
    return out


def sweep(out_md=None):
    rows, counts = [], collections.Counter()
    for group, run in p1_runs():
        st = train_status(os.path.join(WD, run))
        counts[(group, st["status"])] += 1
        if st.get("complete") is not True:
            rows.append((group, st))
    L = [f"# P1 completeness sweep ({time.strftime('%Y-%m-%d %H:%M')})", "",
         "Rule: a main.py training run is complete when `max(Eval epoch) == num_epoch` of its saved "
         "`config.yaml` (or the final-epoch checkpoint exists).", "",
         "| group | status | runs |", "|---|---|---|"]
    L += [f"| {g} | {s} | {n} |" for (g, s), n in sorted(counts.items())]
    L += ["", "## Runs that are not complete", "", "| group | run | status |", "|---|---|---|"]
    L += [f"| {g} | `{st['run']}` | {st['status']} |" for g, st in rows]
    txt = "\n".join(L) + "\n"
    if out_md:
        os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
        open(out_md, "w").write(txt)
    print(txt)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--p1-sweep", action="store_true", help="check every fine-tune the P1 analyses read")
    ap.add_argument("--md", help="with --p1-sweep: also write the report here")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rc = 0
    for d in a.dirs:
        st = train_status(d)
        print(json.dumps({k: v for k, v in st.items() if k != "parent_status"}, default=str) if a.json else short(st))
        rc |= 0 if st.get("complete") else 1
    if a.p1_sweep:
        rc |= 1 if sweep(a.md) else 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
