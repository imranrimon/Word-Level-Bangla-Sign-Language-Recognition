#!/usr/bin/env python
"""Train one (config, seed) run resume-safely, then optionally TEST-eval its best ckpt.

Unlike run_multiseed --skip-existing (which treats any eval pkl as "done" and so
skips killed partial runs), completion here = the final-epoch checkpoint exists.
A killed run resumes from its latest periodic checkpoint (weights + optimizer via
main.py --weights/--start-epoch). main.py resets best_acc on restart, so the
pre-resume best checkpoint/score file are backed up and restored if they beat the
resumed run's best. A partial run with no checkpoint is moved aside, not reused.

  python tools/run_resumable_seed.py --config config/X.yaml --seed 0 --exp X_seed0 \
      [--base-lr 0.05] [--test-config config/X_test.yaml --test-exp X_test_seed0]
"""
import argparse, glob, os, re, shutil, subprocess, sys, time
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ckpts(wd, exp):
    out = {}
    for p in glob.glob(os.path.join(wd, f"{exp}_model-*.pt")):
        m = re.search(r"_model-(\d+)\.pt$", p)
        if m: out[int(m.group(1))] = p
    return out


def _best_acc(path):
    try: return float(torch.load(path, map_location="cpu").get("acc", -1))
    except FileNotFoundError: return -1.0
    except OSError: raise  # unreadable storage must not look like "no best yet"
    except Exception: return -1.0


def _run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0: sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exp", required=True); ap.add_argument("--base-lr", type=float)
    ap.add_argument("--test-config"); ap.add_argument("--test-exp")
    ap.add_argument("--init-weights", help="fresh start only: load these weights (classifier fc dropped)")
    a = ap.parse_args()
    n_ep = int(yaml.safe_load(open(os.path.join(ROOT, a.config)))["num_epoch"])
    wd = os.path.join(ROOT, "work_dir", a.exp)
    base = [sys.executable, "-u", "main.py", "--config", a.config, "--seed", str(a.seed),
            "-Experiment_name", a.exp] + (["--base-lr", str(a.base_lr)] if a.base_lr is not None else [])
    ck = _ckpts(wd, a.exp)
    best = os.path.join(wd, f"{a.exp}_model-best.pt")
    if n_ep - 1 in ck:
        print(f"[done] {a.exp}: final checkpoint exists", flush=True)
    elif ck:
        # a requeue/kill can land mid-torch.save; set a truncated latest checkpoint aside
        while ck:
            last = max(ck)
            try: torch.load(ck[last], map_location="cpu"); break
            except FileNotFoundError: raise
            except OSError as e:  # I/O error (e.g. Lustre OST offline): the file may be fine, so stop loudly
                print(f"[resume] I/O error reading {ck[last]}: {e}; refusing to resume", flush=True); sys.exit(3)
            except Exception:
                shutil.move(ck[last], ck[last] + ".corrupt"); print(f"[resume] unreadable {ck.pop(last)} set aside", flush=True)
        if not ck:
            print(f"[resume] {a.exp}: no readable checkpoint", flush=True); sys.exit(1)
        last = max(ck)
        bk = {p: p + ".preresume" for p in (best, os.path.join(wd, "eval_results", "best_acc.pkl")) if os.path.exists(p)}
        for p, q in bk.items(): shutil.copy2(p, q)
        old = _best_acc(best + ".preresume") if best in bk else -1.0
        print(f"[resume] {a.exp} from epoch {last + 1} (pre-resume best {old:.4f})", flush=True)
        _run(base + ["--weights", ck[last], "--start-epoch", str(last + 1)])
        if old > _best_acc(best):
            for p, q in bk.items(): shutil.copy2(q, p)
            print(f"[resume] restored pre-resume best ({old:.4f})", flush=True)
    else:
        if os.path.isdir(wd):
            aside = f"{wd}.partial_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(wd, aside); print(f"[fresh] moved partial run aside -> {aside}", flush=True)
        init = (["--weights", a.init_weights, "--ignore-weights", "fc.weight", "fc.bias"]
                if a.init_weights else [])
        _run(base + init)
    if a.test_config:
        twd = os.path.join(ROOT, "work_dir", a.test_exp)
        tpk = os.path.join(twd, "eval_results", "best_acc.pkl")
        if os.path.exists(tpk) and os.path.getmtime(tpk) > os.path.getmtime(best):
            print(f"[done] {a.test_exp} already evaluated", flush=True); return
        if os.path.isdir(twd): shutil.rmtree(twd)
        _run([sys.executable, "-u", "main.py", "--config", a.test_config, "--phase", "test",
              "--weights", best, "-Experiment_name", a.test_exp, "--seed", str(a.seed)])


if __name__ == "__main__":
    main()
