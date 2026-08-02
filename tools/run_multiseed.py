"""Run an experiment sweep across multiple seeds.

For each (config, seed) pair, invokes `python main.py --config <config>
--seed <seed> -Experiment_name <stem>_seed<N>` so every run gets its own
work_dir, its own checkpoints, and its own row(s) in results_final.csv.

Usage examples:

    # Run the full signer-independent sweep at 3 seeds
    python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2

    # Run one config at 3 seeds
    python tools/run_multiseed.py --single config/bdsl_block_gcn_si.yaml \
        --seeds 0 1 2

    # Resume after a crash: skip seeds whose work_dir already has a final
    # checkpoint.
    python tools/run_multiseed.py --config experiments_si.yaml --seeds 0 1 2 \
        --skip-existing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config_stem(cfg_path):
    return os.path.splitext(os.path.basename(cfg_path))[0]


def _load_experiments(sweep_path):
    with open(sweep_path) as f:
        data = yaml.safe_load(f)
    return data.get("experiments", [])


def _run_has_final_marker(work_dir):
    # A completed run writes at least one checkpoint + an eval_results dir.
    if not os.path.isdir(work_dir):
        return False
    eval_dir = os.path.join(work_dir, "eval_results")
    if not os.path.isdir(eval_dir):
        return False
    return any(fn.endswith(".pkl") for fn in os.listdir(eval_dir))


def _launch(config_path, seed, skip_existing):
    stem = _config_stem(config_path)
    exp_name = f"{stem}_seed{seed}"
    work_dir = os.path.join(".", "work_dir", exp_name)

    if skip_existing and _run_has_final_marker(work_dir):
        print(f"[skip] {exp_name}: existing run found at {work_dir}")
        return 0

    cmd = [
        sys.executable, "-u", "main.py",
        "--config", config_path,
        "--seed", str(seed),
        "-Experiment_name", exp_name,
    ]
    print("\n" + "=" * 70)
    print(f"[run] {exp_name}")
    print(f"      cmd: {' '.join(cmd)}")
    print("=" * 70)
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="a sweep YAML (with 'experiments:' list)")
    group.add_argument("--single", help="a single model config YAML")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if args.config:
        exps = _load_experiments(args.config)
        configs = [e["config"] for e in exps]
    else:
        configs = [args.single]

    failures = []
    for cfg in configs:
        for seed in args.seeds:
            rc = _launch(cfg, seed, args.skip_existing)
            if rc != 0:
                failures.append((cfg, seed, rc))

    if failures:
        print("\nFAILED RUNS:")
        for cfg, seed, rc in failures:
            print(f"  {cfg} seed={seed} exit={rc}")
        sys.exit(1)
    print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
