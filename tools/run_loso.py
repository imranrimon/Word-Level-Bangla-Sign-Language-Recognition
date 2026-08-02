"""Run a leave-one-signer-out (LOSO) sweep for one or more model configs.

For audit fix #2. Iterates over `--test-signers` (each becomes one LOSO fold),
re-bundles cached pose for that fold via generate_loso_split_bundle.py, then
trains each config x seed combination by spawning main.py with an override
config that points at the fold-specific data dir.

Per-fold data layout:
    data/bdsl_si_loso/test_U<XX>_val_U<YY>/
        train_data.npy, train_label.pkl
        val_data.npy,   val_label.pkl
        test_data.npy,  test_label.pkl
        classes.json

Per-run work_dir layout:
    work_dir/<config_stem>_loso_test<XX>_seed<N>/

Per-run results row in results_final.csv:
    Experiment = <config_stem>_loso_test<XX>_seed<N>

Aggregator (`tools/summarize_seeds.py`) collapses across seeds AND folds via
the LOSO suffix matching; report mean ± std across folds as the headline
variance (signer noise) instead of mean ± std across seeds (init noise only).

Usage:
    # Full 11-fold LOSO at 3 seeds for one headline model:
    python tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \\
        --test-signers 1 4 5 6 8 9 11 12 2 13 15 --seeds 0 1 2

    # 3-fold LOSO (cheap variance) for the 8 baselines:
    python tools/run_loso.py --config experiments_si_baselines.yaml \\
        --test-signers 2 13 15 --seeds 0 1 2

    # Resume after crash:
    python tools/run_loso.py --single <cfg> --test-signers 1 4 --seeds 0 1 2 \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOSO_DATA_ROOT = ROOT / "data" / "bdsl_si_loso"


def _config_stem(cfg_path):
    return os.path.splitext(os.path.basename(cfg_path))[0]


def _load_experiments(sweep_path):
    with open(sweep_path) as f:
        data = yaml.safe_load(f)
    return [e["config"] for e in data.get("experiments", [])]


def _pick_val_signer(test_signer, all_full_signers):
    """Deterministic val-signer choice: smallest signer ID != test_signer."""
    for s in sorted(all_full_signers):
        if s != test_signer:
            return s
    raise RuntimeError("no full signer remains for val")


def _ensure_fold_data(cache_dir, test_signer, val_signer):
    fold_dir = LOSO_DATA_ROOT / f"test_U{test_signer:02d}_val_U{val_signer:02d}"
    sentinel = fold_dir / "test_data.npy"
    if sentinel.exists():
        return fold_dir
    print(f"\n[fold] generating {fold_dir.name}...")
    rc = subprocess.call([
        sys.executable, "-u",
        str(ROOT / "preprocessing" / "generate_loso_split_bundle.py"),
        "--cache-dir", str(cache_dir),
        "--output-dir", str(fold_dir),
        "--test-signer", str(test_signer),
        "--val-signer", str(val_signer),
    ], cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"LOSO data generation failed for test={test_signer}")
    return fold_dir


def _write_fold_config(base_cfg_path, fold_dir):
    """Produce a temporary YAML with data paths overridden to point at fold_dir.

    We mutate `train_feeder_args.data_path`, `train_feeder_args.label_path`,
    `test_feeder_args.data_path`, `test_feeder_args.label_path`. Everything
    else (model class, optimiser, schedule, num_class) is inherited verbatim.
    """
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)
    for split, key in (("train", "train_feeder_args"),
                       ("val",   "test_feeder_args")):
        if key not in cfg:
            cfg[key] = {}
        cfg[key]["data_path"] = str(fold_dir / f"{split}_data.npy").replace("\\", "/")
        cfg[key]["label_path"] = str(fold_dir / f"{split}_label.pkl").replace("\\", "/")
    fd, path = tempfile.mkstemp(prefix="loso_", suffix=".yaml", dir=str(ROOT / ".tmp_loso"))
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return path


def _run_has_final_marker(work_dir):
    if not os.path.isdir(work_dir):
        return False
    eval_dir = os.path.join(work_dir, "eval_results")
    if not os.path.isdir(eval_dir):
        return False
    return any(fn.endswith(".pkl") for fn in os.listdir(eval_dir))


def _launch(config_path, fold_dir, test_signer, seed, skip_existing):
    stem = _config_stem(config_path)
    exp_name = f"{stem}_loso_test{test_signer:02d}_seed{seed}"
    work_dir = os.path.join(".", "work_dir", exp_name)

    if skip_existing and _run_has_final_marker(work_dir):
        print(f"[skip] {exp_name}: existing run at {work_dir}")
        return 0

    os.makedirs(str(ROOT / ".tmp_loso"), exist_ok=True)
    fold_cfg = _write_fold_config(config_path, fold_dir)

    cmd = [
        sys.executable, "-u", "main.py",
        "--config", fold_cfg,
        "--seed", str(seed),
        "-Experiment_name", exp_name,
    ]
    print("\n" + "=" * 70)
    print(f"[run] {exp_name}")
    print(f"      fold: {fold_dir.name}")
    print(f"      cmd:  {' '.join(cmd)}")
    print("=" * 70)
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    from preprocessing.bdsl_signer_split import SIGNER_SPLIT

    full_signers = (set(SIGNER_SPLIT["train"])
                    | set(SIGNER_SPLIT["val"])
                    | set(SIGNER_SPLIT["test"]))

    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="sweep YAML with 'experiments:' list")
    group.add_argument("--single", help="a single model config YAML")
    ap.add_argument(
        "--test-signers", nargs="+", type=int, default=sorted(full_signers),
        help="LOSO test signers (default: all 11 full-vocab signers)",
    )
    ap.add_argument(
        "--val-signer", type=int, default=None,
        help="fix the val signer across folds; default = smallest full signer != test",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cache-dir", default="data/bdsl_cache")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    # Sanity-check test signers.
    for s in args.test_signers:
        if s not in full_signers:
            raise SystemExit(
                f"--test-signer U{s} is not in the 11 full-vocab signers "
                f"{sorted(full_signers)}"
            )

    configs = ([args.single] if args.single
               else _load_experiments(args.config))

    # Prepare every LOSO fold's data bundle ONCE (cached after first run).
    fold_dirs = {}
    for ts in args.test_signers:
        vs = args.val_signer if args.val_signer is not None else _pick_val_signer(ts, full_signers)
        fold_dirs[ts] = _ensure_fold_data(Path(args.cache_dir), ts, vs)

    failures = []
    for cfg in configs:
        for ts in args.test_signers:
            for seed in args.seeds:
                rc = _launch(cfg, fold_dirs[ts], ts, seed, args.skip_existing)
                if rc != 0:
                    failures.append((cfg, ts, seed, rc))

    if failures:
        print("\nFAILED RUNS:")
        for cfg, ts, seed, rc in failures:
            print(f"  {cfg} test=U{ts:02d} seed={seed} exit={rc}")
        sys.exit(1)
    print(f"\nAll {len(configs) * len(args.test_signers) * len(args.seeds)} runs completed.")


if __name__ == "__main__":
    main()
