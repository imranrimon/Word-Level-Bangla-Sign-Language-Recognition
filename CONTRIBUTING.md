# Contributing

Issues and pull requests are welcome. Please keep changes focused and reproducible.

## Development setup

```bash
conda env create -f environment.yml && conda activate bdsl_graph   # or: pip install -r requirements.txt
make setup-dev        # pytest, ruff, pre-commit hook
make check            # lint + tests + no-data reproduction of the paper's numbers (what CI runs)
```

## Guidelines

- **The pre-registered analyses are frozen.** Do not change `tools/p1_refit/refit.py`,
  `tools/p1_fix/refit_v2.py` or the files under `docs/p1_refit_2026-10/` and `docs/p1_fix_2026-10/`
  in a way that changes their output on the archived results; `make reproduce` must stay green.
  New analyses go into new scripts and are labelled as not pre-registered.
- **New config keys** must be declared as `main.py` argparse flags first; unknown YAML keys are
  rejected (`tests/test_configs.py` checks every shipped config).
- **Configs are loaded by dotted path** (`model: model.block_gcn.Model`,
  `feeder: feeders.feeder.Feeder`); keep module paths stable or update every config.
- **New corpora** must pass `tools/check_pose_geometry.py` before they are used, and missing joints
  (stored as 0) must be masked before speeds, distances or other derived quantities are computed.
- **A run counts only if it finished:** check with `tools/check_run_complete.py` before tabulating.
- **Data hygiene:** never commit data, checkpoints, score dumps or anything derived from the source
  videos (`.gitignore` covers `data/`, `work_dir/`, `*.npy`, `*.npz`, `*.pkl`, `*.pt`).
- Tests must run on CPU without data (skip cleanly when a dataset is absent).
