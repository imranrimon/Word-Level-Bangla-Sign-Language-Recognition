# Common tasks. Run from the repository root; override the interpreter with PYTHON=...
PYTHON ?= python
RUFF   ?= ruff

.PHONY: help setup setup-pip setup-dev test lint reproduce reproduce-full figures refit-v2 refit-v1 completeness check clean

help:  ## list targets
	@grep -E "^[a-z0-9-]+:.*## " $(MAKEFILE_LIST) | sed -E "s/:.*## /\t/" | expand -t 20

setup:  ## create the conda env `bdsl_graph` (GPU, CUDA 11.8)
	conda env create -f environment.yml

setup-pip:  ## pip install runtime deps into the active env (install torch first, see requirements.txt)
	$(PYTHON) -m pip install -r requirements.txt

setup-dev:  ## pip install test/lint tooling and the pre-commit hook
	$(PYTHON) -m pip install "pytest>=7" "ruff==0.6.9" pre-commit
	pre-commit install

test:  ## unit and smoke tests (CPU, no data needed)
	CUDA_VISIBLE_DEVICES="" $(PYTHON) -m pytest

lint:  ## ruff (rules in pyproject.toml)
	$(RUFF) check .

reproduce:  ## re-run the analyses on the archived results and diff against the paper's numbers (no data)
	CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg $(PYTHON) tools/reproduce.py

reproduce-full:  ## as reproduce, plus every pre-registered sensitivity analysis of the 7-seed refit
	CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg $(PYTHON) tools/reproduce.py --full

figures:  ## render the paper figures into results/figures/
	MPLBACKEND=Agg $(PYTHON) tools/paper/make_figs.py

refit-v2:  ## pre-registered v2 analysis -> results/refit_v2/ (reads results/results_p1fix_v2.csv)
	$(PYTHON) tools/p1_fix/refit_v2.py --out results/refit_v2

refit-v1:  ## pre-registered 7-seed refit -> results/refit_v1/ (reads results/results_final.csv)
	$(PYTHON) tools/p1_refit/refit.py --pseeds 0-6 --out results/refit_v1 \
	    --alt-dist docs/p1_refit_2026-10/realpool_distances.json

completeness:  ## check every fine-tune the analyses read reached its final epoch (needs work_dir/)
	$(PYTHON) tools/check_run_complete.py --p1-sweep

check: lint test reproduce  ## everything CI runs

clean:  ## remove regenerated outputs and caches
	rm -rf results/figures results/refit_v1 results/refit_v2 .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
