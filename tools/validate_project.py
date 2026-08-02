import argparse
import ast
import csv
import importlib.util
import os
import pickle
import re
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULT_COLUMNS = [
    "Timestamp", "Experiment", "Epoch", "Top1_Acc", "Top5_Acc",
    "Top5_Policy", "WorkDir",
]
REQUIRED_IMPORTS = {
    "yaml": "pyyaml",
    "numpy": "numpy",
    "torch": "pytorch",
    "timm": "timm",
    "einops": "einops",
    "wandb": "wandb",
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "torch_geometric": "pyg",
}


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


class Reporter:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def print(self):
        for message in self.errors:
            print(f"ERROR: {message}")
        for message in self.warnings:
            print(f"WARNING: {message}")
        if not self.errors and not self.warnings:
            print("Project validation passed with no issues.")
        else:
            print(f"Validation complete: {len(self.errors)} error(s), {len(self.warnings)} warning(s).")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def collect_config_paths(include_legacy_wlasl=False):
    paths = set()
    experiments_path = ROOT / "experiments.yaml"
    if experiments_path.exists():
        data = load_yaml(experiments_path)
        for exp in data.get("experiments", []):
            config = exp.get("config")
            if config:
                paths.add((ROOT / config).resolve())

    for path in (ROOT / "config").glob("bdsl*.yaml"):
        paths.add(path.resolve())

    if include_legacy_wlasl:
        for path in (ROOT / "config" / "WLASL").glob("**/*.yaml"):
            paths.add(path.resolve())

    return sorted(paths)


def config_model_signature(model_path):
    parts = model_path.split(".")
    if len(parts) < 3 or parts[0] != "model" or parts[-1] != "Model":
        return None, False

    py_path = ROOT / "model" / f"{parts[1]}.py"
    if not py_path.exists():
        return None, False

    tree = ast.parse(py_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            init = next((item for item in node.body
                         if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
            if not init:
                return set(), False
            args = {arg.arg for arg in init.args.args[1:]}
            has_kwargs = init.args.kwarg is not None
            return args, has_kwargs
    return None, False


def _validate_pretrain_config(path, data, reporter):
    """Schema for SHuBERT-style pretraining configs (used by main_pretrain.py).

    Expects: backbone (dotted Model path), backbone_args (dict), and a
    `pretrain:` section with manifest + targets + num_codes + mask_ratio +
    window_size. No train_feeder_args / test_feeder_args / num_class.
    """
    pretrain = data.get("pretrain") or {}
    for key in ("manifest", "targets", "num_codes", "mask_ratio",
                "window_size", "feat_dim"):
        if key not in pretrain:
            reporter.error(f"{rel(path)} missing pretrain.{key}")
    for key in ("manifest", "targets"):
        value = pretrain.get(key)
        if value:
            if not (ROOT / value).exists():
                reporter.warn(
                    f"{rel(path)} references missing pretrain.{key}: {value} "
                    "(may be produced by build_ssl_pool_manifest.py / "
                    "compute_pretrain_targets.py at run-time)"
                )

    backbone = data.get("backbone")
    if backbone:
        sig, _kw = config_model_signature(backbone)
        if sig is None:
            reporter.error(f"{rel(path)} references unknown backbone: {backbone}")
    else:
        reporter.error(f"{rel(path)} missing backbone")


def validate_config(path, reporter):
    data = load_yaml(path)

    # SHuBERT-style pretraining configs use a different schema than the
    # classification YAMLs main.py consumes. Detect via `pretrain:` key.
    if "pretrain" in data and "train_feeder_args" not in data:
        return _validate_pretrain_config(path, data, reporter)

    model_args = data.get("model_args") or {}

    for section in ("train_feeder_args", "test_feeder_args"):
        feeder_args = data.get(section) or {}
        for key in ("data_path", "label_path"):
            value = feeder_args.get(key)
            if not value:
                reporter.error(f"{rel(path)} missing {section}.{key}")
                continue
            resolved = ROOT / value
            if not resolved.exists():
                # Downgrade to warning for known TODO paths (DINOv2 outputs
                # produced by extract_dinov2_features.py and Path 1) — these
                # are expected-missing in a fresh checkout.
                rel_value = str(value).replace("\\", "/")
                is_pending = any(
                    rel_value.startswith(p) for p in
                    ("./data/bdsl_si_dino/", "data/bdsl_si_dino/",
                     "./data/bdsl_si_bdino/", "data/bdsl_si_bdino/")
                )
                if is_pending:
                    reporter.warn(f"{rel(path)} references pending {section}.{key}: "
                                  f"{value} (produced by Stage B / Path 1)")
                else:
                    reporter.error(f"{rel(path)} references missing {section}.{key}: {value}")

    train_args = data.get("train_feeder_args") or {}
    test_args = data.get("test_feeder_args") or {}
    if test_args.get("random_choose", False):
        reporter.error(f"{rel(path)} has random_choose=True in test_feeder_args; validation should be deterministic")
    for key in ("window_size", "normalization", "is_vector", "lap_pe"):
        train_value = train_args.get(key, False)
        test_value = test_args.get(key, False)
        if train_value != test_value:
            reporter.warn(f"{rel(path)} train/test feeder mismatch for {key}: {train_value!r} vs {test_value!r}")
    if "bone" in str(train_args.get("data_path", "")).lower() and not test_args.get("is_vector", False):
        reporter.error(f"{rel(path)} uses bone data but test_feeder_args.is_vector is not true")

    model = data.get("model")
    signature, has_kwargs = config_model_signature(model or "")
    if signature is None:
        reporter.error(f"{rel(path)} references unknown model: {model}")
    elif not has_kwargs:
        extra = sorted(set(model_args) - signature)
        if extra:
            reporter.error(f"{rel(path)} passes unsupported model_args for {model}: {extra}")

    validate_label_counts(path, data, reporter)


def validate_label_counts(path, data, reporter):
    train_args = data.get("train_feeder_args") or {}
    model_args = data.get("model_args") or {}
    data_path = train_args.get("data_path")
    label_path = train_args.get("label_path")
    if not data_path or not label_path:
        return

    data_file = ROOT / data_path
    label_file = ROOT / label_path
    if not data_file.exists() or not label_file.exists():
        return

    try:
        data_array = np.load(data_file, mmap_mode="r")
        with open(label_file, "rb") as f:
            sample_names, labels = pickle.load(f)
    except Exception as exc:
        reporter.error(f"{rel(path)} could not load labels/data: {exc}")
        return

    if len(data_array) != len(labels) or len(sample_names) != len(labels):
        reporter.error(
            f"{rel(path)} data/label length mismatch: "
            f"N={len(data_array)}, labels={len(labels)}, sample_names={len(sample_names)}"
        )

    num_class = model_args.get("num_class")
    unique_labels = set(labels)
    if num_class is not None and len(unique_labels) != num_class:
        reporter.error(f"{rel(path)} num_class={num_class} but labels contain {len(unique_labels)} class(es)")
    if labels and (min(labels) != 0 or max(labels) != len(unique_labels) - 1):
        reporter.warn(f"{rel(path)} labels are not a contiguous 0-based range")


def validate_experiments(reporter):
    experiments_path = ROOT / "experiments.yaml"
    if not experiments_path.exists():
        reporter.warn("experiments.yaml not found")
        return
    data = load_yaml(experiments_path)
    for exp in data.get("experiments", []):
        config = exp.get("config")
        if config and not (ROOT / config).exists():
            reporter.error(f"experiments.yaml references missing config: {config}")


def validate_results(reporter):
    csv_path = ROOT / "results_final.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        missing = [column for column in RESULT_COLUMNS if column not in header]
        if missing:
            reporter.warn(f"results_final.csv is missing newer reporting columns: {missing}")

    stale_refs = []
    for path in [ROOT / "tools", ROOT / "scripts", ROOT / "readme.md"]:
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in {".py", ".bat", ".md"}:
                if candidate.resolve() == Path(__file__).resolve():
                    continue
                text = candidate.read_text(errors="ignore")
                if re.search(r"(?<!_)results\.csv\b", text):
                    stale_refs.append(rel(candidate))
    if stale_refs:
        reporter.warn(f"stale results.csv references remain in: {sorted(stale_refs)}")


def validate_dependencies(reporter):
    for import_name, package_name in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(import_name) is None:
            reporter.warn(f"missing dependency import '{import_name}' (package: {package_name})")


def main():
    parser = argparse.ArgumentParser(description="Validate SLGTformer project configuration and environment.")
    parser.add_argument("--include-legacy-wlasl", action="store_true",
                        help="also validate legacy WLASL configs and downloadable data paths")
    parser.add_argument("--skip-dependencies", action="store_true",
                        help="skip dependency import availability checks")
    parser.add_argument("--strict-dependencies", action="store_true",
                        help="treat missing dependencies as errors")
    args = parser.parse_args()

    reporter = Reporter()
    validate_experiments(reporter)
    for config_path in collect_config_paths(args.include_legacy_wlasl):
        validate_config(config_path, reporter)
    validate_results(reporter)

    before_dependency_warnings = len(reporter.warnings)
    if not args.skip_dependencies:
        validate_dependencies(reporter)
        if args.strict_dependencies and len(reporter.warnings) > before_dependency_warnings:
            for warning in reporter.warnings[before_dependency_warnings:]:
                reporter.error(warning)

    reporter.print()
    raise SystemExit(1 if reporter.errors else 0)


if __name__ == "__main__":
    main()
