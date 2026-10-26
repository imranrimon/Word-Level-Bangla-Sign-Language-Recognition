"""Every shipped config is loadable: fine-tune configs only use keys that main.py declares
(main.py rejects unknown keys with WRONG ARG) and build their model on CPU; pretraining
configs build the exact backbone + ShubertPretrainer that main_pretrain.py trains and run one
forward/backward pass on random input. No data needed.
"""
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import yaml

torch.set_num_threads(1)
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

ALL = sorted(glob.glob(str(ROOT / "config" / "**" / "*.yaml"), recursive=True))
PRETRAIN = [p for p in ALL if "pretrain" in os.path.basename(p)]
FINETUNE = [p for p in ALL if p not in PRETRAIN]
_id = lambda p: os.path.relpath(p, ROOT / "config")


def test_configs_found():
    assert len(PRETRAIN) >= 20 and len(FINETUNE) >= 25


def test_main_imports_without_wandb():
    import main
    assert hasattr(main, "get_parser")


@pytest.mark.parametrize("path", FINETUNE, ids=_id)
def test_finetune_config_keys_are_declared_and_model_builds(path):
    import main
    known = set(vars(main.get_parser().parse_args([])).keys())
    cfg = yaml.safe_load(open(path))
    unknown = sorted(set(cfg) - known)
    assert not unknown, f"{_id(path)}: keys not declared in main.py: {unknown}"
    Model = main.import_class(cfg["model"])
    m = Model(**cfg["model_args"]).eval()
    a = cfg["model_args"]
    x = torch.randn(1, a.get("in_channels", 3), 16, a["num_point"], a.get("num_person", 1))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, a["num_class"]) and torch.isfinite(y).all()


@pytest.mark.parametrize("path", PRETRAIN, ids=_id)
def test_pretrain_config_builds_and_steps(path):
    from main_pretrain import build_pretrainer

    torch.manual_seed(0)
    cfg = yaml.safe_load(open(path))
    backbone, pretr, use_lang, bargs, p = build_pretrainer(cfg, torch.device("cpu"))
    assert bargs.get("stride_between_stages") is False, "SSL needs T_out == T_in"
    N, T = 4, 24
    x = torch.randn(N, bargs.get("in_channels", 3), T, bargs["num_point"], bargs.get("num_person", 1))
    codes = torch.randint(0, int(p["num_codes"]), (N, T))
    codes[1] = codes[0]
    if use_lang:
        n_lang = max(int(p.get("num_languages", 0)), 2)
        if p.get("source_distances") is not None:
            n_lang = max(n_lang, len(p["source_distances"]))
        lang = torch.arange(N) % n_lang
        if float(p.get("lambda_coral", 0.0)) > 0:
            lang[0] = int(p.get("target_lang", 0))
        loss, aux = pretr(x, codes, lang)
    else:
        loss, aux = pretr(x, codes)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in backbone.parameters())
