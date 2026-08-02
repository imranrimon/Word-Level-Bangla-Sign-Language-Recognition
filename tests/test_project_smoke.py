import importlib
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False


def test_center_choose_is_deterministic_and_padding_preserves_zeros():
    from feeders import tools

    data = np.zeros((3, 10, 2, 1), dtype=np.float32)
    for frame in range(10):
        data[:, frame, :, :] = frame + 1

    cropped = tools.center_choose(data, 4)
    assert cropped.shape == (3, 4, 2, 1)
    assert cropped[0, :, 0, 0].tolist() == [4, 5, 6, 7]

    padded_source = data.copy()
    padded_source[:, 3:, :, :] = 0
    padded = tools.center_choose(padded_source, 5)
    assert padded.shape == (3, 5, 2, 1)
    assert padded[0, :, 0, 0].tolist() == [1, 2, 3, 0, 0]


def test_generate_bone_data_uses_graph_edges(tmp_path):
    from preprocessing.generate_bone_data import generate_bone_data

    data = np.zeros((1, 3, 2, 27, 1), dtype=np.float32)
    data[0, :, :, 0, 0] = np.array([[1, 1], [2, 2], [3, 3]])
    data[0, :, :, 1, 0] = np.array([[4, 4], [6, 6], [8, 8]])

    input_path = tmp_path / "joint.npy"
    output_path = tmp_path / "bone.npy"
    np.save(input_path, data)

    generate_bone_data(str(input_path), str(output_path))
    bone = np.load(output_path)

    np.testing.assert_allclose(bone[0, :, :, 1, 0], data[0, :, :, 1, 0] - data[0, :, :, 0, 0])
    np.testing.assert_allclose(bone[0, :, :, 0, 0], 0)


def test_cli_run_paths_respect_explicit_values():
    from main import get_parser, resolve_run_paths

    parser = get_parser()
    args = parser.parse_args(["--config", "config/bdsl.yaml", "-Experiment_name", "demo"])
    resolved = resolve_run_paths(args)
    assert resolved.work_dir.endswith(os.path.join("work_dir", "demo"))
    assert resolved.model_saved_name.endswith(os.path.join("work_dir", "demo", "demo_model"))

    args = parser.parse_args([
        "--config", "config/bdsl.yaml",
        "-Experiment_name", "demo",
        "--work-dir", "runs/demo",
        "-model_saved_name", "runs/demo/model",
    ])
    resolved = resolve_run_paths(args)
    assert resolved.work_dir == os.path.normpath("runs/demo")
    assert resolved.model_saved_name == os.path.normpath("runs/demo/model")


def test_preprocessing_modules_import_without_stale_openpose_symbol(monkeypatch):
    fake_mediapipe = types.SimpleNamespace(solutions=types.SimpleNamespace(holistic=types.SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)

    for module_name in ("preprocessing.preprocess_bdsl", "preprocessing.preprocess_bdsl_images"):
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        assert module is not None


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        ("model.pose_lstm.Model", {"num_class": 2, "num_point": 27, "num_person": 1, "in_channels": 3}),
        ("model.st_gcn_vanilla.Model", {"num_class": 2, "num_point": 27, "num_person": 1, "graph": "graph.sign_27.Graph", "graph_args": {"labeling_mode": "spatial"}, "in_channels": 3}),
        ("model.attention_gnn.Model", {"num_class": 2, "num_point": 27, "num_person": 1, "graph": "graph.sign_27.Graph", "graph_args": {"labeling_mode": "spatial"}, "in_channels": 3}),
        ("model.adaptive_gnn.Model", {"num_class": 2, "num_point": 27, "num_person": 1, "graph": "graph.sign_27.Graph", "graph_args": {"labeling_mode": "spatial"}, "in_channels": 3}),
        ("model.gnn_lstm.Model", {"num_class": 2, "num_point": 27, "num_person": 1, "graph": "graph.sign_27.Graph", "graph_args": {"labeling_mode": "spatial"}, "in_channels": 3}),
        ("model.gnn_transformer.Model", {"num_class": 2, "num_point": 27, "num_person": 1, "graph": "graph.sign_27.Graph", "graph_args": {"labeling_mode": "spatial"}, "in_channels": 3}),
        ("model.twin_attention.Model", {
            "num_class": 2, "num_point": 27, "num_person": 1,
            "graph": "graph.sign_27.Graph", "graph_args": {"labeling_mode": "spatial"},
            "groups": 1, "block_size": 3, "inner_dim": 8, "depth": 1,
            "s_num_heads": 1, "t_num_heads": 1, "window_size": 16,
            "t_depths": [1], "wss": [2], "sr_ratios": [1],
            "drop_layers": 0, "s_pos_emb": False,
        }),
    ],
)
def test_dummy_forward_pass_per_model_family(model_cls, kwargs):
    module_name, class_name = model_cls.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class(**kwargs).eval()

    x = torch.randn(2, 3, 16, 27, 1)
    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (2, 2)
