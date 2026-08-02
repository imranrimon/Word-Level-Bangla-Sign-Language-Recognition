"""Smoke tests for Path 2 (handshape KD).

Verifies (without running real DINOv2 / MediaPipe / BlockGCN-on-real-data):
  * `block_gcn_kd.Model` forward returns (logits, student_proj) with the right shapes.
  * Both heads receive gradient when CE + cosine-KD loss is backpropped.
  * `KDFeeder` correctly pairs pose and teacher arrays row-by-row, pools
    teacher over time, and yields a `(C_pose, T, V_pose, M)` pose tensor +
    a `(2 * D_teacher,)` teacher vector + a label.
  * KDFeeder fails loud when pose and teacher row counts disagree.
"""

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False


BACKBONE_KWARGS = dict(
    graph="graph.sign_27.Graph",
    graph_args={"labeling_mode": "spatial"},
    in_channels=3,
    num_point=27,
    num_person=1,
)


def test_block_gcn_kd_forward_shapes():
    from model.block_gcn_kd import Model

    torch.manual_seed(0)
    m = Model(num_class=5, num_blocks=(1, 1), stage_channels=(32, 64),
              teacher_dim=128, **BACKBONE_KWARGS).eval()
    x = torch.randn(2, 3, 16, 27, 1)
    with torch.no_grad():
        logits, proj = m(x)
    assert logits.shape == (2, 5)
    assert proj.shape == (2, 128)
    assert torch.isfinite(logits).all() and torch.isfinite(proj).all()


def test_block_gcn_kd_combined_loss_backward():
    from model.block_gcn_kd import Model

    torch.manual_seed(0)
    m = Model(num_class=5, num_blocks=(1, 1), stage_channels=(32, 64),
              teacher_dim=64, **BACKBONE_KWARGS)
    x = torch.randn(2, 3, 16, 27, 1)
    teacher = torch.randn(2, 64)
    label = torch.tensor([0, 3])

    logits, proj = m(x)
    ce = F.cross_entropy(logits, label)
    kd = 1.0 - F.cosine_similarity(proj, teacher, dim=-1).mean()
    loss = ce + 1.0 * kd
    loss.backward()

    # Classifier head has gradient
    assert m.fc.weight.grad is not None and m.fc.weight.grad.abs().sum() > 0
    # KD projection head has gradient
    assert m.kd_proj.weight.grad is not None and m.kd_proj.weight.grad.abs().sum() > 0
    # Backbone has gradient (KD pushes it too)
    backbone_grads = [p.grad for p in m.backbone.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(backbone_grads) > 0


def _make_kd_fixture(tmp_path, n_samples=4, c_pose=3, t_max=8, v_pose=27,
                     d_teacher=16, v_regions=3):
    """Write tiny pose + teacher NPYs + matching label pkls to tmp_path."""
    pose = np.random.RandomState(0).randn(n_samples, c_pose, t_max, v_pose, 1).astype(np.float32)
    teacher = np.random.RandomState(1).randn(n_samples, d_teacher, t_max, v_regions, 1).astype(np.float32)
    labels = list(range(n_samples))
    sample_names = [f"sample_{i}.mp4" for i in range(n_samples)]

    pose_path = tmp_path / "pose_data.npy"
    teacher_path = tmp_path / "teacher_data.npy"
    np.save(pose_path, pose)
    np.save(teacher_path, teacher)
    pose_lbl = tmp_path / "pose_label.pkl"
    teacher_lbl = tmp_path / "teacher_label.pkl"
    with open(pose_lbl, "wb") as f:
        pickle.dump((sample_names, labels), f)
    with open(teacher_lbl, "wb") as f:
        pickle.dump((sample_names, labels), f)
    return str(pose_path), str(pose_lbl), str(teacher_path), str(teacher_lbl)


def test_kd_feeder_yields_paired_tuples(tmp_path):
    from feeders.kd_feeder import KDFeeder

    pp, pl, tp, tl = _make_kd_fixture(tmp_path, t_max=16, d_teacher=16)
    feeder = KDFeeder(
        pose_data_path=pp, pose_label_path=pl,
        teacher_data_path=tp, teacher_label_path=tl,
        random_choose=False, normalization=False,
        window_size=8,
        teacher_region_indices=(0, 1),
    )
    assert len(feeder) == 4

    pose, teacher, label = feeder[0]
    assert isinstance(pose, np.ndarray) or torch.is_tensor(pose)
    assert teacher.shape == (16 * 2,)   # D_teacher * len(region_indices)
    assert isinstance(label, int) or torch.is_tensor(label)


def test_kd_feeder_fails_on_size_mismatch(tmp_path):
    from feeders.kd_feeder import KDFeeder

    pp, pl, _, _ = _make_kd_fixture(tmp_path, n_samples=4, t_max=8, d_teacher=16)
    # Make a teacher NPY of wrong length
    bad_teacher = np.random.randn(3, 16, 8, 3, 1).astype(np.float32)  # 3 != 4
    bad_teacher_path = tmp_path / "bad_teacher.npy"
    np.save(bad_teacher_path, bad_teacher)
    with pytest.raises(ValueError, match="length mismatch"):
        KDFeeder(
            pose_data_path=pp, pose_label_path=pl,
            teacher_data_path=str(bad_teacher_path),
            random_choose=False, normalization=False, window_size=4,
        )


def test_kd_loss_modes_consistent_and_gradients_flow():
    """Audit fix #11: each kd_loss mode produces a finite scalar with gradient.

    - cosine and mse_normalized should be ~ equal when input student/teacher
      are already L2-normalized (proxy for "normalized MSE behaves like a
      cosine variant").
    - raw mse should be invariant to nothing — it's the magnitude-sensitive
      one we warn against in the config.
    """
    from path2_handshape_kd.train_kd import _kd_loss

    torch.manual_seed(0)
    s = torch.randn(8, 32, requires_grad=True)
    t = torch.randn(8, 32)

    # Each mode produces a finite scalar with a gradient on student.
    for mode in ("cosine", "mse", "mse_normalized"):
        s.grad = None
        loss = _kd_loss(s, t, kind=mode)
        assert loss.dim() == 0 and torch.isfinite(loss)
        loss.backward()
        assert s.grad is not None and torch.isfinite(s.grad).all()

    # When inputs are already L2-normalized, mse_normalized and 2*(1-cosine)
    # are mathematically equal (||a - b||^2 = 2 - 2 <a, b> for unit a, b).
    s_n = F.normalize(s.detach(), dim=-1)
    t_n = F.normalize(t, dim=-1)
    l_cos = _kd_loss(s_n, t_n, kind="cosine")
    l_mse = _kd_loss(s_n, t_n, kind="mse_normalized")
    # Per-sample: mse_normalized = mean(||a - b||^2 / D) where D = feat_dim;
    # 2 - 2 <a, b> = mean ||a - b||^2 per sample. So the identity:
    #     mse_normalized * D == 2 * cosine_loss * D / D == 2 * cosine_loss
    # gives us: mse_normalized * D / 2 == cosine_loss (for unit-norm inputs).
    D = s_n.shape[-1]
    assert torch.allclose(l_mse * D / 2.0, l_cos, atol=1e-5), (
        f"L2-normalized MSE * D/2 ({l_mse * D / 2.0}) should equal cosine "
        f"loss ({l_cos}) for unit-norm inputs"
    )


def test_kd_loss_rejects_unknown_kind():
    from path2_handshape_kd.train_kd import _kd_loss

    s = torch.randn(2, 4)
    t = torch.randn(2, 4)
    with pytest.raises(ValueError, match="unknown KD loss kind"):
        _kd_loss(s, t, kind="not_a_real_mode")


def test_kd_feeder_pools_teacher_over_only_nonzero_frames(tmp_path):
    """If teacher has zero-padded tail frames, pool should average only the
    non-padded ones — preserves valid-frame mean."""
    from feeders.kd_feeder import KDFeeder

    pp, pl, tp, _ = _make_kd_fixture(tmp_path, n_samples=2, t_max=10, d_teacher=4, v_regions=2)
    # Overwrite teacher with a known pattern: first 6 frames = ones, rest = zeros.
    teacher = np.zeros((2, 4, 10, 2, 1), dtype=np.float32)
    teacher[:, :, :6, :, :] = 1.0
    np.save(tp, teacher)
    feeder = KDFeeder(
        pose_data_path=pp, pose_label_path=pl,
        teacher_data_path=tp,
        random_choose=False, normalization=False, window_size=4,
        teacher_region_indices=(0, 1),
    )
    _pose, t_vec, _label = feeder[0]
    # All 8 = 4 channels * 2 regions positions should equal 1.0 (the average
    # over the 6 nonzero frames), not 0.6 (average that includes the zeros).
    assert torch.allclose(t_vec, torch.ones(8))
