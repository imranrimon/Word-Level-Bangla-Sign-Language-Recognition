"""Smoke tests for Option C: BlockGCN + RQE + SHuBERT-style masked pretraining.

Each test isolates one component; together they verify that:
  * BlockGCN satisfies the repo's standard (N,C,T,V,M)->(N,num_class) contract
    and backprop flows through.
  * RQE produces the expected shape and is invariant to global translation
    of the input keypoints (the whole point of anchoring to the shoulders).
  * The BlockGCN + RQE combination forwards end-to-end.
  * The SHuBERT-style masking wrapper masks the right positions, computes a
    finite loss on masked positions only, and passes gradients back into
    both the backbone and the mask token.
  * Backbone weights from a pretraining step can be loaded into a fresh
    classifier-only model (the pretrain -> fine-tune handoff).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

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


def _set_seed(seed=0):
    torch.manual_seed(seed)


def test_block_gcn_forward():
    from model.block_gcn import Model

    _set_seed()
    m = Model(
        num_class=5, num_blocks=(1, 1), stage_channels=(32, 64), **BACKBONE_KWARGS
    ).eval()
    x = torch.randn(2, 3, 16, 27, 1)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_block_gcn_backward():
    from model.block_gcn import Model

    _set_seed()
    m = Model(num_class=5, num_blocks=(1, 1), stage_channels=(32, 64), **BACKBONE_KWARGS)
    x = torch.randn(2, 3, 16, 27, 1)
    loss = m(x).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_rqe_shape_and_translation_invariance():
    from model.rqe import RelativeQuantizationEncoding

    _set_seed()
    rqe = RelativeQuantizationEncoding(
        embed_dim=16, num_buckets=16, right_shoulder_idx=1, left_shoulder_idx=2
    ).eval()
    x = torch.randn(2, 3, 16, 27, 1)
    with torch.no_grad():
        emb = rqe(x)
    assert emb.shape == (2, 16, 16, 27)
    assert torch.isfinite(emb).all()

    # Uniform translation of all (x, y) coords must not change bucket assignments.
    shift = torch.zeros_like(x)
    shift[:, 0] = 0.37
    shift[:, 1] = -0.21
    with torch.no_grad():
        emb_shifted = rqe(x + shift)
    assert torch.equal(emb, emb_shifted)


def test_block_gcn_rqe_forward():
    from model.block_gcn_rqe import Model

    _set_seed()
    m = Model(
        num_class=5,
        num_blocks=(1, 1),
        stage_channels=(32, 64),
        rqe_embed_dim=32,
        rqe_num_buckets=16,
        **BACKBONE_KWARGS,
    ).eval()
    x = torch.randn(2, 3, 16, 27, 1)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_slgtformer_rqe_forward():
    """SLGTFormer + RQE variant must forward and accept keep_prob (harness-compat)."""
    from model.slgtformer_rqe import Model

    _set_seed()
    m = Model(
        num_class=5,
        num_point=27,
        num_person=1,
        graph="graph.sign_27.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=3,
        groups=1,
        block_size=3,
        inner_dim=8,
        depth=1,
        s_num_heads=1,
        t_num_heads=1,
        window_size=16,
        t_depths=[1],
        wss=[2],
        sr_ratios=[1],
        drop_layers=0,
        s_pos_emb=False,
        rqe_embed_dim=8,
        rqe_num_buckets=8,
    ).eval()
    x = torch.randn(2, 3, 16, 27, 1)
    with torch.no_grad():
        y = m(x, keep_prob=0.9)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# SHuBERT-style masked pretraining plumbing
# ---------------------------------------------------------------------------

class _TimeFeatureBackbone(nn.Module):
    """Adapter: runs BlockGCN.data_bn + layers + V/M pooling, returns (N, T, feat_dim).

    We need per-time-step features for masked code prediction, not the pooled
    (N, feat_dim) vector BlockGCN.Model returns when return_features=True.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = backbone.feat_dim

    def forward(self, x):
        N, C, T, V, M = x.shape
        bb = self.backbone
        y = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        y = bb.data_bn(y)
        y = y.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for layer in bb.layers:
            y = layer(y)
        # (N*M, Cf, T', V)  -- use no-downsample backbone so T' == T
        y = y.mean(dim=-1)                                        # (N*M, Cf, T')
        y = y.view(N, M, -1, y.size(-1)).mean(dim=1)              # (N, Cf, T')
        return y.transpose(1, 2).contiguous()                     # (N, T', Cf)


def _make_no_downsample_backbone():
    # single-stage backbone -> no stride=2 transitions, so T_out == T_in.
    from model.block_gcn import Model
    return Model(num_class=5, num_blocks=(2,), stage_channels=(32,), **BACKBONE_KWARGS)


def test_shubert_apply_mask_matches_indices():
    from model.shubert_pretrain import ShubertPretrainer

    _set_seed()
    bb = _TimeFeatureBackbone(_make_no_downsample_backbone())
    pretr = ShubertPretrainer(
        backbone=bb, feat_dim=bb.feat_dim, num_codes=32, mask_ratio=0.3
    )
    x = torch.randn(2, 3, 20, 27, 1)
    x_masked, mask = pretr.apply_mask(x)

    assert x_masked.shape == x.shape
    assert mask.shape == (20,)

    # Only masked time steps should differ from the input.
    diff_per_t = (x_masked - x).abs().sum(dim=(0, 1, 3, 4))       # (T,)
    assert (diff_per_t[mask] > 0).all()
    assert (diff_per_t[~mask] == 0).all()


def test_shubert_loss_is_finite_and_gradients_flow():
    from model.shubert_pretrain import ShubertPretrainer

    _set_seed()
    backbone_model = _make_no_downsample_backbone()
    bb = _TimeFeatureBackbone(backbone_model)
    pretr = ShubertPretrainer(
        backbone=bb, feat_dim=bb.feat_dim, num_codes=32, mask_ratio=0.3
    )

    x = torch.randn(2, 3, 16, 27, 1)
    targets = torch.randint(0, 32, (2, 16))
    loss, aux = pretr(x, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert aux["num_masked"] > 0

    loss.backward()

    # Loss must push gradient into the pretraining head...
    assert pretr.head.weight.grad is not None
    assert pretr.head.weight.grad.abs().sum() > 0
    # ...into the mask token...
    assert pretr.mask_token.grad is not None
    # ...and into the backbone (at least one parameter).
    backbone_grads = [
        p.grad for p in backbone_model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(backbone_grads) > 0


def test_shubert_xlingual_mechanism_gradients():
    """Cross-lingual mechanism (proposed): with lambda_adv/lambda_contrast > 0 the
    total loss must add the adversarial + contrastive terms and pass gradients
    into the language head, the contrastive projection head, AND the backbone."""
    from model.shubert_pretrain import ShubertPretrainer

    _set_seed()
    backbone_model = _make_no_downsample_backbone()
    bb = _TimeFeatureBackbone(backbone_model)
    pretr = ShubertPretrainer(
        backbone=bb, feat_dim=bb.feat_dim, num_codes=32, mask_ratio=0.3,
        num_languages=2, lambda_adv=0.5, lambda_contrast=0.5,
        proj_dim=16, code_sim_threshold=0.5,
    )
    assert pretr.lang_head is not None and pretr.proj_head is not None

    N, T = 4, 16
    x = torch.randn(N, 3, T, 27, 1)
    codes = torch.randint(0, 32, (N, T))
    codes[1] = codes[0]                       # guarantee one shared-codebook positive pair
    lang = torch.tensor([0, 1, 0, 1])

    total, aux = pretr(x, codes, lang)
    assert total.dim() == 0 and torch.isfinite(total)
    for k in ("mask_loss", "adv_loss", "contrast_loss"):
        assert k in aux and np.isfinite(aux[k])
    assert aux["contrast_loss"] > 0.0         # the crafted positive pair is active

    total.backward()
    assert pretr.lang_head.weight.grad is not None
    assert pretr.lang_head.weight.grad.abs().sum() > 0
    proj_grads = [p.grad for p in pretr.proj_head.parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(proj_grads) > 0
    bb_grads = [p.grad for p in backbone_model.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(bb_grads) > 0


def test_shubert_mechanism_off_is_backward_compatible():
    """Defaults (lambda_adv=lambda_contrast=0) must reproduce the pool-only
    ablation: no extra heads, 2-arg forward still works, total == mask_loss."""
    from model.shubert_pretrain import ShubertPretrainer

    _set_seed()
    bb = _TimeFeatureBackbone(_make_no_downsample_backbone())
    pretr = ShubertPretrainer(backbone=bb, feat_dim=bb.feat_dim, num_codes=32, mask_ratio=0.3)
    assert pretr.lang_head is None and pretr.proj_head is None

    x = torch.randn(2, 3, 16, 27, 1)
    codes = torch.randint(0, 32, (2, 16))
    total, aux = pretr(x, codes)               # 2-arg call, unchanged API
    assert torch.isfinite(total)
    assert "adv_loss" not in aux and "contrast_loss" not in aux
    assert abs(float(total) - aux["mask_loss"]) < 1e-6


def test_pretrain_feeder_returns_language(tmp_path):
    """With return_language=True the feeder yields (x, codes, lang), mapping
    WLASL->ASL and BdSL sources->BdSL."""
    import json

    from feeders.pretrain_feeder import PretrainFeeder, LANG_BDSL, LANG_ASL

    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = {"clips": []}
    targets = {}
    srcs = ["data/bdslw401_pose_cache_front", "data/wlasl_pose_cache"]
    for i, src in enumerate(srcs):
        path = cache / f"clip_{i}.npz"
        np.savez_compressed(path, data=np.random.randn(3, 40, 27, 1).astype(np.float32))
        key = str(path).replace("\\", "/")
        manifest["clips"].append({"path": key, "source": src})
        targets[key] = np.random.randint(0, 32, size=(40,)).astype(np.int32)
    mp = tmp_path / "m.json"
    with open(mp, "w") as f:
        json.dump(manifest, f)
    tp = tmp_path / "t.npz"
    np.savez_compressed(tp, keys=np.array(list(targets.keys()), dtype=object),
                        **{f"t_{j}": v for j, v in enumerate(targets.values())})

    feeder = PretrainFeeder(str(mp), str(tp), window_size=32, random_crop=False,
                            return_language=True)
    out0, out1 = feeder[0], feeder[1]
    assert len(out0) == 3
    x, t, lang = out0
    assert x.shape == (3, 32, 27, 1) and t.shape == (32,)
    assert int(out0[2]) == LANG_BDSL and int(out1[2]) == LANG_ASL


def test_pretrain_feeder_handles_unequal_pose_and_codes_lengths(tmp_path):
    """`compute_pretrain_targets.py` truncates codes to last non-zero pose
    frame, so per-clip codes are usually shorter than the pose cache. Feeder
    must trim both consistently rather than crash on a broadcast mismatch.
    """
    import json
    import os

    # Build a tiny synthetic SSL pool fixture
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {"clips": []}
    targets = {}
    pose_T_per_clip = [108, 60, 130]
    codes_T_per_clip = [101,  60,  90]   # codes shorter on clips 0 and 2
    for i, (t_pose, t_codes) in enumerate(zip(pose_T_per_clip, codes_T_per_clip)):
        path = cache_dir / f"clip_{i}.npz"
        np.savez_compressed(path, data=np.random.randn(3, t_pose, 27, 1).astype(np.float32))
        manifest["clips"].append({"path": str(path).replace("\\", "/"), "source": str(cache_dir)})
        targets[str(path).replace("\\", "/")] = np.random.randint(0, 32, size=(t_codes,)).astype(np.int32)

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    targets_path = tmp_path / "targets.npz"
    np.savez_compressed(
        targets_path,
        cluster_centers=np.zeros((32, 81), dtype=np.float32),
        keys=np.array(list(targets.keys()), dtype=object),
        **{f"t_{j}": v for j, v in enumerate(targets.values())},
    )

    from feeders.pretrain_feeder import PretrainFeeder
    feeder = PretrainFeeder(
        manifest_path=str(manifest_path),
        targets_path=str(targets_path),
        window_size=120,
        random_crop=False,
    )
    assert len(feeder) == 3
    # Each item must yield (3, 120, 27, 1) pose and (120,) codes regardless of
    # the underlying length mismatch. No exceptions allowed.
    for i in range(3):
        x, t = feeder[i]
        assert x.shape == (3, 120, 27, 1)
        assert t.shape == (120,)
        # Codes must be in the original cluster-id range (no -1, no >=32)
        assert int(t.min()) >= 0 and int(t.max()) < 32


def test_pretrain_target_feature_modes_shapes_and_temporal_signal():
    """Audit fix #1: per-frame pose-only clustering let SHuBERT solve the
    masked-prediction task by copying the neighbour's cluster ID. Default mode
    is now 'pose_motion' (pose + first-derivative). Verify:
      * each mode returns the right (T, D) shape,
      * pose_motion features at t differ MORE from features at t+1 than bare
        pose does, which is the whole point — neighbours are no longer
        interchangeable.
    """
    from preprocessing.compute_pretrain_targets import (
        _clip_to_per_frame_features, _feature_dim,
    )

    _set_seed()
    T = 20
    base = 3 * 27
    # Random-walk pose mirrors real BdSL statistics: highly correlated frame
    # to frame (people don't teleport) but with non-trivial per-frame deltas.
    # Linear drift would make pose-only and pose_motion equally distinct,
    # which is not the regime we care about.
    steps = 0.5 * np.random.randn(3, T, 27, 1).astype(np.float32)
    data = np.cumsum(steps, axis=1)

    for mode, expected_dim in [
        ("frame", base),
        ("pose_motion", 2 * base),
        ("window4", 4 * base),
        ("window8", 8 * base),
    ]:
        feats = _clip_to_per_frame_features(data, mode)
        assert feats.shape == (T, expected_dim), \
            f"{mode}: got {feats.shape}, expected ({T}, {expected_dim})"
        assert _feature_dim(mode) == expected_dim
        assert np.isfinite(feats).all()

    # The temporal-signal check: pose_motion neighbours must be more distinct
    # than bare-pose neighbours on the same input.
    pose = _clip_to_per_frame_features(data, "frame")
    posemo = _clip_to_per_frame_features(data, "pose_motion")
    neighbour_dist_pose = np.linalg.norm(pose[1:] - pose[:-1], axis=1).mean()
    neighbour_dist_posemo = np.linalg.norm(posemo[1:] - posemo[:-1], axis=1).mean()
    assert neighbour_dist_posemo > neighbour_dist_pose * 1.05, (
        f"pose_motion neighbours ({neighbour_dist_posemo:.4f}) should be more "
        f"distinct than bare pose neighbours ({neighbour_dist_pose:.4f}); "
        f"otherwise the SSL task remains trivially solvable."
    )


def test_pretrain_target_window_mode_boundary_padding():
    """window<K> mode zero-pads at clip boundaries — confirm shape and that
    boundary frames have at least one all-zero block (the out-of-range slot)."""
    from preprocessing.compute_pretrain_targets import _clip_to_per_frame_features

    _set_seed()
    T, C, V = 6, 3, 27
    data = np.ones((C, T, V, 1), dtype=np.float32)
    K = 4
    feats = _clip_to_per_frame_features(data, f"window{K}")
    base = C * V
    assert feats.shape == (T, K * base)
    # At t=0 with K=4, half_l=2 -> the first 2 slots must be zero, last 2 nonzero.
    assert np.all(feats[0, :2 * base] == 0)
    assert np.all(feats[0, 2 * base:] != 0)
    # At t=T-1 (last frame) the trailing slot is out of range -> zero.
    assert np.all(feats[T - 1, (K - 1) * base:K * base] == 0)


def test_shubert_pretrain_to_finetune_weight_transfer(tmp_path):
    from model.block_gcn import Model
    from model.shubert_pretrain import ShubertPretrainer

    _set_seed()
    kwargs = dict(num_class=5, num_blocks=(2,), stage_channels=(32,), **BACKBONE_KWARGS)
    pre_backbone = Model(**kwargs)
    pretr = ShubertPretrainer(
        backbone=_TimeFeatureBackbone(pre_backbone),
        feat_dim=pre_backbone.feat_dim,
        num_codes=16,
        mask_ratio=0.3,
    )

    # one pretraining update
    opt = torch.optim.SGD(pretr.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 12, 27, 1)
    targets = torch.randint(0, 16, (2, 12))
    loss, _ = pretr(x, targets)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # save the pretrained backbone only
    ckpt_path = tmp_path / "pretrained_backbone.pt"
    torch.save(pre_backbone.state_dict(), ckpt_path)

    # build a fresh classifier-only model, transfer weights strictly
    ft_backbone = Model(**kwargs)
    state = torch.load(ckpt_path, map_location="cpu")
    ft_backbone.load_state_dict(state, strict=True)  # raises on any mismatch

    # forward still produces well-shaped logits after transfer
    ft_backbone.eval()
    with torch.no_grad():
        y = ft_backbone(x)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
