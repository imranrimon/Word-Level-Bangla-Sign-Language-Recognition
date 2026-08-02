"""Smoke tests for the B3 co-training ablation (multihead BlockGCN + main_cotrain)."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from model.block_gcn_multihead import Model as MultiheadModel


TINY_ARGS = dict(
    num_class=60,
    num_class_aux=401,
    num_point=27,
    num_person=1,
    graph="graph.sign_27.Graph",
    graph_args={"labeling_mode": "spatial"},
    in_channels=3,
    num_blocks=(1, 1),
    stage_channels=(8, 16),
    max_distance=8,
)


def test_multihead_forward_shapes():
    model = MultiheadModel(**TINY_ARGS)
    x = torch.randn(2, 3, 16, 27, 1)
    logits_t, logits_a = model(x)
    assert logits_t.shape == (2, 60)
    assert logits_a.shape == (2, 401)


def test_multihead_masked_loss_step():
    torch.manual_seed(0)
    model = MultiheadModel(**TINY_ARGS)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ce = torch.nn.CrossEntropyLoss()

    x = torch.randn(6, 3, 16, 27, 1)
    label = torch.tensor([3, 59, 0, 400, 17, 250])
    domain = torch.tensor([0, 0, 0, 1, 1, 1])  # first 3 target, last 3 aux

    logits_t, logits_a = model(x)
    mask_t, mask_a = domain == 0, domain == 1
    loss = ce(logits_t[mask_t], label[mask_t]) + ce(logits_a[mask_a], label[mask_a])
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    # Both heads and the shared backbone must receive gradient.
    assert model.fc_target.weight.grad is not None
    assert model.fc_aux.weight.grad is not None
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in backbone_grads)
    optimizer.step()


def test_backbone_state_dict_loads_into_plain_block_gcn():
    from model.block_gcn import Model as BlockGCN

    multi = MultiheadModel(**TINY_ARGS)
    plain_args = {k: v for k, v in TINY_ARGS.items() if k != "num_class_aux"}
    plain = BlockGCN(**plain_args)
    # Backbone-only export (what main_cotrain saves) must load into
    # block_gcn.Model for BPT-style reuse; only the classifier fc differs.
    missing, unexpected = plain.load_state_dict(multi.backbone.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("fc.") for k in missing)


def test_cotrain_concat_resamples_aux():
    from main_cotrain import _CotrainConcat

    class _FakeFeeder:
        def __init__(self, n, label):
            self.n, self.label_value = n, label

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return np.zeros((3, 4, 27, 1), dtype=np.float32), self.label_value, i

    ds = _CotrainConcat(_FakeFeeder(10, 0), _FakeFeeder(100, 1), aux_fraction=0.2, seed=0)
    assert len(ds) == 10 + 20
    data, label, domain = ds[0]
    assert domain == 0
    data, label, domain = ds[10]
    assert domain == 1
    first = set(ds.aux_indices.tolist())
    ds.resample()
    second = set(ds.aux_indices.tolist())
    assert len(second) == 20
    assert first != second  # resampling actually changes the aux subset
