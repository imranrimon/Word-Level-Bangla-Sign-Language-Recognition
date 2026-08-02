"""SHuBERT-style masked pretraining wrapper, with an optional CROSS-LINGUAL mechanism.

Base objective (always on): mask a subset of time steps with a learnable mask
token and predict a discrete pose code (k-means cluster id) per masked step via a
linear head — cross-entropy over masked positions only.

Cross-lingual mechanism (flag-gated; BOTH default OFF, so with the defaults this
class reproduces the pool-composition-only backbone that serves as the ablation):

  * Language-adversarial invariance  (``lambda_adv`` > 0):
      a gradient-reversal language classifier on the pooled clip feature makes the
      backbone learn pose representations INVARIANT to source language (ASL vs
      BdSL). This is the explicit mechanism behind ASL->BdSL transfer.

  * Cross-lingual contrastive alignment  (``lambda_contrast`` > 0):
      a SupCon-style loss that pulls together clips sharing pose structure, where
      "shared structure" is overlap in the SHARED k-means code vocabulary (so no
      cross-language alignment labels are needed). Because the codebook is common
      to both languages, an ASL clip and a BdSL clip with similar handshape/motion
      become a positive pair and get aligned.

Together: representations that are invariant to *which* language yet aligned on
*shared* sign structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_time_mask(T, mask_ratio, device):
    """Boolean mask of shape (T,); True marks positions to mask."""
    if T <= 0:
        raise ValueError("T must be positive")
    n_mask = max(1, int(round(T * mask_ratio)))
    idx = torch.randperm(T, device=device)[:n_mask]
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    mask[idx] = True
    return mask


class _GradReverse(torch.autograd.Function):
    """Identity forward; sign-flipped, alpha-scaled gradient backward (DANN)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return _GradReverse.apply(x, alpha)


class ShubertPretrainer(nn.Module):
    def __init__(
        self,
        backbone,
        feat_dim,
        num_codes,
        in_channels=3,
        num_point=27,
        mask_ratio=0.15,
        # --- cross-lingual mechanism (all default off) ---
        num_languages=0,
        lambda_adv=0.0,
        lambda_contrast=0.0,
        grl_alpha=1.0,
        proj_dim=128,
        contrast_temp=0.1,
        code_sim_threshold=0.5,
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        self.backbone = backbone
        self.feat_dim = int(feat_dim)
        self.num_codes = int(num_codes)
        self.mask_ratio = float(mask_ratio)
        self.mask_token = nn.Parameter(
            torch.zeros(1, in_channels, 1, num_point, 1)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.head = nn.Linear(feat_dim, num_codes)

        # --- cross-lingual mechanism ---
        self.num_languages = int(num_languages)
        self.lambda_adv = float(lambda_adv)
        self.lambda_contrast = float(lambda_contrast)
        self.grl_alpha = float(grl_alpha)
        self.contrast_temp = float(contrast_temp)
        self.code_sim_threshold = float(code_sim_threshold)
        # Language-adversarial head: only built if enabled (>=2 languages).
        self.lang_head = (
            nn.Linear(self.feat_dim, self.num_languages)
            if self.lambda_adv > 0 and self.num_languages >= 2 else None
        )
        # Contrastive projection head (2-layer MLP), only if enabled.
        self.proj_head = (
            nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, int(proj_dim)),
            )
            if self.lambda_contrast > 0 else None
        )

    def apply_mask(self, x):
        # x: (N, C, T, V, M)
        if x.dim() != 5:
            raise ValueError(f"expected 5D (N,C,T,V,M) input, got {x.dim()}D")
        N, C, T, V, M = x.shape
        mask = random_time_mask(T, self.mask_ratio, x.device)
        m = mask.view(1, 1, T, 1, 1).expand(N, C, T, V, M)
        token = self.mask_token.expand(N, C, T, V, M)
        x_masked = torch.where(m, token, x)
        return x_masked, mask

    def _adv_loss(self, clip_feat, lang):
        """Gradient-reversed language classification -> language-invariant feats."""
        logits = self.lang_head(grad_reverse(clip_feat, self.grl_alpha))
        return F.cross_entropy(logits, lang)

    def _contrastive_loss(self, clip_feat, target_codes):
        """SupCon over clip embeddings; positives = shared-codebook overlap.

        Positives are pairs whose k-means code HISTOGRAMS are cosine-similar
        beyond ``code_sim_threshold``. The codebook is shared across languages,
        so cross-language pairs of similar signs qualify — that is what makes the
        alignment cross-lingual, with no alignment labels.
        """
        N = clip_feat.size(0)
        if N < 2:
            return clip_feat.new_zeros(())
        z = F.normalize(self.proj_head(clip_feat), dim=1)          # (N, D)
        sim = z @ z.t() / self.contrast_temp                       # (N, N)

        # Per-clip code histogram (clamp guards padded/out-of-range codes; not in place).
        codes = target_codes.clamp(0, self.num_codes - 1)
        hist = torch.zeros(N, self.num_codes, device=z.device)
        hist.scatter_add_(1, codes, torch.ones_like(codes, dtype=hist.dtype))
        hist = F.normalize(hist, dim=1)                            # (N, num_codes)
        code_sim = hist @ hist.t()                                 # (N, N) cosine

        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        pos = (code_sim > self.code_sim_threshold) & ~eye          # (N, N) bool

        # SupCon log-prob with self excluded from the denominator.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()   # numerical stability
        exp = torch.exp(sim) * (~eye).to(sim.dtype)
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)

        pos_count = pos.sum(1)
        valid = pos_count > 0
        if not bool(valid.any()):
            return clip_feat.new_zeros(())
        per_anchor = -(pos.to(log_prob.dtype) * log_prob).sum(1)[valid] / pos_count[valid].to(log_prob.dtype)
        return per_anchor.mean()

    def forward(self, x, target_codes, lang=None):
        """
        x: (N, C, T, V, M)
        target_codes: (N, T) int64 k-means cluster assignments.
        lang: optional (N,) int64 source-language ids (0=BdSL, 1=ASL). Required
              only when the language-adversarial term is enabled.

        Returns (total_loss, aux) where aux breaks out the loss components.
        """
        if target_codes.shape != (x.shape[0], x.shape[2]):
            raise ValueError(
                "target_codes must have shape (N, T) = "
                f"({x.shape[0]}, {x.shape[2]}); got {tuple(target_codes.shape)}"
            )
        x_masked, mask = self.apply_mask(x)
        feats = self.backbone(x_masked)
        if feats.dim() != 3 or feats.size(-1) != self.feat_dim:
            raise RuntimeError(
                "backbone must return (N, T, feat_dim) = "
                f"(N, T, {self.feat_dim}); got {tuple(feats.shape)}"
            )
        if feats.size(1) != x.size(2):
            raise RuntimeError(
                f"backbone time dim {feats.size(1)} does not match input T "
                f"{x.size(2)}. SHuBERT-style masking requires T_out == T_in. "
                "For BlockGCN, pass `stride_between_stages: False` in the "
                "model_args of your pretraining config."
            )
        logits = self.head(feats)                                # (N, T, num_codes)
        masked_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if masked_idx.numel() == 0:
            mask_loss = logits.new_zeros(())
        else:
            logits_m = logits[:, masked_idx, :].reshape(-1, self.num_codes)
            targets_m = target_codes[:, masked_idx].reshape(-1)
            mask_loss = F.cross_entropy(logits_m, targets_m)

        total = mask_loss
        aux = {"num_masked": int(masked_idx.numel()), "mask_loss": mask_loss.item()}

        use_adv = self.lang_head is not None and lang is not None
        if use_adv or self.proj_head is not None:
            clip_feat = feats.mean(dim=1)                        # (N, feat_dim)
            if use_adv:
                adv = self._adv_loss(clip_feat, lang)
                total = total + self.lambda_adv * adv
                aux["adv_loss"] = adv.item()
            if self.proj_head is not None:
                con = self._contrastive_loss(clip_feat, target_codes)
                total = total + self.lambda_contrast * con
                aux["contrast_loss"] = con.item()

        return total, aux
