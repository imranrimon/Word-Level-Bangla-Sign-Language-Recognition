"""Distance-gated cross-lingual alignment weighting (P1, reviewer §13/§31/§32).

Turns a per-source pose-motion distance ``d_i`` (unlabeled MMD from
``tools/compute_lang_distance.py``) into a per-source alignment weight that
scales that source's cross-lingual adversarial + contrastive terms.

Three modes (config-selected via ``pretrain.gate_mode``):

  * ``"hardgate"`` (legacy DG-XL, DEFAULT when ``source_distances`` is set):
        w_i = exp(-(d_i - min_j d_j) / tau)
    MIN-NORMALISED and used as a RE-WEIGHTING gate (the alignment loss is a
    weighted MEAN over clips). The closest source always gets w=1, so the mean
    normalisation cancels any global scale — this path can DOWN-WEIGHT far
    sources RELATIVE to close ones but cannot make the whole alignment vanish.
    Kept verbatim so the existing DG-XL results reproduce bit-for-bit.

  * ``"sigmoid"`` (continuous λ(d), reviewer ask):
        λ_i = λ_max · σ((d0 - d_i) / tau)
    ABSOLUTE weight in [0, λ_max]. Applied as a SCALE on the (unweighted) mean
    alignment loss, per source. As d_i grows past the threshold d0, λ_i -> 0, so
    a pool of uniformly-distant sources (e.g. every source far from AUTSL)
    provably reduces to plain masked-SSL pooling — fixing the DG-XL 66.7 <
    pool 70.2 contradiction on AUTSL.

  * ``"exp"`` (continuous λ(d), exponential variant):
        λ_i = λ_max · exp(-γ · d_i)
    ABSOLUTE, monotone-decreasing, no threshold. Same scale semantics as
    ``"sigmoid"``.

The distinction that matters: the legacy ``hardgate`` is a *relative* re-weight
(mean-normalised, so global magnitude is lost); the continuous modes are
*absolute* scales (a far pool shrinks the alignment toward zero). ``build_gate``
returns both the per-language weight vector and a flag telling the pretrainer
which application semantics to use.
"""

from __future__ import annotations


import torch


# Application semantics for the returned per-language weight vector.
#   "reweight" -> weighted MEAN over clips (weight.sum() normaliser); relative.
#   "scale"    -> per-clip multiplier on the MEAN loss; absolute, far -> 0.
_REWEIGHT_MODES = {"hardgate"}
_SCALE_MODES = {"sigmoid", "exp"}
GATE_MODES = _REWEIGHT_MODES | _SCALE_MODES


def gate_application(mode):
    """Return "reweight" or "scale" for a gate mode string."""
    m = (mode or "hardgate").lower()
    if m in _REWEIGHT_MODES:
        return "reweight"
    if m in _SCALE_MODES:
        return "scale"
    raise ValueError(f"unknown gate_mode {mode!r}; expected one of {sorted(GATE_MODES)}")


def build_gate_weights(
    source_distances,
    mode="hardgate",
    lambda_max=1.0,
    tau=0.1,
    gamma=1.0,
    d0=None,
):
    """Per-language alignment weight vector from source distances.

    Args:
        source_distances: list[float] length = num_languages. d_i = MMD(source
            language i -> target). The target language's own entry is 0.
        mode: "hardgate" | "sigmoid" | "exp".
        lambda_max: ceiling for the continuous modes (weight lands in
            [0, lambda_max]). Ignored by "hardgate" (which caps at 1).
        tau: temperature. "hardgate": exp(-(d-min)/tau). "sigmoid": slope of
            σ((d0-d)/tau).
        gamma: decay rate for "exp": exp(-gamma*d).
        d0: sigmoid threshold distance (weight = λ_max/2 at d==d0). Defaults to
            the MEDIAN source distance so roughly half the pool is "close".

    Returns:
        (weights, application) where ``weights`` is a 1-D float tensor of length
        num_languages and ``application`` is "reweight" or "scale".
    """
    d = torch.tensor([float(x) for x in source_distances], dtype=torch.float32)
    m = (mode or "hardgate").lower()
    if m == "hardgate":
        w = torch.exp(-(d - d.min()) / float(tau))
    elif m == "sigmoid":
        if d0 is None:
            d0 = float(d.median())
        w = float(lambda_max) * torch.sigmoid((float(d0) - d) / float(tau))
    elif m == "exp":
        w = float(lambda_max) * torch.exp(-float(gamma) * d)
    else:
        raise ValueError(f"unknown gate_mode {mode!r}; expected one of {sorted(GATE_MODES)}")
    return w, gate_application(m)


def coral_loss(source_feat, target_feat, eps=1e-5):
    """CORAL alignment (reviewer §29): match source/target feature covariance.

    Squared Frobenius distance between the (batch) covariance matrices of the
    source-language and target-language pooled clip features, normalised by
    ``4 * d^2`` per the CORAL paper (Sun & Saenko, 2016). Differentiable,
    label-free, and an ALTERNATIVE alignment mechanism to the DANN/GRL term for
    the mechanism-baseline table.

    Args:
        source_feat: (Ns, D) pooled clip features from source-language clips.
        target_feat: (Nt, D) pooled clip features from target-language clips.
    Returns:
        scalar tensor (0-dim). Zero if either side has < 2 clips.
    """
    if source_feat.dim() != 2 or target_feat.dim() != 2:
        raise ValueError("coral_loss expects 2-D (N, D) feature tensors")
    ns, d = source_feat.shape
    nt = target_feat.shape[0]
    if ns < 2 or nt < 2:
        return source_feat.new_zeros(())

    def _cov(f):
        fc = f - f.mean(dim=0, keepdim=True)
        return (fc.t() @ fc) / (f.size(0) - 1)

    cs = _cov(source_feat)
    ct = _cov(target_feat)
    diff = cs - ct
    return (diff * diff).sum() / (4.0 * float(d) * float(d) + eps)
