"""Data feeder for SHuBERT-style masked pretraining.

Reads pose-cache `.npz` files listed in an SSL pool manifest, pairs them with
precomputed k-means cluster IDs (from compute_pretrain_targets.py), and yields
(pose_tensor, target_codes) pairs sized for `model.shubert_pretrain.ShubertPretrainer`.

Variable-length clips are cropped or zero-padded to `window_size` to match the
frame count expected by the backbone.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset


# Source-dir -> source-LANGUAGE id. The SSL pool mixes American (WLASL) and
# Bangla (BdSLW401 / BdSLW102_A / BdSLW60) pose. The cross-lingual mechanism
# needs the language of each clip; everything that isn't WLASL/ASL is Bangla.
NUM_LANGUAGES = 2
LANG_BDSL = 0
LANG_ASL = 1


def language_of(source):
    """Map an SSL-pool source path to a source-language id.

    ASL sources are WLASL (and ASL-Citizen, if ever added); everything else in
    the pool is Bangla (BdSLW401 / BdSLW102_A / BdSLW60 / SingleTrial).
    """
    s = (source or "").lower()
    if "wlasl" in s or "asl_citizen" in s:
        return LANG_ASL
    return LANG_BDSL


class PretrainFeeder(Dataset):
    def __init__(self, manifest_path, targets_path, window_size=120,
                 num_channels=3, num_joints=27, num_person=1, random_crop=True,
                 max_clips=None, return_language=False):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.clips = manifest["clips"]
        self.window_size = int(window_size)
        self.num_channels = int(num_channels)
        self.num_joints = int(num_joints)
        self.num_person = int(num_person)
        self.random_crop = bool(random_crop)
        self.return_language = bool(return_language)

        # Load targets
        loaded = np.load(targets_path, allow_pickle=True)
        keys = loaded["keys"]
        self._target_by_key = {}
        for i, k in enumerate(keys):
            self._target_by_key[str(k)] = loaded[f"t_{i}"]

        # Filter clips to those with targets available.
        self.clips = [c for c in self.clips if c["path"] in self._target_by_key]

        # Smoke-test convenience: cap at a small number of clips.
        if max_clips is not None and max_clips > 0:
            self.clips = self.clips[: int(max_clips)]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        entry = self.clips[idx]
        path = entry["path"]
        with np.load(path) as h:
            data = h["data"]   # (C, T, V, M)
        codes = self._target_by_key[path]  # (T,) int

        data, codes = self._crop_or_pad(data, codes)
        x = torch.from_numpy(data).float()
        t = torch.from_numpy(codes).long()
        if self.return_language:
            lang = language_of(entry.get("source", path))
            return x, t, torch.tensor(lang, dtype=torch.long)
        return x, t

    def _crop_or_pad(self, data, codes):
        # Pose cache may have trailing zero-padded frames (MediaPipe failure or
        # post-clip silence). compute_pretrain_targets.py truncates codes to
        # the last non-zero pose frame, so len(codes) is typically <= T_data.
        # Trim both to the smaller length so they stay row-aligned, then crop
        # or pad to window_size W.
        C, T_data, V, M = data.shape
        T_codes = len(codes)
        T = min(T_data, T_codes)
        if T == 0:
            T = max(T_data, T_codes, 1)   # degenerate; pad both to non-empty
            data = np.zeros((C, T, V, M), dtype=data.dtype)
            codes = np.zeros((T,), dtype=codes.dtype if T_codes else np.int32)
        else:
            data = data[:, :T]
            codes = codes[:T]

        W = self.window_size
        if T == W:
            return data, codes
        if T > W:
            start = int(np.random.randint(0, T - W + 1)) if self.random_crop else 0
            return data[:, start:start + W], codes[start:start + W]
        # T < W: pad tail with zeros
        padded = np.zeros((C, W, V, M), dtype=data.dtype)
        padded[:, :T] = data
        codes_padded = np.zeros((W,), dtype=codes.dtype)
        codes_padded[:T] = codes
        return padded, codes_padded
