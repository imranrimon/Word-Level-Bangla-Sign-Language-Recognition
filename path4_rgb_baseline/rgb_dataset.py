"""RGB video dataset for BdSLW60 under the canonical signer-independent split.

Reuses `preprocessing.bdsl_signer_split.scan_dataset` for filename parsing /
split assignment and `data/bdsl_si/classes.json` for the label order, so RGB
labels are guaranteed identical to the pose NPY bundles.

Sampling: TSN-style segment sampling (train) / uniform centers (eval) over
the decoded frame count, sequential cv2 decode keeping only wanted indices
(per-frame seeking is slow and unreliable on some codecs).

Normalization modes:
  * "kinetics" — frames to [0,1] then Kinetics-400 mean/std (torchvision
    video models: s3d, r2plus1d_18, mvit_v2_s).
  * "pm1"      — frames scaled to [-1, 1] (vendored piergiaj I3D convention).
"""

from __future__ import annotations

import json
import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.bdsl_signer_split import scan_dataset  # noqa: E402

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def load_class_map(classes_json):
    with open(classes_json) as f:
        classes = json.load(f)["classes"]
    return {name: i for i, name in enumerate(classes)}, classes


def segment_indices(num_available, num_frames, train):
    """TSN segment sampling: one index per equal segment (random in train)."""
    if num_available <= 0:
        return [0] * num_frames
    edges = np.linspace(0, num_available, num_frames + 1)
    idx = []
    for i in range(num_frames):
        lo, hi = int(edges[i]), max(int(edges[i]), int(edges[i + 1]) - 1)
        idx.append(random.randint(lo, hi) if (train and hi > lo) else (lo + hi) // 2)
    return [min(i, num_available - 1) for i in idx]


class BdSLW60RGBDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        classes_json,
        num_frames=32,
        size=224,
        resize_short=256,
        train=False,
        normalize="kinetics",
        max_clips=None,
    ):
        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.size = size
        self.resize_short = resize_short
        self.train = train
        self.normalize = normalize

        class_to_idx, self.classes = load_class_map(classes_json)
        self.items = []  # (filepath, label)
        for filepath, class_name, _signer, split_name in scan_dataset(root):
            if split_name != split:
                continue
            if class_name not in class_to_idx:
                continue
            self.items.append((filepath, class_to_idx[class_name]))
        if max_clips is not None:
            self.items = self.items[: int(max_clips)]
        if not self.items:
            raise RuntimeError(f"No clips found for split={split!r} under {root!r}")
        self.sample_name = [os.path.basename(p) for p, _ in self.items]
        self.label = [l for _, l in self.items]

    def __len__(self):
        return len(self.items)

    def _decode(self, path):
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            n = 10 ** 6  # unknown count: decode to exhaustion
        wanted = set(segment_indices(n, self.num_frames, self.train))
        frames = {}
        i = 0
        while i <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                frames[i] = self._prep(frame)
            i += 1
        cap.release()
        if not frames:
            # A single unreadable clip must not kill a multi-hour run (test
            # videos are decoded for the first time at the final eval).
            # Return a black clip and log loudly; the sample scores ~random.
            print(f"[rgb_dataset] WARNING: no decodable frames in {path}; using black clip")
            side = self.resize_short
            black = np.zeros((side, side, 3), dtype=np.uint8)
            return np.stack([black] * self.num_frames, axis=0)
        # Rebuild the ordered clip; pad missing indices with the last frame.
        order = sorted(wanted)
        last = frames[max(frames)]
        clip = [frames.get(j, last) for j in order]
        while len(clip) < self.num_frames:  # duplicates from set() dedup
            clip.append(clip[-1])
        return np.stack(clip[: self.num_frames], axis=0)  # (T, H, W, C)

    def _prep(self, frame_bgr):
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = self.resize_short / min(h, w)
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
        return frame

    def _crop(self, clip):
        _, h, w, _ = clip.shape
        s = self.size
        if self.train:
            y = random.randint(0, max(0, h - s))
            x = random.randint(0, max(0, w - s))
        else:
            y, x = (h - s) // 2, (w - s) // 2
        return clip[:, y : y + s, x : x + s, :]

    def __getitem__(self, index):
        path, label = self.items[index]
        clip = self._decode(path)          # (T, H, W, C) uint8
        clip = self._crop(clip).astype(np.float32) / 255.0
        if self.normalize == "kinetics":
            clip = (clip - KINETICS_MEAN) / KINETICS_STD
        elif self.normalize == "pm1":
            clip = clip * 2.0 - 1.0
        else:
            raise ValueError(f"unknown normalize mode {self.normalize!r}")
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)
        return clip, label, index
