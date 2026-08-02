"""Convert WLASL MediaPipe-encoded landmarks (V3 format from
`abd0kamel/mutemotion-output` on Kaggle) into per-clip .npz pose cache files
matching the data/bdsl_cache layout (3, T, 27, 1) float32.

Our 27 keypoints (matches preprocessing/preprocess_bdsl.py and graph/sign_27.py):

  0-6   body:   nose, R-shoulder, L-shoulder, R-elbow, L-elbow, R-wrist, L-wrist
                (MediaPipe pose indices [0, 12, 11, 14, 13, 16, 15])
  7-16  right hand: wrist, thumb-tip, index-MCP, index-tip, mid-MCP, mid-tip,
                    ring-MCP, ring-tip, pinky-MCP, pinky-tip
                    (MediaPipe hand local indices [0, 4, 5, 8, 9, 12, 13, 16, 17, 20])
  17-26 left hand:  same hand-local indices as 7-16

V3 layout (verified by inspection — `(T, 553, 3)` float64):
  indices 0..32    -> MediaPipe pose (33)
  indices 33..500  -> face mesh (468, dropped)
  indices 501..521 -> left hand (21)
  indices 522..542 -> right hand (21)
  indices 543..552 -> extras (dropped)

Channel 3 in our format is binary confidence (1 if MediaPipe detected the
keypoint, 0 otherwise). V3 stores raw z-depth in channel 3, whose statistics
are very different from BdSL's visibility scores; mixing them in the same SSL
pool would create a dataset-source feature the model could shortcut on.

Usage:
    python preprocessing/convert_wlasl_landmarks_to_pose_cache.py \\
        --landmarks data/wlasl_mediapipe/landmarks_V3.npz \\
        --output-dir data/wlasl_pose_cache
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm


# Our 27 keypoint layout, mapped to V3's 553-keypoint absolute indexing.
_BODY_V3 = [0, 12, 11, 14, 13, 16, 15]
_HAND_LOCAL = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
_RIGHT_HAND_V3 = [522 + i for i in _HAND_LOCAL]
_LEFT_HAND_V3 = [501 + i for i in _HAND_LOCAL]
V3_TO_OUR27 = _BODY_V3 + _RIGHT_HAND_V3 + _LEFT_HAND_V3
assert len(V3_TO_OUR27) == 27


def convert_clip(v3_arr):
    """v3_arr: (T, K, 3) where K >= 553 -> (3, T, 27, 1) float32.

    Channel 3 is binarized to 1 where any of (x,y,z) is nonzero, 0 otherwise.
    """
    sel = v3_arr[:, V3_TO_OUR27, :]                 # (T, 27, 3)
    conf = (np.abs(sel).sum(axis=-1) > 0).astype(np.float32)   # (T, 27)
    T = sel.shape[0]
    out = np.zeros((3, T, 27, 1), dtype=np.float32)
    out[0, :, :, 0] = sel[:, :, 0]
    out[1, :, :, 0] = sel[:, :, 1]
    out[2, :, :, 0] = conf
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", default="data/wlasl_mediapipe/landmarks_V3.npz")
    ap.add_argument("--output-dir", default="data/wlasl_pose_cache")
    ap.add_argument(
        "--max-clips", type=int, default=None,
        help="smoke-test: process only the first N clips",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="skip clips whose output .npz already exists",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"opening {args.landmarks}...")
    with np.load(args.landmarks, allow_pickle=True) as bundle:
        video_ids = list(bundle.keys())
        if args.max_clips:
            video_ids = video_ids[: args.max_clips]
        print(f"converting {len(video_ids)} clips -> {args.output_dir}")

        n_written = 0
        n_skipped = 0
        n_malformed = 0
        for vid in tqdm(video_ids):
            out_path = os.path.join(args.output_dir, f"{vid}.npz")
            if args.skip_existing and os.path.exists(out_path):
                n_skipped += 1
                continue
            v3_arr = bundle[vid]
            if v3_arr.ndim != 3 or v3_arr.shape[1] < 553:
                print(f"[skip] {vid}: unexpected shape {v3_arr.shape}")
                n_malformed += 1
                continue
            out = convert_clip(v3_arr)
            np.savez_compressed(out_path, data=out)
            n_written += 1

    print(f"written: {n_written}, skipped existing: {n_skipped}, "
          f"malformed: {n_malformed}, total examined: {len(video_ids)}")


if __name__ == "__main__":
    main()
