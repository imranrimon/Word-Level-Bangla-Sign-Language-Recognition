"""Extract DINOv2 hand/face-crop features for BdSLW60, by signer-disjoint split.

For each clip we:
  1. decode every frame,
  2. run MediaPipe Holistic to get hand + face landmarks,
  3. crop a square box around each set of landmarks (30% pad, squared),
  4. resize to 224x224 and push through a pretrained DINOv2 encoder,
  5. stack (left_hand, right_hand, face) features per frame.

Output per split mirrors the pose NPY convention used by feeders.feeder.Feeder,
so the existing training harness can consume it without changes:

    data/bdsl_si_dino/<split>_data.npy    (N, D, T_max, V=3, M=1) float32
    data/bdsl_si_dino/<split>_label.pkl   (sample_names, labels)
    data/bdsl_si_dino/classes.json        mirrors data/bdsl_si/classes.json

For V=3, index 0=left hand, 1=right hand, 2=face. Frames with no detected
region get a zero vector (model should learn to ignore with attention).

Usage:

    python preprocessing/extract_dinov2_features.py \
        --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" \
        --output-dir data/bdsl_si_dino \
        --cache-dir data/bdsl_dino_cache \
        --splits train val test \
        --device cuda \
        --model vit_small_patch14_dinov2.lvd142m \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Make project root importable whether we are launched from root or elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.bdsl_signer_split import (  # noqa: E402
    ALL_SPLITS,
    LABELED_SPLITS,
    scan_dataset,
    summarize_splits,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REGION_NAMES = ("left_hand", "right_hand", "face")
NUM_REGIONS = len(REGION_NAMES)
INPUT_SIZE = 224
PAD_RATIO = 0.30


def _load_dinov2(model_name, device):
    """Return a DINOv2 encoder that emits CLS-token features of shape (B, D)."""
    import timm

    # DINOv2 lvd142m checkpoints are published at a 518x518 native resolution,
    # and current timm strictly asserts that input size. We feed 224x224 crops,
    # so reconfigure the patch grid to 224 (timm interpolates the pretrained
    # positional embeddings) and enable dynamic sizing for robustness.
    try:
        model = timm.create_model(
            model_name, pretrained=True, num_classes=0,
            img_size=INPUT_SIZE, dynamic_img_size=True,
        )
    except TypeError:
        # Older timm without dynamic_img_size support: fall back to img_size only.
        model = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=INPUT_SIZE,
        )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    feat_dim = int(getattr(model, "num_features", 384))
    return model, feat_dim


def _crop_square_region(frame_bgr, xs, ys, pad_ratio=PAD_RATIO, target=INPUT_SIZE):
    """Return a (target,target,3) RGB uint8 array, or None if no valid crop."""
    h, w = frame_bgr.shape[:2]
    if len(xs) == 0:
        return None
    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * (1.0 + pad_ratio)
    side = max(side, 0.08)   # avoid tiny crops; ~8% of frame min
    px1 = int(max(0, (cx - side / 2) * w))
    py1 = int(max(0, (cy - side / 2) * h))
    px2 = int(min(w, (cx + side / 2) * w))
    py2 = int(min(h, (cy + side / 2) * h))
    if px2 <= px1 or py2 <= py1:
        return None
    crop = frame_bgr[py1:py2, px1:px2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (target, target), interpolation=cv2.INTER_AREA)


def _landmarks_xy(landmark_list):
    if landmark_list is None:
        return [], []
    xs = [float(lm.x) for lm in landmark_list.landmark]
    ys = [float(lm.y) for lm in landmark_list.landmark]
    return xs, ys


def _crops_for_frame(frame_bgr, holistic_result):
    """Return (L, R, F) crops, each a (224,224,3) uint8 array or None."""
    xL, yL = _landmarks_xy(holistic_result.left_hand_landmarks)
    xR, yR = _landmarks_xy(holistic_result.right_hand_landmarks)
    xF, yF = _landmarks_xy(holistic_result.face_landmarks)
    return (
        _crop_square_region(frame_bgr, xL, yL) if xL else None,
        _crop_square_region(frame_bgr, xR, yR) if xR else None,
        _crop_square_region(frame_bgr, xF, yF) if xF else None,
    )


def _dino_forward_batch(crops_uint8, dino, device, batch_size, feat_dim):
    """Run DINOv2 over a flat list of RGB crops (may contain None). Output (N, D)."""
    n = len(crops_uint8)
    out = np.zeros((n, feat_dim), dtype=np.float32)
    valid_idx = [i for i, c in enumerate(crops_uint8) if c is not None]
    if not valid_idx:
        return out
    for start in range(0, len(valid_idx), batch_size):
        idxs = valid_idx[start:start + batch_size]
        batch = np.stack([crops_uint8[i] for i in idxs]).astype(np.float32) / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        batch_t = torch.from_numpy(batch.transpose(0, 3, 1, 2)).contiguous().to(device)
        with torch.no_grad():
            feat = dino(batch_t)
        feat = feat.detach().cpu().numpy().astype(np.float32)
        for j, target_i in enumerate(idxs):
            out[target_i] = feat[j]
    return out


def _process_video(file_path, holistic, dino, device, batch_size, feat_dim):
    """Return (D, T, V=3, M=1) float32, or None if the video could not be read."""
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None

    region_crops = [[], [], []]   # one list per region, length T, None-or-crop
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        L, R, F = _crops_for_frame(frame, res)
        region_crops[0].append(L)
        region_crops[1].append(R)
        region_crops[2].append(F)

    # Pack all regions into one big list, track where they came from, one DINOv2 pass.
    flat, back_idx = [], []
    for region_i, per_frame in enumerate(region_crops):
        for frame_i, crop in enumerate(per_frame):
            flat.append(crop)
            back_idx.append((region_i, frame_i))
    flat_feat = _dino_forward_batch(flat, dino, device, batch_size, feat_dim)

    T = len(frames)
    out = np.zeros((feat_dim, T, NUM_REGIONS, 1), dtype=np.float32)
    for (region_i, frame_i), f in zip(back_idx, flat_feat):
        out[:, frame_i, region_i, 0] = f
    return out


def _process_or_load_cached(args_tuple, holistic, dino, device, batch_size, feat_dim, cache_dir):
    file_path, label, label_name = args_tuple
    cache_path = None
    if cache_dir:
        rel = os.path.splitext(os.path.basename(file_path))[0] + ".npz"
        cache_path = os.path.join(cache_dir, label_name, rel)
        if os.path.exists(cache_path):
            with np.load(cache_path) as handle:
                return (handle["features"], label, label_name, file_path)

    features = _process_video(
        file_path, holistic, dino, device, batch_size, feat_dim
    )
    if features is None:
        return None

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, features=features)
    return (features, label, label_name, file_path)


def _save_split(results, split_name, output_dir, max_frames, feat_dim):
    if not results:
        print("[{}] no clips, skipping".format(split_name))
        return
    n = len(results)
    large = np.zeros((n, feat_dim, max_frames, NUM_REGIONS, 1), dtype=np.float32)
    sample_names = []
    labels = []
    for i, (feat, label, _cls, path) in enumerate(results):
        t = feat.shape[1]
        if t > max_frames:
            large[i, :, :max_frames, :, :] = feat[:, :max_frames, :, :]
        else:
            large[i, :, :t, :, :] = feat
        labels.append(label)
        sample_names.append(os.path.basename(path))

    data_path = os.path.join(output_dir, "{}_data.npy".format(split_name))
    label_path = os.path.join(output_dir, "{}_label.pkl".format(split_name))
    np.save(data_path, large)
    with open(label_path, "wb") as f:
        pickle.dump((sample_names, labels), f)
    print("[{}] wrote {} shape={}".format(split_name, data_path, large.shape))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=list(LABELED_SPLITS),
        choices=list(ALL_SPLITS),
    )
    ap.add_argument(
        "--model",
        default="vit_small_patch14_dinov2.lvd142m",
        help="timm model name for DINOv2",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    # Summary + plan up front.
    summary = summarize_splits(args.dataset_root)
    print("{:<10} {:>6} {:>8} {:>8}".format("split", "clips", "classes", "signers"))
    for name in ALL_SPLITS:
        s = summary.get(name, {"clips": 0, "classes": set(), "signers": set()})
        print("{:<10} {:>6} {:>8} {:>8}".format(
            name, s["clips"], len(s["classes"]), len(s["signers"])
        ))
    print("model:  {}".format(args.model))
    print("device: {}".format(args.device))
    print("writing to: {}".format(args.output_dir))
    if args.cache_dir:
        print("cache dir:  {}".format(args.cache_dir))
    if args.dry_run:
        print("--dry-run set, skipping extraction")
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dino, feat_dim = _load_dinov2(args.model, device)
    print("DINOv2 feature dim: {}".format(feat_dim))

    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Group and process.
    per_split_tasks = defaultdict(list)
    class_set = set()
    for filepath, class_name, _signer, split_name in scan_dataset(args.dataset_root):
        per_split_tasks[split_name].append((filepath, class_name))
        class_set.add(class_name)
    classes = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx,
                   "feat_dim": feat_dim, "regions": list(REGION_NAMES)},
                  f, indent=2)

    for name in args.splits:
        tasks = per_split_tasks.get(name, [])
        if not tasks:
            print("[{}] no parseable clips".format(name))
            continue
        results = []
        for fp, cls in tqdm(tasks, desc="[{}] dino".format(name)):
            r = _process_or_load_cached(
                (fp, class_to_idx[cls], cls), holistic, dino, device,
                args.batch_size, feat_dim, args.cache_dir,
            )
            if r is not None:
                results.append(r)
        _save_split(results, name, args.output_dir, args.max_frames, feat_dim)

    holistic.close()
    print("done.")


if __name__ == "__main__":
    main()
