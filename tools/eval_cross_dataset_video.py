"""Cross-dataset video-SLR evaluation driver.

For each (source-trained model, target dataset) pair, load the source
checkpoint, filter the target's val/test bundle to the vocabulary-aligned
subset (source ∩ target via the curated alignment JSON), and report
target Top-1 / Top-5.

This is the analog of `path3_handshape_benchmark/eval_cross_dataset.py`
for video skeleton SLR (Paper 2). Differences:
  * Source and target use the SAME pose layout (3, T, 27, 1) so we don't
    re-fit a head — we *constrain* the source model's softmax to only
    the classes present in the alignment subset.
  * Target labels are remapped from source's class_to_idx via the
    alignment table.

Usage:

    python tools/eval_cross_dataset_video.py \\
        --checkpoint work_dir/bdsl_block_gcn_si_seed0/best.pt \\
        --source-config config/bdsl_block_gcn_si.yaml \\
        --target-data data/bdslw102a_si/test_data.npy \\
        --target-label data/bdslw102a_si/test_label.pkl \\
        --target-classes data/bdslw102a_si/classes.json \\
        --alignment data/bangla_vocab_alignment_curated.json \\
        --source-name bdslw60 --target-name bdslw102a \\
        --output results/cross_dataset_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _import_class(import_str):
    mod, cls = import_str.rsplit(".", 1)
    __import__(mod)
    return getattr(sys.modules[mod], cls)


def _load_source_model(checkpoint_path, source_config_path, device):
    """Build the source model from its training config + load best.pt."""
    with open(source_config_path) as f:
        cfg = yaml.safe_load(f)
    model_cls = _import_class(cfg["model"])
    model = model_cls(**(cfg.get("model_args") or {}))
    state = torch.load(checkpoint_path, map_location="cpu")
    # Tolerate the common 'module.<name>' prefix from DataParallel saves.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)} "
              f"keys (often fc.* is fine if num_class differs)")
    return model.to(device).eval(), cfg


def _aligned_label_map(alignment, source_name, target_name,
                      source_classes, target_classes):
    """For each target class that has an exact match in the alignment, return
    the source-class index it should be evaluated against.

    Returns (target_idx_to_source_idx, kept_target_idxs).
    """
    key = f"{source_name}__to__{target_name}"
    rkey = f"{target_name}__to__{source_name}"
    if key not in alignment["alignments"] and rkey not in alignment["alignments"]:
        raise ValueError(f"alignment missing for {source_name} <-> {target_name}")

    # Build a target_name -> source_name dict from whichever direction exists.
    # Prefer source->target (it tells us which source classes appear in target).
    pair = alignment["alignments"].get(key) or alignment["alignments"].get(rkey)
    exact = pair["exact"]  # {source_class: target_class} (if key was source->target)
    if pair["source"] == target_name:
        # We loaded the reverse direction; flip.
        exact = {tgt: src for src, tgt in exact.items()}

    src_class_to_idx = {c: i for i, c in enumerate(source_classes)}
    tgt_class_to_idx = {c: i for i, c in enumerate(target_classes)}

    target_idx_to_source_idx = {}
    kept_target_idxs = []
    for src_class, tgt_class in exact.items():
        if src_class not in src_class_to_idx:
            continue
        if tgt_class not in tgt_class_to_idx:
            continue
        ti = tgt_class_to_idx[tgt_class]
        si = src_class_to_idx[src_class]
        target_idx_to_source_idx[ti] = si
        kept_target_idxs.append(ti)
    return target_idx_to_source_idx, set(kept_target_idxs)


@torch.no_grad()
def _evaluate(model, data, labels, target_idx_to_source_idx,
              source_num_class, device, batch_size=32):
    """Run the source model over target clips. For each clip:
      * forward → source logits (size: source_num_class)
      * MASK logits for source-classes NOT in the alignment to -inf
      * argmax over the remaining → predicted source class
      * compare to the source-class that was mapped from the clip's target label
    """
    # Build mask: 1 for aligned-source classes, 0 for the rest.
    mask = torch.full((source_num_class,), float("-inf"), device=device)
    for si in target_idx_to_source_idx.values():
        mask[si] = 0.0

    # Pre-filter: only keep clips whose label has an alignment.
    keep_idx = [i for i, lbl in enumerate(labels) if int(lbl) in target_idx_to_source_idx]
    if not keep_idx:
        return {"n_aligned": 0, "top1": float("nan"), "top5": float("nan")}

    kept_data = data[keep_idx]
    kept_labels = np.asarray([target_idx_to_source_idx[int(labels[i])] for i in keep_idx])

    correct1 = correct5 = 0
    n = len(keep_idx)
    for start in range(0, n, batch_size):
        batch = torch.from_numpy(kept_data[start:start + batch_size]).float().to(device)
        logits = model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits + mask    # mask out non-aligned classes
        topk = logits.topk(5, dim=-1).indices.cpu().numpy()
        for i, t in enumerate(kept_labels[start:start + batch_size]):
            if topk[i, 0] == t:
                correct1 += 1
            if t in topk[i]:
                correct5 += 1

    return {
        "n_aligned": int(n),
        "n_target_total": int(len(labels)),
        "top1": correct1 / n,
        "top5": correct5 / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source-config", required=True,
                    help="config used to train the source model")
    ap.add_argument("--source-classes", default=None,
                    help="JSON with source dataset classes (auto-loaded from "
                         "source_config's <data_dir>/classes.json if omitted)")
    ap.add_argument("--target-data", required=True)
    ap.add_argument("--target-label", required=True)
    ap.add_argument("--target-classes", required=True)
    ap.add_argument("--alignment", required=True,
                    help="JSON from build_bangla_vocab_alignment.py (curated)")
    ap.add_argument("--source-name", required=True,
                    help="name as it appears in the alignment JSON")
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", default="results/cross_dataset_eval.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load source model + config.
    model, source_cfg = _load_source_model(args.checkpoint, args.source_config, device)
    source_num_class = int(source_cfg.get("model_args", {}).get("num_class", -1))
    if source_num_class < 0:
        raise SystemExit("source config has no model_args.num_class")

    # Source classes.
    if args.source_classes:
        with open(args.source_classes, encoding="utf-8") as f:
            sc = json.load(f)
        source_classes = sc["classes"] if isinstance(sc, dict) and "classes" in sc else sc
    else:
        # Try to find classes.json next to the source config's train data.
        data_path = (source_cfg.get("train_feeder_args") or {}).get("data_path")
        candidate = Path(data_path).parent / "classes.json" if data_path else None
        if candidate and candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                sc = json.load(f)
            source_classes = sc["classes"] if isinstance(sc, dict) and "classes" in sc else sc
        else:
            raise SystemExit("--source-classes not provided and could not auto-locate")

    # Target classes.
    with open(args.target_classes, encoding="utf-8") as f:
        tc = json.load(f)
    target_classes = tc["classes"] if isinstance(tc, dict) and "classes" in tc else tc

    # Alignment.
    with open(args.alignment, encoding="utf-8") as f:
        alignment = json.load(f)

    target_idx_to_source_idx, _ = _aligned_label_map(
        alignment, args.source_name, args.target_name,
        source_classes, target_classes,
    )
    print(f"alignment: {len(target_idx_to_source_idx)} target classes "
          f"mapped to source classes")

    # Load target data + labels.
    data = np.load(args.target_data, mmap_mode="r")
    with open(args.target_label, "rb") as f:
        _names, labels = pickle.load(f)

    result = _evaluate(model, data, labels, target_idx_to_source_idx,
                       source_num_class, device, batch_size=args.batch_size)
    result.update({
        "source": args.source_name,
        "target": args.target_name,
        "checkpoint": args.checkpoint,
        "source_classes_count": len(source_classes),
        "target_classes_count": len(target_classes),
        "n_aligned_classes": len(target_idx_to_source_idx),
    })

    print(f"\n=== Cross-dataset eval ===")
    print(f"  {args.source_name} -> {args.target_name}")
    print(f"  aligned classes: {result['n_aligned_classes']}")
    print(f"  aligned clips: {result['n_aligned']} / {result['n_target_total']}")
    print(f"  Top-1: {result['top1']*100:.2f}%")
    print(f"  Top-5: {result['top5']*100:.2f}%")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nappended row to {args.output}")


if __name__ == "__main__":
    main()
