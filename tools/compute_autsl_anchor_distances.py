"""P1 #5 — source->AUTSL pose-motion MMD, to anchor the DG-XL gate to AUTSL.

Reuses the descriptor/MMD from tools/compute_lang_distance.py. Emits source_distances
indexed by pretrain_feeder.language_of ids: [0=BdSL, 1=WLASL/ASL, 2=LSA64, 3=AUTSL, 4=KArSL].
AUTSL (the target) is 0 by definition; others are MMD(source_language -> AUTSL).
"""
import numpy as np
from tools.compute_lang_distance import sample_npz_dir, sample_npy, mmd_rbf

N = 400
LANG_CACHE = {                       # language_of id -> pose-cache dir (npz)
    0: "data/bdslw401_pose_cache_front",
    1: "data/wlasl_pose_cache",
    2: "data/lsa64_cache",
    4: "data/karsl_cache",
}
AUTSL_NPY = "data/autsl_si/train_data.npy"


def main():
    rng = np.random.default_rng(0)
    tgt = np.array(sample_npy(AUTSL_NPY, N, rng))        # AUTSL target descriptors
    print(f"AUTSL target sample: {len(tgt)} clips (desc dim {tgt.shape[1]})")
    dist = {3: 0.0}                                       # AUTSL is the anchor
    for lid, path in LANG_CACHE.items():
        src = np.array(sample_npz_dir(path, N, rng))
        d = float(mmd_rbf(src, tgt))
        dist[lid] = d
        print(f"  lang {lid} ({path.split('/')[-1]:28s}) MMD->AUTSL = {d:.4f}")
    ordered = [round(dist[i], 4) for i in range(5)]
    print("\nsource_distances (AUTSL-anchored, id order 0..4):")
    print(f"  {ordered}")
    print("  -> paste into config/bdsl_shubert_pretrain_dgxl_autsl.yaml : pretrain.source_distances")


if __name__ == "__main__":
    main()
