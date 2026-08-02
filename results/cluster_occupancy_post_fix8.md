# K-means cluster occupancy by source

Targets file: `data/pretrain_kmeans_targets.npz`  
Manifest: `data/ssl_pool_manifest.json`  
Feature mode: `pose_motion`, K = 64, total frames = 5844998

## Per-source totals and cluster ownership

| source | frames | frame-share | clusters >50% owned | expected-uniform | over/under |
|---|---:|---:|---:|---:|---:|
| bdslw102_a_pose_cache | 258834 | 4.4% | 1/64 | 2.8 | -1.8 |
| bdslw401_pose_cache_front | 4239366 | 72.5% | 52/64 | 46.4 | +5.6 |
| wlasl_pose_cache | 1346798 | 23.0% | 11/64 | 14.7 | -3.7 |

## Cluster-size imbalance

- Cluster-size entropy: **3.811 nats** (max possible at K=64: 4.159)
- Largest cluster: **7.05%** of frames
- Smallest non-empty cluster: **0.095%** of frames
- Largest / smallest ratio: **74.0x**

## Per-cluster dominance (sorted by frame count)

| cluster | frames | dominant source | dominance | entropy (nats) |
|---:|---:|---|---:|---:|
| 3 | 412335 | bdslw401_pose_cache_front | 97.1% | 0.145 |
| 39 | 374903 | bdslw401_pose_cache_front | 98.5% | 0.076 |
| 2 | 255476 | wlasl_pose_cache | 100.0% | -0.000 |
| 57 | 242494 | bdslw401_pose_cache_front | 80.5% | 0.561 |
| 4 | 229780 | wlasl_pose_cache | 100.0% | -0.000 |
| 25 | 223888 | bdslw401_pose_cache_front | 97.5% | 0.116 |
| 46 | 204210 | wlasl_pose_cache | 100.0% | -0.000 |
| 22 | 200154 | bdslw401_pose_cache_front | 98.8% | 0.065 |
| 56 | 183062 | wlasl_pose_cache | 100.0% | -0.000 |
| 50 | 161689 | bdslw401_pose_cache_front | 99.8% | 0.013 |
| 5 | 157346 | bdslw401_pose_cache_front | 97.4% | 0.121 |
| 52 | 140575 | wlasl_pose_cache | 100.0% | -0.000 |
| 26 | 139702 | bdslw401_pose_cache_front | 98.6% | 0.072 |
| 41 | 133945 | bdslw401_pose_cache_front | 95.7% | 0.177 |
| 31 | 129965 | wlasl_pose_cache | 100.0% | -0.000 |
| 40 | 124133 | bdslw401_pose_cache_front | 97.0% | 0.136 |
| 38 | 121280 | bdslw401_pose_cache_front | 97.8% | 0.105 |
| 63 | 120683 | bdslw401_pose_cache_front | 98.1% | 0.093 |
| 19 | 116439 | bdslw401_pose_cache_front | 98.5% | 0.077 |
| 13 | 115206 | bdslw401_pose_cache_front | 96.8% | 0.141 |
| 1 | 113643 | bdslw401_pose_cache_front | 97.0% | 0.134 |
| 37 | 94804 | bdslw401_pose_cache_front | 95.1% | 0.195 |
| 6 | 93828 | bdslw401_pose_cache_front | 96.7% | 0.146 |
| 60 | 92288 | bdslw401_pose_cache_front | 92.1% | 0.277 |
| 34 | 85825 | bdslw401_pose_cache_front | 94.1% | 0.225 |
| 24 | 81954 | bdslw401_pose_cache_front | 92.0% | 0.278 |
| 30 | 75514 | wlasl_pose_cache | 100.0% | -0.000 |
| 10 | 75185 | bdslw401_pose_cache_front | 97.9% | 0.100 |
| 23 | 74784 | bdslw401_pose_cache_front | 100.0% | 0.001 |
| 36 | 73251 | bdslw401_pose_cache_front | 89.0% | 0.346 |
| 47 | 69134 | bdslw401_pose_cache_front | 99.8% | 0.014 |
| 61 | 65410 | bdslw401_pose_cache_front | 93.6% | 0.238 |
| 43 | 65234 | bdslw401_pose_cache_front | 94.5% | 0.212 |
| 51 | 63868 | bdslw401_pose_cache_front | 99.8% | 0.014 |
| 29 | 60933 | bdslw401_pose_cache_front | 89.6% | 0.334 |
| 45 | 53412 | bdslw102_a_pose_cache | 81.3% | 0.572 |
| 44 | 52601 | bdslw401_pose_cache_front | 99.9% | 0.007 |
| 62 | 51686 | bdslw401_pose_cache_front | 82.4% | 0.466 |
| 11 | 49541 | wlasl_pose_cache | 100.0% | -0.000 |
| 18 | 48340 | bdslw401_pose_cache_front | 94.6% | 0.209 |
| 12 | 47556 | bdslw401_pose_cache_front | 91.4% | 0.294 |
| 9 | 45935 | bdslw401_pose_cache_front | 94.8% | 0.204 |
| 58 | 40331 | bdslw401_pose_cache_front | 98.9% | 0.061 |
| 28 | 38286 | bdslw401_pose_cache_front | 72.9% | 0.584 |
| 16 | 37336 | bdslw401_pose_cache_front | 89.0% | 0.347 |
| 15 | 37173 | bdslw401_pose_cache_front | 91.9% | 0.281 |
| 49 | 29676 | bdslw401_pose_cache_front | 94.5% | 0.213 |
| 20 | 27269 | wlasl_pose_cache | 100.0% | -0.000 |
| 8 | 25322 | bdslw401_pose_cache_front | 91.1% | 0.301 |
| 14 | 25055 | bdslw401_pose_cache_front | 89.1% | 0.343 |
| 35 | 24382 | bdslw401_pose_cache_front | 90.6% | 0.312 |
| 59 | 22891 | bdslw401_pose_cache_front | 94.7% | 0.207 |
| 7 | 22427 | wlasl_pose_cache | 100.0% | -0.000 |
| 17 | 21901 | wlasl_pose_cache | 62.3% | 0.663 |
| 33 | 21900 | bdslw401_pose_cache_front | 93.5% | 0.241 |
| 21 | 20489 | bdslw401_pose_cache_front | 94.6% | 0.211 |
| 54 | 20379 | bdslw401_pose_cache_front | 89.5% | 0.335 |
| 48 | 19385 | bdslw401_pose_cache_front | 89.0% | 0.347 |
| 55 | 19057 | bdslw401_pose_cache_front | 91.4% | 0.294 |
| 32 | 18814 | bdslw401_pose_cache_front | 89.9% | 0.326 |
| 0 | 17110 | bdslw401_pose_cache_front | 92.7% | 0.287 |
| 42 | 15169 | bdslw401_pose_cache_front | 87.4% | 0.379 |
| 27 | 13081 | bdslw401_pose_cache_front | 91.7% | 0.287 |
| 53 | 5574 | bdslw401_pose_cache_front | 92.0% | 0.279 |
