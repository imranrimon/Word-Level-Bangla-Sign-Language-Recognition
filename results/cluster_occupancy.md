# K-means cluster occupancy by source

Targets file: `data/pretrain_kmeans_targets.npz`  
Manifest: `data/ssl_pool_manifest.json`  
Feature mode: `pose_motion`, K = 64, total frames = 6165235

## Per-source totals and cluster ownership

| source | frames | frame-share | clusters >50% owned | expected-uniform | over/under |
|---|---:|---:|---:|---:|---:|
| bdsl_cache | 320237 | 5.2% | 0/64 | 3.3 | -3.3 |
| bdslw102_a_pose_cache | 258834 | 4.2% | 0/64 | 2.7 | -2.7 |
| bdslw401_pose_cache_front | 4239366 | 68.8% | 53/64 | 44.0 | +9.0 |
| wlasl_pose_cache | 1346798 | 21.8% | 10/64 | 14.0 | -4.0 |

## Cluster-size imbalance

- Cluster-size entropy: **3.805 nats** (max possible at K=64: 4.159)
- Largest cluster: **7.27%** of frames
- Smallest non-empty cluster: **0.093%** of frames
- Largest / smallest ratio: **78.2x**

## Per-cluster dominance (sorted by frame count)

| cluster | frames | dominant source | dominance | entropy (nats) |
|---:|---:|---|---:|---:|
| 4 | 448100 | bdslw401_pose_cache_front | 96.7% | 0.168 |
| 22 | 376779 | bdslw401_pose_cache_front | 97.0% | 0.151 |
| 54 | 308506 | bdslw401_pose_cache_front | 92.1% | 0.351 |
| 41 | 262670 | wlasl_pose_cache | 100.0% | -0.000 |
| 57 | 261978 | wlasl_pose_cache | 100.0% | -0.000 |
| 39 | 208650 | wlasl_pose_cache | 100.0% | -0.000 |
| 2 | 197708 | wlasl_pose_cache | 100.0% | -0.000 |
| 25 | 196257 | bdslw401_pose_cache_front | 95.3% | 0.209 |
| 8 | 172501 | wlasl_pose_cache | 100.0% | -0.000 |
| 21 | 167106 | bdslw401_pose_cache_front | 90.9% | 0.356 |
| 24 | 160041 | bdslw401_pose_cache_front | 83.0% | 0.555 |
| 0 | 149797 | bdslw401_pose_cache_front | 83.9% | 0.510 |
| 29 | 149723 | bdslw401_pose_cache_front | 91.1% | 0.333 |
| 44 | 143955 | bdslw401_pose_cache_front | 44.5% | 1.039 |
| 32 | 141931 | bdslw401_pose_cache_front | 85.3% | 0.429 |
| 52 | 136665 | bdslw401_pose_cache_front | 88.4% | 0.378 |
| 63 | 136430 | bdslw401_pose_cache_front | 86.2% | 0.469 |
| 50 | 135314 | bdslw401_pose_cache_front | 81.7% | 0.590 |
| 30 | 130106 | bdslw401_pose_cache_front | 89.3% | 0.413 |
| 47 | 124036 | bdslw401_pose_cache_front | 82.7% | 0.567 |
| 27 | 117852 | bdslw401_pose_cache_front | 83.1% | 0.508 |
| 1 | 115075 | wlasl_pose_cache | 100.0% | -0.000 |
| 37 | 110109 | bdslw401_pose_cache_front | 87.7% | 0.444 |
| 9 | 94603 | bdslw401_pose_cache_front | 85.5% | 0.512 |
| 3 | 93000 | bdslw401_pose_cache_front | 85.9% | 0.480 |
| 16 | 92328 | bdslw401_pose_cache_front | 93.5% | 0.253 |
| 19 | 89766 | bdslw401_pose_cache_front | 87.8% | 0.379 |
| 33 | 80399 | bdslw401_pose_cache_front | 87.7% | 0.446 |
| 43 | 79629 | bdslw401_pose_cache_front | 87.7% | 0.422 |
| 36 | 77835 | bdslw401_pose_cache_front | 88.9% | 0.368 |
| 38 | 66895 | bdslw401_pose_cache_front | 88.0% | 0.447 |
| 5 | 65706 | bdslw401_pose_cache_front | 87.4% | 0.385 |
| 49 | 63339 | bdslw401_pose_cache_front | 86.3% | 0.401 |
| 6 | 59114 | bdslw401_pose_cache_front | 93.2% | 0.265 |
| 45 | 58068 | bdslw401_pose_cache_front | 88.3% | 0.437 |
| 14 | 56656 | bdslw401_pose_cache_front | 84.8% | 0.530 |
| 59 | 53600 | bdslw401_pose_cache_front | 82.7% | 0.572 |
| 17 | 49696 | wlasl_pose_cache | 100.0% | -0.000 |
| 10 | 48839 | bdslw401_pose_cache_front | 75.5% | 0.644 |
| 35 | 48239 | bdslw401_pose_cache_front | 82.6% | 0.581 |
| 53 | 47049 | bdslw401_pose_cache_front | 93.2% | 0.252 |
| 20 | 45298 | bdslw401_pose_cache_front | 85.4% | 0.517 |
| 7 | 38745 | bdslw401_pose_cache_front | 85.1% | 0.525 |
| 51 | 37461 | bdslw401_pose_cache_front | 89.2% | 0.401 |
| 40 | 35530 | bdslw401_pose_cache_front | 87.4% | 0.432 |
| 56 | 31263 | bdslw401_pose_cache_front | 83.6% | 0.555 |
| 60 | 29168 | bdslw401_pose_cache_front | 83.4% | 0.543 |
| 18 | 28397 | bdslw401_pose_cache_front | 87.8% | 0.450 |
| 15 | 27514 | bdslw401_pose_cache_front | 88.9% | 0.416 |
| 23 | 25239 | bdslw401_pose_cache_front | 87.9% | 0.433 |
| 11 | 25193 | wlasl_pose_cache | 100.0% | -0.000 |
| 28 | 24763 | bdslw401_pose_cache_front | 88.3% | 0.422 |
| 12 | 24655 | bdslw401_pose_cache_front | 86.9% | 0.469 |
| 42 | 24573 | bdslw401_pose_cache_front | 87.3% | 0.469 |
| 31 | 24348 | wlasl_pose_cache | 100.0% | -0.000 |
| 26 | 23612 | bdslw401_pose_cache_front | 87.3% | 0.468 |
| 62 | 23400 | bdslw401_pose_cache_front | 88.8% | 0.427 |
| 55 | 22879 | bdslw401_pose_cache_front | 86.8% | 0.481 |
| 13 | 21911 | wlasl_pose_cache | 62.3% | 0.667 |
| 48 | 21080 | bdslw401_pose_cache_front | 84.4% | 0.527 |
| 58 | 17826 | bdslw401_pose_cache_front | 89.2% | 0.438 |
| 61 | 16293 | bdslw401_pose_cache_front | 88.7% | 0.390 |
| 34 | 14308 | bdslw401_pose_cache_front | 88.0% | 0.443 |
| 46 | 5729 | bdslw401_pose_cache_front | 88.2% | 0.437 |
