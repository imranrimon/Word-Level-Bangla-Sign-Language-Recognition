# K-means cluster occupancy by source

Targets file: `data/pretrain_kmeans_targets_bdsl_asl.npz`  
Manifest: `data/ssl_pool_manifest_bdsl_asl.json`  
Feature mode: `pose_motion`, K = 64, total frames = 5844998

## Per-source totals and cluster ownership

| source | frames | frame-share | clusters >50% owned | expected-uniform | over/under |
|---|---:|---:|---:|---:|---:|
| bdslw102_a_pose_cache | 258834 | 4.4% | 4/64 | 2.8 | +1.2 |
| bdslw401_pose_cache_front | 4239366 | 72.5% | 46/64 | 46.4 | -0.4 |
| wlasl_pose_cache | 1346798 | 23.0% | 14/64 | 14.7 | -0.7 |

## Cluster-size imbalance

- Cluster-size entropy: **3.818 nats** (max possible at K=64: 4.159)
- Largest cluster: **6.24%** of frames
- Smallest non-empty cluster: **0.060%** of frames
- Largest / smallest ratio: **104.0x**

## Per-cluster dominance (sorted by frame count)

| cluster | frames | dominant source | dominance | entropy (nats) |
|---:|---:|---|---:|---:|
| 2 | 364580 | bdslw401_pose_cache_front | 98.3% | 0.084 |
| 36 | 304510 | bdslw401_pose_cache_front | 97.7% | 0.110 |
| 15 | 257831 | bdslw401_pose_cache_front | 97.8% | 0.108 |
| 25 | 242992 | bdslw401_pose_cache_front | 93.0% | 0.267 |
| 3 | 239431 | wlasl_pose_cache | 100.0% | -0.000 |
| 42 | 238239 | bdslw401_pose_cache_front | 97.0% | 0.149 |
| 52 | 203019 | bdslw401_pose_cache_front | 98.4% | 0.081 |
| 44 | 197008 | wlasl_pose_cache | 100.0% | -0.000 |
| 21 | 182937 | bdslw401_pose_cache_front | 98.8% | 0.066 |
| 29 | 170336 | bdslw401_pose_cache_front | 96.4% | 0.155 |
| 17 | 168020 | bdslw401_pose_cache_front | 95.9% | 0.171 |
| 63 | 161942 | wlasl_pose_cache | 100.0% | -0.000 |
| 30 | 153610 | bdslw401_pose_cache_front | 96.9% | 0.137 |
| 62 | 146395 | wlasl_pose_cache | 100.0% | -0.000 |
| 41 | 142002 | bdslw401_pose_cache_front | 97.3% | 0.123 |
| 10 | 127785 | bdslw401_pose_cache_front | 99.5% | 0.032 |
| 0 | 123151 | bdslw401_pose_cache_front | 94.9% | 0.200 |
| 49 | 117524 | bdslw401_pose_cache_front | 97.8% | 0.106 |
| 50 | 108292 | bdslw401_pose_cache_front | 99.5% | 0.030 |
| 53 | 104258 | wlasl_pose_cache | 100.0% | -0.000 |
| 1 | 102279 | wlasl_pose_cache | 100.0% | -0.000 |
| 18 | 100228 | bdslw401_pose_cache_front | 94.6% | 0.211 |
| 58 | 98888 | bdslw401_pose_cache_front | 92.8% | 0.260 |
| 14 | 96788 | wlasl_pose_cache | 100.0% | -0.000 |
| 28 | 96621 | wlasl_pose_cache | 100.0% | -0.000 |
| 4 | 94946 | bdslw401_pose_cache_front | 95.8% | 0.174 |
| 57 | 86598 | bdslw401_pose_cache_front | 92.2% | 0.273 |
| 61 | 84426 | bdslw401_pose_cache_front | 96.4% | 0.155 |
| 43 | 82916 | bdslw401_pose_cache_front | 100.0% | 0.002 |
| 8 | 74827 | bdslw401_pose_cache_front | 89.4% | 0.339 |
| 60 | 73860 | wlasl_pose_cache | 100.0% | -0.000 |
| 45 | 69095 | bdslw401_pose_cache_front | 88.5% | 0.357 |
| 56 | 64045 | bdslw401_pose_cache_front | 99.5% | 0.034 |
| 46 | 63491 | bdslw401_pose_cache_front | 99.9% | 0.007 |
| 48 | 62216 | bdslw102_a_pose_cache | 70.7% | 0.800 |
| 20 | 56587 | bdslw401_pose_cache_front | 99.9% | 0.010 |
| 47 | 53694 | bdslw401_pose_cache_front | 91.4% | 0.292 |
| 5 | 52534 | bdslw401_pose_cache_front | 98.7% | 0.068 |
| 27 | 52476 | bdslw401_pose_cache_front | 91.9% | 0.282 |
| 16 | 52412 | bdslw401_pose_cache_front | 91.3% | 0.296 |
| 9 | 49541 | wlasl_pose_cache | 100.0% | -0.000 |
| 24 | 46975 | bdslw401_pose_cache_front | 91.5% | 0.291 |
| 51 | 41977 | bdslw401_pose_cache_front | 73.7% | 0.576 |
| 13 | 36436 | bdslw401_pose_cache_front | 91.4% | 0.294 |
| 12 | 33094 | bdslw401_pose_cache_front | 90.3% | 0.318 |
| 33 | 31940 | bdslw102_a_pose_cache | 96.5% | 0.151 |
| 55 | 27520 | bdslw401_pose_cache_front | 96.2% | 0.161 |
| 34 | 27435 | bdslw401_pose_cache_front | 95.3% | 0.190 |
| 26 | 25075 | bdslw401_pose_cache_front | 94.4% | 0.215 |
| 39 | 23951 | bdslw401_pose_cache_front | 90.5% | 0.314 |
| 19 | 21975 | bdslw401_pose_cache_front | 92.7% | 0.262 |
| 35 | 21642 | bdslw401_pose_cache_front | 89.6% | 0.334 |
| 11 | 21545 | bdslw401_pose_cache_front | 93.4% | 0.243 |
| 23 | 20615 | bdslw401_pose_cache_front | 94.2% | 0.223 |
| 59 | 19171 | wlasl_pose_cache | 100.0% | -0.000 |
| 7 | 18506 | bdslw401_pose_cache_front | 94.6% | 0.211 |
| 31 | 18430 | wlasl_pose_cache | 67.5% | 0.631 |
| 38 | 18407 | bdslw401_pose_cache_front | 88.8% | 0.352 |
| 37 | 17055 | bdslw401_pose_cache_front | 93.0% | 0.280 |
| 32 | 15784 | wlasl_pose_cache | 100.0% | -0.000 |
| 6 | 14741 | wlasl_pose_cache | 100.0% | -0.000 |
| 54 | 13032 | bdslw401_pose_cache_front | 91.7% | 0.285 |
| 40 | 3856 | bdslw102_a_pose_cache | 56.9% | 0.686 |
| 22 | 3506 | bdslw102_a_pose_cache | 65.6% | 0.646 |
