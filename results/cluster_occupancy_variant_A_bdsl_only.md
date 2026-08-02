# K-means cluster occupancy by source

Targets file: `data/pretrain_kmeans_targets_bdsl_only.npz`  
Manifest: `data/ssl_pool_manifest_bdsl_only.json`  
Feature mode: `pose_motion`, K = 64, total frames = 4498200

## Per-source totals and cluster ownership

| source | frames | frame-share | clusters >50% owned | expected-uniform | over/under |
|---|---:|---:|---:|---:|---:|
| bdslw102_a_pose_cache | 258834 | 5.8% | 3/64 | 3.7 | -0.7 |
| bdslw401_pose_cache_front | 4239366 | 94.2% | 61/64 | 60.3 | +0.7 |

## Cluster-size imbalance

- Cluster-size entropy: **3.830 nats** (max possible at K=64: 4.159)
- Largest cluster: **8.13%** of frames
- Smallest non-empty cluster: **0.184%** of frames
- Largest / smallest ratio: **44.3x**

## Per-cluster dominance (sorted by frame count)

| cluster | frames | dominant source | dominance | entropy (nats) |
|---:|---:|---|---:|---:|
| 14 | 365820 | bdslw401_pose_cache_front | 97.7% | 0.111 |
| 27 | 350709 | bdslw401_pose_cache_front | 98.9% | 0.060 |
| 4 | 178206 | bdslw401_pose_cache_front | 78.0% | 0.527 |
| 0 | 166575 | bdslw401_pose_cache_front | 99.0% | 0.054 |
| 53 | 158484 | bdslw401_pose_cache_front | 95.6% | 0.181 |
| 37 | 155920 | bdslw401_pose_cache_front | 98.8% | 0.064 |
| 25 | 147451 | bdslw401_pose_cache_front | 97.2% | 0.126 |
| 35 | 143806 | bdslw401_pose_cache_front | 98.7% | 0.071 |
| 1 | 125039 | bdslw401_pose_cache_front | 99.0% | 0.055 |
| 57 | 121111 | bdslw401_pose_cache_front | 97.5% | 0.116 |
| 15 | 109438 | bdslw401_pose_cache_front | 96.3% | 0.158 |
| 61 | 106821 | bdslw401_pose_cache_front | 97.3% | 0.126 |
| 60 | 105683 | bdslw401_pose_cache_front | 96.2% | 0.162 |
| 45 | 104703 | bdslw401_pose_cache_front | 93.9% | 0.230 |
| 6 | 93848 | bdslw401_pose_cache_front | 97.4% | 0.122 |
| 34 | 86558 | bdslw401_pose_cache_front | 95.9% | 0.172 |
| 55 | 78989 | bdslw401_pose_cache_front | 99.8% | 0.012 |
| 42 | 76220 | bdslw401_pose_cache_front | 100.0% | 0.001 |
| 62 | 71142 | bdslw401_pose_cache_front | 90.3% | 0.318 |
| 48 | 69892 | bdslw401_pose_cache_front | 88.8% | 0.351 |
| 32 | 67417 | bdslw401_pose_cache_front | 94.3% | 0.219 |
| 30 | 64902 | bdslw401_pose_cache_front | 83.6% | 0.446 |
| 3 | 64018 | bdslw401_pose_cache_front | 100.0% | 0.001 |
| 11 | 62046 | bdslw401_pose_cache_front | 90.2% | 0.320 |
| 51 | 61359 | bdslw401_pose_cache_front | 96.8% | 0.143 |
| 46 | 60896 | bdslw401_pose_cache_front | 99.9% | 0.005 |
| 28 | 59136 | bdslw401_pose_cache_front | 98.5% | 0.080 |
| 2 | 59106 | bdslw401_pose_cache_front | 99.1% | 0.051 |
| 5 | 56863 | bdslw401_pose_cache_front | 90.5% | 0.314 |
| 31 | 56480 | bdslw401_pose_cache_front | 96.4% | 0.156 |
| 24 | 51566 | bdslw401_pose_cache_front | 91.2% | 0.298 |
| 20 | 51523 | bdslw401_pose_cache_front | 91.7% | 0.285 |
| 41 | 50922 | bdslw401_pose_cache_front | 99.7% | 0.022 |
| 21 | 50638 | bdslw401_pose_cache_front | 99.9% | 0.008 |
| 44 | 47486 | bdslw401_pose_cache_front | 98.9% | 0.061 |
| 13 | 47473 | bdslw401_pose_cache_front | 94.7% | 0.209 |
| 33 | 45663 | bdslw401_pose_cache_front | 97.9% | 0.103 |
| 9 | 44520 | bdslw401_pose_cache_front | 91.8% | 0.283 |
| 47 | 43835 | bdslw401_pose_cache_front | 90.0% | 0.326 |
| 29 | 41277 | bdslw401_pose_cache_front | 96.8% | 0.141 |
| 39 | 40430 | bdslw401_pose_cache_front | 92.0% | 0.279 |
| 17 | 40406 | bdslw401_pose_cache_front | 99.2% | 0.048 |
| 19 | 40115 | bdslw401_pose_cache_front | 98.3% | 0.085 |
| 40 | 39354 | bdslw401_pose_cache_front | 91.5% | 0.292 |
| 50 | 36005 | bdslw401_pose_cache_front | 90.9% | 0.305 |
| 59 | 34946 | bdslw401_pose_cache_front | 99.9% | 0.007 |
| 10 | 34003 | bdslw401_pose_cache_front | 90.3% | 0.319 |
| 49 | 31329 | bdslw102_a_pose_cache | 98.2% | 0.088 |
| 23 | 28501 | bdslw401_pose_cache_front | 93.0% | 0.253 |
| 36 | 25282 | bdslw102_a_pose_cache | 84.8% | 0.427 |
| 8 | 24810 | bdslw401_pose_cache_front | 91.1% | 0.299 |
| 16 | 24809 | bdslw401_pose_cache_front | 94.4% | 0.215 |
| 56 | 21483 | bdslw401_pose_cache_front | 94.8% | 0.205 |
| 63 | 21376 | bdslw401_pose_cache_front | 95.1% | 0.196 |
| 54 | 20869 | bdslw401_pose_cache_front | 89.8% | 0.330 |
| 52 | 18909 | bdslw401_pose_cache_front | 88.8% | 0.350 |
| 7 | 18154 | bdslw401_pose_cache_front | 94.0% | 0.227 |
| 43 | 17593 | bdslw401_pose_cache_front | 88.4% | 0.358 |
| 22 | 17285 | bdslw401_pose_cache_front | 93.8% | 0.233 |
| 12 | 17191 | bdslw401_pose_cache_front | 96.0% | 0.168 |
| 58 | 13244 | bdslw401_pose_cache_front | 91.7% | 0.286 |
| 38 | 10335 | bdslw401_pose_cache_front | 89.8% | 0.330 |
| 26 | 9969 | bdslw401_pose_cache_front | 92.5% | 0.266 |
| 18 | 8261 | bdslw102_a_pose_cache | 100.0% | 0.002 |
