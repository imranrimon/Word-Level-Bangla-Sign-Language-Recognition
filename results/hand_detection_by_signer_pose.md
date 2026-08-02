# Hand/face detection by signer (pose cache)

Cache: `data/bdsl_cache`  
Train-signer cohort statistics (excludes val/test/pretrain):
- Left-hand detection rate:  mean=42.5%, std=15.3
- Right-hand detection rate: mean=67.2%, std=17.5

Outlier flag: |z| >= 1.5 = mild, >= 2.0 = strong (potential identity-shortcut source)

| signer | split | clips | total frames | L-hand % | R-hand % |
|---:|---|---:|---:|---:|---:|
| U01 | train | 975 | 36987 | 78.5% **+2.4SD** | 48.4% |
| U02 | test | 744 | 35859 | 45.6% | 78.5% |
| U03 | pretrain | 583 | 22910 | 34.8% | 82.7% |
| U04 | train | 637 | 27129 | 36.6% | 51.1% |
| U05 | train | 600 | 28823 | 34.6% | 82.0% |
| U06 | train | 815 | 31395 | 40.5% | 76.1% |
| U07 | pretrain | 107 | 5734 | 61.1% | 88.0% |
| U08 | train | 639 | 29466 | 40.7% | 90.7% |
| U09 | train | 656 | 30755 | 39.8% | 73.2% |
| U10 | pretrain | 383 | 15831 | 47.0% | 68.6% |
| U11 | train | 771 | 30572 | 47.4% | 77.7% |
| U12 | train | 655 | 32986 | 21.6% | 38.1% -1.7SD |
| U13 | test | 621 | 29588 | 51.4% | 71.4% |
| U14 | pretrain | 112 | 7027 | 56.7% | 90.1% |
| U15 | val | 655 | 28405 | 29.0% | 73.8% |
| U16 | pretrain | 97 | 4831 | 85.0% **+2.8SD** | 59.8% |
| U17 | pretrain | 116 | 8429 | 95.3% **+3.5SD** | 60.6% |
| U18 | pretrain | 141 | 7362 | 43.6% | 72.5% |
