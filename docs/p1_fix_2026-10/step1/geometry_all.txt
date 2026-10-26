| corpus | clips | med T | shoulder w | R wrist own/wrong (s) | L wrist own/wrong (s) | hand span (s) | hand/shoulder speed | nose up | miss body/R/L | dropout speed share | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BdSL (BdSLW60 si train) | 300 | 39 | 0.250 | 0.08/1.68 | 0.09/0.97 | 0.38 | 10.8 | 1.00 | 0.00/0.34/0.56 | 0.61 | **PASS**  |
| BdSLW60 pretrain split | 300 | 40 | 0.241 | 0.07/1.66 | 0.10/0.98 | 0.45 | 8.5 | 1.00 | 0.00/0.24/0.51 | 0.56 | **PASS**  |
| BdSLW401 front cache (pool) | 300 | 78 | 0.262 | 0.08/1.28 | 0.09/0.96 | 0.36 | 10.4 | 1.00 | 0.00/0.52/0.63 | 0.80 | **PASS**  |
| BdSLW401 si train | 300 | 80 | 0.261 | 0.08/1.45 | 0.09/1.02 | 0.36 | 10.8 | 1.00 | 0.00/0.52/0.66 | 0.78 | **PASS**  |
| BdSLW102a cache (pool) | 176 | 125 | 0.125 | 0.15/1.97 | 0.13/1.24 | 0.46 | 5.5 | 0.99 | 0.00/0.60/0.75 | 0.86 | **PASS**  |
| WLASL cache v1 (pool, OLD) | 300 | 56 | 0.019 | 16.72/16.57 | 14.38/14.59 | 2.61 | 0.1 | 0.22 | 0.32/0.01/0.01 | 0.66 | **FAIL** A1 shoulder width 0.019 outside [0.02,1.0]; A2 R-hand wrist 16.72s from pose wrist (>=0.35s); A3 R-hand nearer the WRONG body wrist (16.72 vs 16.57); A2 L-hand wrist 14.38s from pose wrist (>=0.35s); A4 hand span 2.61s outside [0.08,0.9]; A5 hand/shoulder speed ratio 0.10 < 1.5; A6 nose above shoulders in only 0.22 of frames |
| WLASL cache v2 (pool, FIXED) | 300 | 58 | 0.303 | 0.07/1.07 | 0.07/0.78 | 0.27 | 13.3 | 1.00 | 0.00/0.32/0.55 | 0.71 | **PASS**  |
| LSA64 si train | 300 | 122 | 0.206 | 0.07/2.11 | 0.10/1.22 | 0.40 | 10.8 | 1.00 | 0.00/0.12/0.72 | 0.55 | **PASS**  |
| AUTSL si train | 300 | 61 | 0.206 | 0.05/1.15 | 0.05/1.11 | 0.22 | 10.0 | 1.00 | 0.00/0.10/0.13 | 0.80 | **PASS**  |
| KArSL si train | 300 | 19 | 0.153 | 0.05/2.36 | 0.07/2.36 | 0.42 | 2.3 | 1.00 | 0.00/0.02/0.01 | 0.07 | **PASS**  |
| GSL si train | 300 | 15 | 0.217 | 0.07/1.48 | 0.06/1.52 | 0.34 | 11.4 | 1.00 | 0.00/0.11/0.06 | 0.44 | **PASS**  |
| LSFB si train | 300 | 17 | 0.217 | 0.08/1.06 | 0.07/1.03 | 0.32 | 7.8 | 1.00 | 0.00/0.00/0.00 | 0.00 | **PASS**  |
| LIBRAS si train | 300 | 138 | 0.182 | 0.07/1.62 | 0.08/1.18 | 0.34 | 12.3 | 1.00 | 0.00/0.43/0.71 | 0.83 | **PASS**  |
| BISINDO si train | 300 | 70 | 0.180 | 0.07/2.66 | 0.09/0.87 | 0.30 | 22.0 | 1.00 | 0.00/0.35/0.91 | 0.77 | **PASS**  |
| ASL-Citizen-60 si train | 300 | 78 | 0.406 | 0.09/1.10 | 0.10/0.90 | 0.38 | 14.9 | 1.00 | 0.00/0.63/0.70 | 0.72 | **PASS**  |
