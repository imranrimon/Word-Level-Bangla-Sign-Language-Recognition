# PREREG addendum A — fixed v2 distances (x-axis)

Written 2026-10-22 05:48:22 EDT. At writing NO v2 fine-tune result exists (`results_p1fix_v2.csv` absent) and no v2
backbone has been trained (the k-means target job has not even been submitted — submit cap). Values
are the PREREG §3 definition, computed by job J38 (`step2/distances_v2.json`, `--draws 20`, seed 0),
`results.v2.pool_v2.<lang>.pooled.mean`. Draw SDs shown for information. These values are now frozen.

Deviation from PREREG wording (disclosed): the first full run (job J37) had a bug in the v1-replica
source weighting (affects only the `oldsrc_*` sources, which are sensitivity/decomposition rows, not
the primary `pool_v2` source); it was fixed and everything recomputed once (job J38) before this
addendum. The unmodified v1 tool rerun reproduces the paper values exactly (step2/v1_tool_rerun.txt).

| language | d_l (v2, pool_v2, pooled bw) | draw SD |
|---|---|---|
| BdSL | 0.0810 | 0.0090 |
| LSA64 | 0.1572 | 0.0104 |
| AUTSL | 0.1557 | 0.0100 |
| KArSL | 0.4041 | 0.0169 |
| GSL | 0.0913 | 0.0072 |
| LSFB | 0.1447 | 0.0084 |
| LIBRAS | 0.1373 | 0.0094 |
| BISINDO | 0.3959 | 0.0132 |

Primary slope prior SD therefore = 50 / (max d − min d) = 154.8 pp per unit MMD^2_u.
