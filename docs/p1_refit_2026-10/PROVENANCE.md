# Provenance of `PREREG.md` in this release

`PREREG.md` is the pre-registration of the 7-pretraining-seed refit, written and hash-locked
before any result for pretraining seeds 3-6 was read.

| file | SHA-256 |
|---|---|
| original `PREREG.md` (as hash-locked; recorded in `PREREG.sha256`, file time 2026-10-22 04:51 EDT, 10,730 bytes) | `6eb9d4261550c808582f2362d63598416d865e3f642e599bd3a85a9ef5bb4891` |
| `PREREG.md` as distributed here (redacted copy) | `1d2d3792d7ad571541540b2a944b94ce67e5d0c274912019ade0d9b073dc5733` |

The distributed copy differs from the original only on lines 5-7, where 13 cluster job numbers
(the status of the pretraining and fine-tuning jobs at the time of writing) were replaced by the
pseudonymous labels `J20`-`J32` to remove infrastructure identifiers. No other byte was changed:
the design, models, priors, decision rule and sensitivity analyses are verbatim.

`PREREG.sha256` and the `prereg_sha256` field of `refit_results.json` refer to the **original,
unredacted** file, so hashing the distributed copy gives the second value above, not the
recorded one. `tools/p1_refit/report.py` recognises the redacted copy through this file.
The same job labels are used consistently in every document of this release.
