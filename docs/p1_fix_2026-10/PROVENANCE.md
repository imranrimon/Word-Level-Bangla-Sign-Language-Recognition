# Provenance of the pre-registration files in this release

| file | as hash-locked (recorded in `PREREG.sha256`) | as distributed here |
|---|---|---|
| `PREREG.md` (2026-10-22 05:17:19 EDT) | `bd28232ae8703eaf28cf6b261238f8eb41f5f456970ad497fe4d403d6c95fd75` | identical (byte-for-byte original) |
| `PREREG_addendum_A_distances.md` (2026-10-22 05:48:22 EDT) | `e19f78f8e749183022e0955be59f8c3725083108b9458aa9f0f459962c0d9757` | `314d516b3a7b40e30e49a2086fd643a2a6859418b126bb03e4d663e473547937` (redacted copy) |

`PREREG.md` is distributed unchanged; `sha256sum PREREG.md` reproduces the recorded hash.

The distributed `PREREG_addendum_A_distances.md` differs from the original only on lines 5, 8
and 10, where two cluster job numbers (the distance runs) were replaced by the pseudonymous labels
`J37` (first, superseded run) and `J38` (the run that produced `step2/distances_v2.json`) to remove
infrastructure identifiers. The frozen distances and every other byte are verbatim. The hash in
`PREREG.sha256` refers to the **original, unredacted** addendum. The same job labels are used
consistently in every document of this release.
