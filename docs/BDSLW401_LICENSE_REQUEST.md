# BdSLW401 release-permission request (draft email — P0, gates NeurIPS D&B)

**Why:** BdSLW401 is CC BY-NC-ND 4.0. Extracted pose arrays are a *derivative* (the
ND clause blocks redistribution) and pretrained weights are an unsettled grey area.
Metrics/analysis are always publishable, but releasing the **pose arrays + backbone
weights** — which the D&B artifact route benefits from — needs the authors' written
permission. This single email potentially unlocks the NeurIPS D&B path. Send it now;
it does not block the paper's method/benchmark claims.

---

**To:** BdSLW401 corresponding author(s) (arXiv:2503.02360)
**Subject:** Permission to release derived pose data / weights from BdSLW401 (non-commercial research)

Dear Dr. [NAME] and colleagues,

We are researchers at West Virginia University building an open, **signer-independent
benchmark for word-level Bangla Sign Language recognition**, and BdSLW401 is central
to our pretraining pool. We deeply appreciate your work releasing this dataset.

BdSLW401 is licensed CC BY-NC-ND 4.0. To support reproducible research we would like
to release, **for non-commercial academic use only** and **with full attribution**:

1. **MediaPipe-extracted 27-keypoint pose arrays** derived from BdSLW401 clips
   (skeleton coordinates only — no raw video, no reconstructable imagery), and
2. **Self-supervised pretrained backbone weights** trained on those pose arrays.

We recognise the ND clause makes both derivatives, which is why we are writing to
request your explicit written permission. In return we commit to:

- **Non-commercial** use only, with clear **attribution** and citation of BdSLW401
  (arXiv:2503.02360) and of your Relative Quantization Encoding (RQE);
- releasing **only pose keypoints** (never raw or reconstructable video);
- honouring any conditions you specify (e.g. gated access, acknowledgement text).

If a full release is not possible, we would still value permission to release the
**extraction pipeline, manifests, canonical split files, and result hashes** (the
"pointers + tooling" model accepted at NeurIPS D&B for YouTube-ASL), which lets others
reproduce our numbers without redistributing your data.

Thank you for considering this — and for the dataset. We are happy to discuss any
terms that work for you.

Best regards,
[YOUR NAME], [AFFILIATION]
[EMAIL]

---
*Send-checklist:* fill [NAME]/[YOUR NAME]/[AFFILIATION]/[EMAIL]; CC your advisor;
keep a copy of any written reply (needed as evidence for the D&B artifact statement).
