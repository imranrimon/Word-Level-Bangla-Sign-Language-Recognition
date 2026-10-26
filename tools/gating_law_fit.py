"""P1 ★#1 — Predictive distance-gating law.

Turns the distance-gating *finding* into a predictive *law*:
  mechanism_gain(target) = xling - pool   (pp, val Top-1 @10% labels, 3-seed mean)
regressed on the pose-motion MMD distance MMD(pretrain-pool -> target).

The gate scales the alignment mechanism, so the mechanism effect (xling - pool)
is exactly what the gate modulates; we model it vs distance, validate OUT-OF-SAMPLE
by leave-one-language-out (fit on 3, predict the 4th), and derive the critical
distance d* where the mechanism switches from helping (+) to hurting (-).

Data (the original single pretraining run, pseed 0; @10% val Top-1, fine-tune-seed means from
results/results_final.csv, `<stem>_lr01_f10_{mono,pool,xling}_seed<S>`):
  target  MMD     mono   pool   xling
  GSL     0.1206   --    72.00  74.06   (5-seed; closest target)
  BdSL    0.1429  50.8   53.3   55.0
  LSA64   0.1763  60.0   60.8   61.8
  KArSL   0.3445   --    69.9   65.5
  AUTSL   0.4115  69.0   70.0   63.5
  (LSFB and LIBRAS below; BISINDO was excluded post hoc in this version -- see the paper and
   results/provenance/bisindo_v1_lawpoint_2026-10-22.txt.)
MMD via tools/compute_lang_distance.py (RBF-MMD on per-joint speed signatures);
ALL targets recomputed in ONE consistent pass — supersedes earlier per-target
values (e.g. KArSL 0.314 -> 0.3445), so distances are directly comparable.

Budget-axis evidence (gate-isolation, docs: dgxl gated vs pool6-uniform, per-clip n=655):
  +7.48 pp @2.5%, +8.85 @5%, +3.36 @10%  -> the gate matters MORE when labels are scarcer.
"""
import numpy as np

# --- verified @10% per-target table ---
# LSFB (Belgian-French, top-60 signs, 74/9/16 signer SI split) added as the 6th
# language / prospective test on a NEW family. BISINDO EXCLUDED (below learnability
# floor: 34% max even @100%). LSFB left-align bundle fix (was a center-pad artifact
# giving 5.6%); corrected pool 40.84 / xling 35.78, MMD 0.4798.
# LIBRAS (MINDS-Libras, Brazilian SL, top-20 signs, 8/2/2 signer SI split) added as
# the 7th language / SECOND prospective test — chosen specifically to fill the empty
# distance gap [0.18,0.34] (MMD 0.3151 bisects it). 5-seed pool 46.17 / xling 41.49.
PROSPECTIVE = ["LSFB", "LIBRAS"]          # held out of every calibration fold
TARGETS = ["BdSL", "LSA64", "KArSL", "AUTSL", "GSL", "LSFB", "LIBRAS"]
DIST  = np.array([0.1429, 0.1763, 0.3445, 0.4115, 0.1206, 0.4798, 0.3151])   # consistent one-pass MMDs
POOL  = np.array([53.3, 60.8, 69.9, 70.0, 72.00, 40.84, 46.17])
XLING = np.array([55.0, 61.8, 65.5, 63.5, 74.06, 35.78, 41.49])   # GSL/LSFB/LIBRAS 5-seed
GAIN  = XLING - POOL                      # mechanism gain (pp) = what the gate modulates
# 7-lang OLS refreshed at runtime; the in-gap LIBRAS point (d=0.315, gain -4.69)
# lands on the backfire side of d* exactly as the gate predicts.
# 6-lang OLS: gain=4.91-24.29*MMD, R^2=0.92, d*=0.202, LOOCV sign 6/6 (gate robust);
# original-5 subset: R^2=0.997, d*=0.198 (the tight-linear idealization).

GATE_ISO = {"2.5%": 7.48, "5%": 8.85, "10%": 3.36}   # dgxl - pool6uniform (per-clip bootstrap)


def linfit(d, g):
    """OLS gain = a + b*d ; return a, b, R^2."""
    b, a = np.polyfit(d, g, 1)
    pred = a + b * d
    ss_res = np.sum((g - pred) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return a, b, r2


def main():
    print("=" * 68)
    print("P1 ★#1  PREDICTIVE DISTANCE-GATING LAW  (mechanism gain = xling - pool)")
    print("=" * 68)
    print(f"{'target':7} {'MMD':>6} {'pool':>6} {'xling':>6} {'gain(pp)':>9}")
    for t, d, p, x, g in zip(TARGETS, DIST, POOL, XLING, GAIN):
        print(f"{t:7} {d:6.3f} {p:6.1f} {x:6.1f} {g:+9.2f}")

    # --- global fit ---
    a, b, r2 = linfit(DIST, GAIN)
    dstar = -a / b
    from scipy.stats import spearmanr, pearsonr
    rho, prho = spearmanr(DIST, GAIN)
    r, pr = pearsonr(DIST, GAIN)
    print("\n--- GLOBAL LINEAR LAW ---")
    print(f"  gain(pp) = {a:+.2f} {b:+.2f}*MMD     R^2 = {r2:.3f}")
    print(f"  Pearson r = {r:+.3f} (p={pr:.3f}) | Spearman rho = {rho:+.3f} (p={prho:.3f})")
    print(f"  >>> CRITICAL DISTANCE d* = {dstar:.3f}  (mechanism helps if MMD < d*, backfires if MMD > d*)")

    # --- jackknife CI on d* (leave-one-language-out) ---
    dstars = []
    print("\n--- LEAVE-ONE-LANGUAGE-OUT (out-of-sample prediction) ---")
    maes = []
    for i, t in enumerate(TARGETS):
        mask = np.arange(len(TARGETS)) != i
        ai, bi, _ = linfit(DIST[mask], GAIN[mask])
        pred_i = ai + bi * DIST[i]
        err = pred_i - GAIN[i]
        maes.append(abs(err))
        dstars.append(-ai / bi)
        print(f"  hold out {t:6}: predicted gain {pred_i:+6.2f} pp | actual {GAIN[i]:+6.2f} pp "
              f"| err {err:+5.2f} pp | d*(3-lang)={-ai/bi:.3f}")
    print(f"  >>> LOOCV mean abs error = {np.mean(maes):.2f} pp  (max {np.max(maes):.2f} pp)")
    print(f"  >>> d* jackknife = {np.mean(dstars):.3f} +/- {np.std(dstars):.3f}  "
          f"range [{min(dstars):.3f}, {max(dstars):.3f}]")

    # sign-agreement: does the law predict the correct helps/hurts direction OOS?
    ok = 0
    for i in range(len(TARGETS)):
        mask = np.arange(len(TARGETS)) != i
        ai, bi, _ = linfit(DIST[mask], GAIN[mask])
        if np.sign(ai + bi * DIST[i]) == np.sign(GAIN[i]):
            ok += 1
    print(f"  >>> OOS sign agreement (helps vs backfires): {ok}/{len(TARGETS)} targets")

    # --- CLEAN protocol: calibration-N LOO (prospective langs NEVER in any fold)
    #     + each prospective language predicted from the frozen calibration fit ---
    prosp = [TARGETS.index(t) for t in PROSPECTIVE]
    calib = [i for i, t in enumerate(TARGETS) if t not in PROSPECTIVE]
    print(f"\n--- CLEAN PROTOCOL: {len(calib)}-calibration LOO "
          f"({', '.join(PROSPECTIVE)} excluded from every fold) ---")
    cerr, cok = [], 0
    for i in calib:
        m = [j for j in calib if j != i]
        ai, bi, _ = linfit(DIST[m], GAIN[m])
        pred = ai + bi * DIST[i]; e = pred - GAIN[i]
        cerr.append(abs(e)); cok += int(np.sign(pred) == np.sign(GAIN[i]))
        print(f"  hold {TARGETS[i]:6}: pred {pred:+6.2f} | actual {GAIN[i]:+6.2f} | err {e:+5.2f}")
    print(f"  >>> calibration-{len(calib)} LOO MAE = {np.mean(cerr):.3f} pp | sign {cok}/{len(calib)}")
    ac, bc, r2c = linfit(DIST[calib], GAIN[calib])
    from scipy.stats import spearmanr as _sp
    print(f"  {len(calib)}-calib fit: gain = {ac:+.2f} {bc:+.2f}*MMD  R^2={r2c:.3f}  d*={-ac/bc:.3f}  "
          f"Spearman={_sp(DIST[calib], GAIN[calib]).correlation:+.3f}")
    pok = 0
    for i in prosp:
        predP = ac + bc * DIST[i]
        s = np.sign(predP) == np.sign(GAIN[i]); pok += int(s)
        print(f"  >>> {TARGETS[i]:6} PROSPECTIVE (fit on {len(calib)} calib, unseen): "
              f"pred {predP:+.2f} | actual {GAIN[i]:+.2f} | sign {'OK' if s else 'X'}")
    print(f"  >>> COMBINED: {cok}/{len(calib)} calibration LOO + {pok}/{len(prosp)} prospective "
          f"= {cok+pok}/{len(TARGETS)} sign-correct")

    print("\n--- BUDGET AXIS (gate value vs label scarcity; gate-isolation deltas) ---")
    for k, v in GATE_ISO.items():
        print(f"  @{k:>4}: gated - uniform6 = {v:+.2f} pp")
    print("  -> gate value grows as labels shrink (consistent with a low-label mechanism).")


if __name__ == "__main__":
    main()
