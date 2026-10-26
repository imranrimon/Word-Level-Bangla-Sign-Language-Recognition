#!/usr/bin/env python
"""P1: score the pose-motion MMD against established transferability estimators
at predicting cross-lingual transfer gain (answers the reviewer's "why your distance
vs LEEP/LogME/OTDD/A-distance/naive?"). Label-free distance R^2 vs gain, LOOCV, sign.

Naive baselines (target #classes, train-size) are computed here; the label-free
distance estimators (energy/Frechet/A-distance/kNN) come from the robustness sweep
(tools/distance_robustness.py); LogME/LEEP (feature-based) come from
tools/logme_transfer.py (GPU: source-backbone features on target).

    python tools/transferability_baselines.py
"""
from __future__ import annotations
import numpy as np

# 5 pool->target points (headline law, tab:law)
TG   = ['GSL','BdSL','LSA64','KArSL','AUTSL']
GAIN = np.array([2.06, 1.70, 1.00, -4.40, -6.50])
MMD  = np.array([0.121,0.143,0.176,0.344,0.411])
NCLS = np.array([310, 60, 64, 70, 226], float)     # target vocabulary
SIZE = np.array([29175,5748,2240,6212,28130], float)  # target train clips

# label-free distance estimators (gain~estimator R^2, sign) from the robustness sweep
SWEEP = [('energy distance',0.932,'5/5'),('Frechet (Gaussian)',0.627,'2/5'),
         ('A-distance (H-div / Proxy-A)',0.310,'3/5'),('kNN two-sample',0.317,'3/5')]


def fit_scores(x, gain, negate=True):
    xx = -x if negate else x
    b,a = np.polyfit(xx,gain,1); pred=a+b*xx
    r2 = 1-((gain-pred)**2).sum()/((gain-gain.mean())**2).sum()
    rx=np.argsort(np.argsort(xx)); ry=np.argsort(np.argsort(gain))
    rho=float(np.corrcoef(rx,ry)[0,1])
    dstar = -a/b if b!=0 else np.median(xx)
    signacc = float(np.mean((xx>dstar)==(gain>0)) if b>0 else np.mean((xx<dstar)==(gain>0)))
    errs=[]
    for i in range(len(gain)):
        m=np.ones(len(gain),bool); m[i]=False
        bb,aa=np.polyfit(xx[m],gain[m],1); errs.append(abs((aa+bb*xx[i])-gain[i]))
    return r2, rho, signacc, float(np.mean(errs))


def main():
    print(f"{'estimator':30s} {'R2':>6} {'Spearman':>9} {'signAcc':>8} {'LOOCV-MAE':>10}")
    for name,x,neg in [('pose-motion MMD (ours)',MMD,True),
                       ('target #classes',NCLS,False),
                       ('target train-size',SIZE,False)]:
        r2,rho,sa,mae=fit_scores(x,GAIN,negate=neg)
        print(f"{name:30s} {r2:6.3f} {rho:+9.2f} {sa*100:6.0f}% {mae:10.2f}")
    print("  -- label-free distance estimators (robustness sweep) --")
    for name,r2,sign in SWEEP:
        print(f"{name:30s} {r2:6.3f} {'--':>9} {sign:>7} {'--':>10}")
    print("  -- LogME / LEEP (feature-based): run tools/logme_transfer.py (GPU) --")


if __name__ == '__main__':
    main()
