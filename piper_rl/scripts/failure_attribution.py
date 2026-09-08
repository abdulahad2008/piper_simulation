"""Are the residual failures environment-determined, or are they noise?

At a tolerance tau, some episodes succeed and some do not. The question this
answers is whether the ones that fail are predictable from the episode's
sampled domain-randomisation vector and initial conditions. If a
cross-validated classifier cannot beat chance, the residual failures are
irreducible under this randomisation and there is nothing to fix by covering
more of the randomisation space; if it can, the failures are concentrated in a
region of the randomisation space and the paper should say which.

Reads only the archived per-episode CSVs.

    python -m piper_rl.scripts.failure_attribution \
        --results results/precision --cells v1_n2000 v2_n2000 --out paper/generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["obj_r", "obj_th", "obj_yaw", "obj_half_w", "obj_half_h",
            "obj_mass", "mu_obj", "mu_table", "tgt_r", "tgt_th",
            "kp_scale_mean", "kv_scale_mean", "frictionloss_scale_mean"]


def attribute(df: pd.DataFrame, tau_mm: float, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[FEATURES].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    valid = (df["lift"].astype(bool) & df["obj_settled"].astype(bool)).to_numpy()[ok]
    y = (~(valid & (df["place_err_mm"].to_numpy(float)[ok] <= tau_mm))).astype(int)

    out = {"tau_mm": float(tau_mm), "n": int(len(y)),
           "failure_rate": float(y.mean())}
    if y.sum() < 20 or (1 - y).sum() < 20:
        out["note"] = "too few of one class for a meaningful classifier"
        return out

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=seed))
    gb = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    out["auc_logistic"] = float(np.mean(
        cross_val_score(logit, X, y, cv=cv, scoring="roc_auc")))
    out["auc_gbm"] = float(np.mean(
        cross_val_score(gb, X, y, cv=cv, scoring="roc_auc")))

    gb.fit(X, y)
    imp = permutation_importance(gb, X, y, n_repeats=20, random_state=seed,
                                 scoring="roc_auc")
    order = np.argsort(imp.importances_mean)[::-1]
    out["permutation_importance"] = [
        {"feature": FEATURES[i], "mean": float(imp.importances_mean[i]),
         "std": float(imp.importances_std[i])} for i in order[:6]]
    return out


def icc_binary(df: pd.DataFrame, tau_mm: float) -> dict:
    """Intra-class correlation of the outcome within a fixed scene.

    Requires a nested evaluation (``--nested-noise M``): the same
    randomisation draw replayed under M independent noise streams. The ICC is
    the fraction of outcome variance that the scene explains. ICC near 0 means
    the residual failures are irreducible under this randomisation -- the same
    scene succeeds or fails depending only on noise; ICC near 1 means failure
    is a property of the scene and could in principle be predicted or trained
    away.

    Computed as the one-way random-effects ANOVA estimator on the binary
    success indicator, which is the standard estimator for this design and is
    what the paper reports.
    """
    if df["dr_seed"].nunique() == len(df):
        return {"note": "not a nested evaluation; ICC undefined"}
    valid = df["lift"].astype(bool) & df["obj_settled"].astype(bool)
    y = (valid & (df["place_err_mm"] <= tau_mm)).astype(float)
    g = df["dr_seed"].to_numpy()
    groups = [y.to_numpy()[g == k] for k in np.unique(g)]
    k = len(groups)
    m = float(np.mean([len(x) for x in groups]))
    grand = float(y.mean())
    msb = sum(len(x) * (x.mean() - grand) ** 2 for x in groups) / max(1, k - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in groups) / max(1, len(y) - k)
    icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + (m - 1) * msw) > 0 else 0.0
    return {"tau_mm": float(tau_mm), "n_scenes": int(k),
            "replays_per_scene": m, "success_rate": grand,
            "icc": float(max(0.0, min(1.0, icc))),
            "msb": float(msb), "msw": float(msw)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=str, default="results/precision")
    p.add_argument("--cells", type=str, nargs="+", required=True)
    p.add_argument("--taus", type=float, nargs="+", default=None,
                   help="mm; default: the cell's own c50 and 45 mm")
    p.add_argument("--out", type=str, default="paper/generated")
    args = p.parse_args(argv)

    from piper_rl.precision.stats import c50_from_errors

    R, O = Path(args.results), Path(args.out)
    (O / "tables").mkdir(parents=True, exist_ok=True)
    report = {}
    for cell in args.cells:
        f = R / f"{cell}.csv"
        if not f.exists():
            print(f"missing {f}")
            continue
        d = pd.read_csv(f)
        valid = d["lift"].astype(bool) & d["obj_settled"].astype(bool)
        taus = args.taus or [round(c50_from_errors(d["place_err_mm"], valid), 1), 45.0]
        report[cell] = [attribute(d, t) for t in taus]
        if "dr_seed" in d.columns and d["dr_seed"].nunique() < len(d):
            report[cell + "__icc"] = [icc_binary(d, t) for t in taus]
            for r in report[cell + "__icc"]:
                if "icc" in r:
                    print(f"{cell:16s} tau={r['tau_mm']:5.1f} mm  "
                          f"ICC={r['icc']:.3f} over {r['n_scenes']} scenes "
                          f"x {r['replays_per_scene']:.0f} noise replays")
        for r in report[cell]:
            if "auc_gbm" in r:
                top = ", ".join(f"{i['feature']} {i['mean']:+.3f}"
                                for i in r["permutation_importance"][:3])
                print(f"{cell:16s} tau={r['tau_mm']:5.1f} mm  "
                      f"fail={r['failure_rate']:.1%}  "
                      f"AUC logit {r['auc_logistic']:.3f} / gbm {r['auc_gbm']:.3f}  "
                      f"| top: {top}")
            else:
                print(f"{cell:16s} tau={r['tau_mm']:5.1f} mm  {r['note']}")

    (O / "failure_attribution.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
