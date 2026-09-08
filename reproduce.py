#!/usr/bin/env python3
"""Regenerate every table and figure of the precision-floor paper.

Reads only the archived per-episode CSVs in ``results/precision/`` -- never the
simulator -- so a reviewer with the repository can reproduce every printed
number without a GPU, a licence, or six hours of wall clock:

    python reproduce.py --results results/precision --out paper/generated

Outputs
-------
    tables/*.tex        LaTeX tables, \\input-ed by main.tex
    figures/*.pdf       Figure 1 panels and the checkpoint curve
    summary.json        every number, machine-readable
    summary.md          the same, human-readable

Any table whose input CSV is missing is skipped with a warning, so the script
runs on a partially-completed campaign and tells you what is still missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from piper_rl.precision.stats import (TAU_GRID_MM, bootstrap_c, c50_from_errors,
                                      c_quantile, iqm, success_curve,
                                      two_proportion_z, wilson)


# --------------------------------------------------------------------- #
def load(results: Path, tag: str) -> pd.DataFrame | None:
    p = results / f"{tag}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    # An episode counts as a success at tolerance tau only if it also satisfies
    # the non-tolerance parts of the success condition. `obj_settled` and
    # `lift` are those parts; an episode that never lifted the object is a
    # failure at every tolerance, however close the object happens to lie.
    d["valid"] = d["lift"].astype(bool) & d["obj_settled"].astype(bool)
    return d


def cell_summary(d: pd.DataFrame, name: str, boot: int = 2000) -> dict:
    e = d["place_err_mm"].to_numpy(float)
    v = d["valid"].to_numpy(bool)
    n = len(d)
    curve = success_curve(e, TAU_GRID_MM, v)
    c50 = bootstrap_c(e, v, 0.5, draws=boot)
    c90 = bootstrap_c(e, v, 0.9, draws=boot)
    lifted = e[d["lift"].astype(bool).to_numpy()]
    rel = d["tcp_err_at_release_mm"].to_numpy(float)
    sca = d["scatter_mm"].to_numpy(float)
    return {
        "cell": name, "n": n,
        "success_at_45": wilson(int(np.sum(v & (e <= 45))), n).__dict__,
        "grasp": float(d["grasp"].mean()), "lift": float(d["lift"].mean()),
        "curve": {str(k): vv.__dict__ for k, vv in curve.items()},
        "c50_mm": c50.__dict__, "c90_mm": c90.__dict__,
        "err_median_mm": float(np.median(lifted)) if lifted.size else float("nan"),
        "err_p90_mm": float(np.percentile(lifted, 90)) if lifted.size else float("nan"),
        "tcp_err_at_release_median_mm": float(np.nanmedian(rel)),
        "scatter_median_mm": float(np.nanmedian(sca)),
        "clamp_rate": float(d["clamp_rate"].mean()),
        "steps_mean": float(d["steps"].mean()),
        "max_step_dist_mm": float(d["max_step_dist_mm"].iloc[0]),
        "action_smoothing": float(d["action_smoothing"].iloc[0]),
        "control_hz": float(d["control_hz"].iloc[0]),
        "max_joint_vel": float(d["max_joint_vel"].iloc[0]),
        "solver_timestep": float(d["solver_timestep"].iloc[0]),
        "solver_solref_scale": float(d["solver_solref_scale"].iloc[0]),
        "solver_iterations": int(d["solver_iterations"].iloc[0]),
        "solver_impratio": float(d["solver_impratio"].iloc[0]),
        "git_hash": str(d["git_hash"].iloc[0]),
    }


# --------------------------------------------------------------------- #
PRETTY = {"pi_v1": r"$\pi_{v1}$", "pi_v2": r"$\pi_{v2}$"}


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    return r"\texttt{" + name.replace("v2_solver_", "").replace("_", r"\_") + "}"


def tex_curve_table(summaries: list[dict], label: str, caption: str) -> str:
    head = " & ".join(f"{t:g}" for t in TAU_GRID_MM)
    lines = [r"\begin{table}[t]", r"\centering\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{@{}l" + "c" * len(TAU_GRID_MM) + "c@{}}",
             r"\toprule",
             r"policy / cell & " + head + r" & $c_{50}$ (mm) \\",
             r"$\tau$ (mm) & " + " & ".join([""] * len(TAU_GRID_MM)) + r" & \\",
             r"\midrule"]
    for s in summaries:
        cells = []
        for t in TAU_GRID_MM:
            iv = s["curve"][str(float(t))]
            cells.append(f"{100*iv['point']:.1f} [{100*iv['lo']:.1f}, {100*iv['hi']:.1f}]")
        c = s["c50_mm"]
        lines.append(f"{pretty(s['cell'])} & " + " & ".join(cells)
                     + f" & {c['point']:.1f} [{c['lo']:.1f}, {c['hi']:.1f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def tex_solver_table(summaries: list[dict], baseline: dict) -> str:
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Phase 0 kill gate (H0d), eval-only. The released policy "
             r"$\pi_{v2}$ re-evaluated under each contact-solver setting, "
             r"$n{=}1000$ episodes per cell on a common evaluation seed block. "
             r"$\Delta c_{50}$ is relative to the nominal solver; a cell moves "
             r"the floor only if its interval excludes the nominal point.}",
             r"\label{tab:killgate}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"solver cell & $S(45)$ \% & median $e$ (mm) & $c_{50}$ (mm) [95\% CI] & $\Delta c_{50}$ \\",
             r"\midrule"]
    for s in summaries:
        c, a = s["c50_mm"], s["success_at_45"]
        d = c["point"] - baseline["c50_mm"]["point"]
        lines.append(
            f"{pretty(s['cell'])} & "
            f"{100*a['point']:.1f} & {s['err_median_mm']:.1f} & "
            f"{c['point']:.1f} [{c['lo']:.1f}, {c['hi']:.1f}] & {d:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------- #
def figure_curves(summaries: list[dict], out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    taus = np.array(TAU_GRID_MM, dtype=float)
    for s in summaries:
        pts = np.array([s["curve"][str(t)]["point"] for t in taus]) * 100
        lo = np.array([s["curve"][str(t)]["lo"] for t in taus]) * 100
        hi = np.array([s["curve"][str(t)]["hi"] for t in taus]) * 100
        line, = ax.plot(taus, pts, marker="o", ms=3.5, lw=1.4, label=s["cell"])
        ax.fill_between(taus, lo, hi, alpha=0.15, color=line.get_color(), lw=0)
    ax.axhline(50, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"tolerance $\tau$ (mm)")
    ax.set_ylabel(r"success rate $S(\tau)$ (%)")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 47); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fit_scaling_law(rows: list[dict]) -> dict:
    """Fit Xu et al.'s form with environment steps in place of demonstrations.

    They report log N  ~  a/(P - c) + b, where N is the number of
    demonstrations needed to reach precision P and c is the limit precision.
    The RL analogue replaces N by environment steps and P by the precision
    actually achieved at that budget, so the curve is read the other way round:
    for each stored checkpoint we have (steps, achieved precision), and we fit

        log(steps) = a / (P - c) + b,    P = 1 / c50   (mm^-1),

    over the checkpoints that actually solve the task. `c` is then a limit
    precision in the same units as c50. The fit is reported alongside the
    model-free c50 and is explicitly the model-DEPENDENT estimator: it is only
    meaningful if the curve has visibly flattened, which is stated with it.
    """
    from scipy.optimize import curve_fit

    usable = [r for r in rows if np.isfinite(r["c50"]) and r["success_45"] > 0.5]
    if len(usable) < 5:
        return {"note": f"only {len(usable)} checkpoints solve the task; "
                        "the scaling-law fit is not attempted"}
    steps = np.array([r["steps"] for r in usable], float)
    c50 = np.array([r["c50"] for r in usable], float)
    P = 1.0 / c50                       # precision, mm^-1

    def f(P_, a, c_inv, b):
        return a / np.maximum(P_ - c_inv, 1e-6) + b

    try:
        popt, pcov = curve_fit(f, P, np.log(steps),
                               p0=[1e-3, 0.9 * P.max(), 10.0], maxfev=20000)
    except Exception as e:
        return {"note": f"fit failed: {e}"}
    a, c_inv, b = popt
    resid = np.log(steps) - f(P, *popt)
    ss = 1 - float(np.sum(resid ** 2) / np.sum((np.log(steps) - np.log(steps).mean()) ** 2))
    c_hat = 1.0 / c_inv if c_inv > 0 else float("inf")
    return {"c_hat_mm": float(c_hat), "a": float(a), "b": float(b),
            "r2_log_steps": ss, "n_checkpoints_used": len(usable),
            "c50_range_mm": [float(c50.min()), float(c50.max())],
            "caveat": "model-dependent; compare against the model-free c50 and "
                      "read only if the curve has flattened"}


def figure_checkpoints(runs: dict, out: Path) -> None:
    """Two panels, because c50 is undefined for most of a training run.

    Left: success at the released 45 mm tolerance, defined at every checkpoint.
    Right: c50, defined only once the policy succeeds on more than half of
    episodes -- which is the honest way to show that the floor is a property of
    the *converged* policy and not something that decays smoothly from the
    first checkpoint. Plotting an imputed c50 for a policy that solves 4 % of
    episodes would be an invention.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sac_full": "#2b6cb0", "corner_ft": "#b7791f"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for name, rows in runs.items():
        col = colors.get(name, None)
        rows = sorted(rows, key=lambda r: r["steps"])
        steps = np.array([r["steps"] for r in rows], float) / 1e6
        s45 = np.array([100 * r["success_45"] for r in rows], float)
        c50 = np.array([r["c50"] for r in rows], float)
        lo = np.array([r["c50_lo"] for r in rows], float)
        hi = np.array([r["c50_hi"] for r in rows], float)
        ok = np.isfinite(c50)
        line, = a1.plot(steps, s45, marker="o", ms=3.2, lw=1.3, color=col,
                        label=name.replace("_", r"\_"))
        col = line.get_color()
        if ok.any():
            a2.plot(steps[ok], c50[ok], marker="o", ms=3.2, lw=1.3, color=col,
                    label=name.replace("_", r"\_"))
            a2.fill_between(steps[ok], lo[ok], hi[ok], alpha=0.18, lw=0,
                            color=col)
        if (~ok).any():
            a2.scatter(steps[~ok], np.zeros((~ok).sum()), marker="x", s=14,
                       color="0.55")
    a1.axhline(50, color="0.5", lw=0.8, ls=":")
    a1.set_ylim(0, 100)
    a1.set_ylabel(r"$S(45\,\mathrm{mm})$ (%)")
    a1.set_xlabel("environment steps (M)")
    a1.set_title("Success at the released tolerance", fontsize=9)
    a1.legend(fontsize=7, frameon=False)
    a2.text(0.03, 0.06, r"$\times$: $S(\tau)$ never reaches 50 %,"
                        "\n$c_{50}$ undefined",
            transform=a2.transAxes, fontsize=7, color="0.35")
    a2.legend(fontsize=7, frameon=False)
    a2.set_xlabel("environment steps (M)")
    a2.set_ylabel(r"$c_{50}$ (mm)")
    a2.set_title("Precision floor vs. training budget", fontsize=9)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=str, default="results/precision")
    p.add_argument("--out", type=str, default="paper/generated")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args(argv)

    R, O = Path(args.results), Path(args.out)
    (O / "tables").mkdir(parents=True, exist_ok=True)
    (O / "figures").mkdir(parents=True, exist_ok=True)
    out: dict = {"missing": [], "cells": {}}

    # ---- baselines -------------------------------------------------- #
    base = []
    for tag, name in [("v1_n2000", "pi_v1"), ("v2_n2000", "pi_v2")]:
        d = load(R, tag)
        if d is None:
            out["missing"].append(tag); continue
        s = cell_summary(d, name, args.bootstrap)
        base.append(s); out["cells"][name] = s
    if base:
        (O / "tables" / "baselines.tex").write_text(tex_curve_table(
            base, "tab:baselines",
            r"Success rate versus tolerance for the two released policies, "
            r"$n{=}2000$ evaluation episodes each (seeds 10000--11999), "
            r"domain randomisation and observation noise on, deterministic "
            r"actions, curriculum off. 95\% Wilson intervals; $c_{50}$ with a "
            r"2000-draw percentile bootstrap."))
        figure_curves(base, O / "figures" / "baselines.pdf",
                      r"Released policies, $n=2000$ each")
        if len(base) == 2:
            e1 = load(R, "v1_n2000"); e2 = load(R, "v2_n2000")
            tests = {}
            for t in TAU_GRID_MM:
                s1 = int(((e1["place_err_mm"] <= t) & e1["valid"]).sum())
                s2 = int(((e2["place_err_mm"] <= t) & e2["valid"]).sum())
                tests[str(t)] = two_proportion_z(s2, len(e2), s1, len(e1))
            out["v2_vs_v1"] = tests

    # ---- solver kill gate ------------------------------------------- #
    gate_tags = ["v2_solver_nominal", "v2_solver_dt1ms", "v2_solver_dt4ms",
                 "v2_solver_solref0.5", "v2_solver_solref2",
                 "v2_solver_iter20", "v2_solver_imp1",
                 "v2_solver_extremeA", "v2_solver_extremeB"]
    gate = []
    for tag in gate_tags:
        d = load(R, tag)
        if d is None:
            out["missing"].append(tag); continue
        s = cell_summary(d, tag, args.bootstrap)
        gate.append(s); out["cells"][tag] = s
    if gate:
        nominal = next((g for g in gate if g["cell"].endswith("nominal")), gate[0])
        (O / "tables" / "killgate.tex").write_text(tex_solver_table(gate, nominal))
        figure_curves(gate, O / "figures" / "killgate.pdf",
                      r"H0d: contact-solver cells, $\pi_{v2}$, $n=1000$ each")
        pts = [g["c50_mm"]["point"] for g in gate]
        spread = max(pts) - min(pts)
        out["killgate_c50_spread_mm"] = spread
        out["killgate_nominal_c50_mm"] = nominal["c50_mm"]["point"]
        out["killgate_max_abs_delta_mm"] = max(
            abs(p_ - nominal["c50_mm"]["point"]) for p_ in pts)
        # cells whose interval excludes the nominal point estimate
        moved = [g["cell"] for g in gate
                 if g["c50_mm"]["lo"] > nominal["c50_mm"]["point"]
                 or g["c50_mm"]["hi"] < nominal["c50_mm"]["point"]]
        out["killgate_cells_that_moved"] = moved
        (O / "tables" / "inline_gate_span.tex").write_text(
            f"at most {out['killgate_max_abs_delta_mm']:.1f}\\,mm "
            f"(a {spread:.1f}\\,mm total span across nine cells, "
            f"against a nominal {nominal['c50_mm']['point']:.1f}\\,mm)")

    # ---- release decomposition (H0e) --------------------------------- #
    dec = []
    for tag, name in [("v1_n2000", "pi_v1"), ("v2_n2000", "pi_v2")]:
        d = load(R, tag)
        if d is None:
            continue
        # Only episodes in which the object was actually lifted and released.
        # Including dropped episodes makes the scatter/terminal-error
        # correlation spuriously large: a dropped object trivially has a huge
        # "scatter" and a huge terminal error, and the correlation then
        # measures dropping, not release physics.
        held = d[(d["release_step"] > 0) & (d["lift"] == 1)]
        dec.append({"cell": name, "n": len(held),
                    "tcp_err_median_mm": float(held["tcp_err_at_release_mm"].median()),
                    "scatter_median_mm": float(held["scatter_mm"].median()),
                    "place_err_median_mm": float(held["place_err_mm"].median()),
                    "corr_tcp_place": float(held["tcp_err_at_release_mm"]
                                            .corr(held["place_err_mm"])),
                    "corr_scatter_place": float(held["scatter_mm"]
                                                .corr(held["place_err_mm"])),
                    "tcp_err_p90_mm": float(held["tcp_err_at_release_mm"].quantile(0.9)),
                    "scatter_p90_mm": float(held["scatter_mm"].quantile(0.9))})
    if dec:
        out["release_decomposition"] = dec
        rows = [r"\begin{table}[t]", r"\centering\small",
                r"\caption{H0e: decomposition of the terminal error into the "
                r"tool-centre-point error at the moment the object leaves the "
                r"fingers and the distance the object then travels before it "
                r"settles. Medians over episodes in which the object was lifted "
                r"and released.}", r"\label{tab:release}",
                r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
                r"policy & $n$ & TCP err.\ at release & post-release scatter & "
                r"terminal $e$ & $\rho$(TCP, $e$) / $\rho$(scatter, $e$) \\",
                r"\midrule"]
        for r in dec:
            rows.append(f"{pretty(r['cell'])} & {r['n']} & "
                        f"{r['tcp_err_median_mm']:.1f} & {r['scatter_median_mm']:.1f} & "
                        f"{r['place_err_median_mm']:.1f} & "
                        f"{r['corr_tcp_place']:.2f} / {r['corr_scatter_place']:.2f} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (O / "tables" / "release.tex").write_text("\n".join(rows))

    # ---- checkpoint curve (H0b) -------------------------------------- #
    # Checkpoint curves, grouped BY RUN. sac_full and corner_ft share a step
    # axis (the fine-tune continues the baseline's counter) but they are two
    # different training histories; pooling them would draw one curve through
    # two runs and invent a discontinuity.
    by_run: dict[str, list[dict]] = {}
    for f in sorted(R.glob("ckpt_*.csv")):
        stem = f.stem                      # ckpt_<run>_<steps>
        run_name = stem[len("ckpt_"):stem.rindex("_")]
        steps = int(stem.rsplit("_", 1)[-1])
        d = load(R, stem)
        e, v = d["place_err_mm"].to_numpy(float), d["valid"].to_numpy(bool)
        iv = bootstrap_c(e, v, 0.5, draws=500)
        by_run.setdefault(run_name, []).append(
            {"run": run_name, "steps": steps, "c50": iv.point, "c50_lo": iv.lo,
             "c50_hi": iv.hi, "n": len(d),
             "success_45": float(((e <= 45) & v).mean())})
    if by_run:
        out["checkpoint_curve"] = {k: sorted(v, key=lambda r: r["steps"])
                                   for k, v in by_run.items()}
        out["scaling_law_fit"] = {k: fit_scaling_law(v)
                                  for k, v in by_run.items()}
        figure_checkpoints(out["checkpoint_curve"],
                           O / "figures" / "checkpoints.pdf")
        out["budget_plateau"] = {}
        for k, v in out["checkpoint_curve"].items():
            solved = [r for r in v if np.isfinite(r["c50"])]
            if len(solved) >= 3:
                out["budget_plateau"][k] = {
                    "steps_from": solved[0]["steps"],
                    "steps_to": solved[-1]["steps"],
                    "c50_min": min(r["c50"] for r in solved),
                    "c50_max": max(r["c50"] for r in solved),
                    "c50_first": solved[0]["c50"], "c50_last": solved[-1]["c50"],
                    "n_defined": len(solved),
                    "argmin_steps": min(solved, key=lambda r: r["c50"])["steps"]}

    (O / "summary.json").write_text(json.dumps(out, indent=2, default=float))

    md = ["# Generated results", ""]
    if out["missing"]:
        md += ["**Missing CSVs (cells not yet run):** " +
               ", ".join(f"`{m}`" for m in out["missing"]), ""]
    for name, s in out["cells"].items():
        c = s["c50_mm"]
        md.append(f"- **{name}** (n={s['n']}): S(45)="
                  f"{100*s['success_at_45']['point']:.1f}%, "
                  f"c50={c['point']:.1f} [{c['lo']:.1f}, {c['hi']:.1f}] mm, "
                  f"median e={s['err_median_mm']:.1f} mm, "
                  f"clamp rate={s['clamp_rate']:.2f}")
    (O / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if out["missing"]:
        print(f"\n{len(out['missing'])} cells still missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
