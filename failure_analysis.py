"""Run N deterministic episodes and correlate failures with the sampled
domain-randomisation parameters. Read-only: nothing in the repo is modified."""
import sys, numpy as np
from stable_baselines3 import SAC
from piper_rl.config import EnvConfig
from piper_rl.piper_env import PiperPickPlaceEnv

MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/sac_full/best/best_model.zip"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000

cfg = EnvConfig().eval_variant()
env = PiperPickPlaceEnv(cfg)
model = SAC.load(MODEL, device="cpu")
print(f"model: {MODEL}")

# monkeypatch to capture the DR draw
orig = env.randomizer.randomize
cap = {}
def wrapped(rng):
    d = orig(rng)
    cap.clear(); cap.update(d)
    return d
env.randomizer.randomize = wrapped

rows = []
for ep in range(N):
    obs, _ = env.reset(seed=SEED + ep)
    dr = dict(cap)
    done = False; R = 0.0; steps = 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        R += r; steps += 1
        done = term or trunc
    es = info["episode_summary"]
    succ = int(es["success"]); grasp = int(es["grasp_success"])
    lift = int(es["lift_success"]); place_err = es["placement_error"]
    ox, oy = dr["obj_xy"]; tx, ty = dr["target_xy"]
    rows.append(dict(
        succ=succ, grasp=grasp, lift=lift, R=R, steps=steps, place_err=place_err,
        mass=dr.get("obj_mass", np.nan), mu=dr.get("obj_friction", np.nan),
        half_w=dr["obj_half_w"], half_h=dr["obj_half_h"],
        obj_r=float(np.hypot(ox, oy)), obj_th=float(np.arctan2(oy, ox)),
        tgt_r=float(np.hypot(tx, ty)), tgt_th=float(np.arctan2(ty, tx)),
        yaw=float(dr["obj_yaw"]),
        reach=float(np.hypot(ox - tx, oy - ty)),
        term=info.get("termination", "?"),
    ))
    if (ep + 1) % 25 == 0:
        s = np.mean([r["succ"] for r in rows])
        print(f"  {ep+1}/{N}  running success {s:.3f}", flush=True)

import collections
S = np.array([r["succ"] for r in rows], bool)
print("\n" + "=" * 62)
print(f"episodes {len(rows)}  success {S.mean()*100:.1f}%  "
      f"grasp {np.mean([r['grasp'] for r in rows])*100:.1f}%  "
      f"lift {np.mean([r['lift'] for r in rows])*100:.1f}%")
p = S.mean(); se = np.sqrt(p*(1-p)/len(S))
print(f"95% CI  {100*(p-1.96*se):.1f}% - {100*(p+1.96*se):.1f}%")
print("failure terminations:", collections.Counter(
    r["term"] for r in rows if not r["succ"]))
print("failure stage:", collections.Counter(
    ("no-grasp" if not r["grasp"] else "grasp-no-lift" if not r["lift"]
     else "lift-no-place") for r in rows if not r["succ"]))
print("-" * 62)
print(f"{'param':<10}{'succ mean':>12}{'fail mean':>12}{'fail min':>10}{'fail max':>10}")
for k in ["mass", "mu", "half_w", "half_h", "obj_r", "obj_th", "tgt_r",
          "yaw", "reach"]:
    v = np.array([r[k] for r in rows], float)
    if np.isnan(v).all():
        continue
    f = v[~S]
    print(f"{k:<10}{v[S].mean():12.4f}{f.mean():12.4f}"
          f"{(f.min() if len(f) else np.nan):10.4f}"
          f"{(f.max() if len(f) else np.nan):10.4f}")

# per-quartile success on the most suspicious continuous params
print("-" * 62)
for k in ["mass", "mu", "half_w", "half_h", "obj_r", "tgt_r"]:
    v = np.array([r[k] for r in rows], float)
    if np.isnan(v).all():
        continue
    q = np.quantile(v, [0, .25, .5, .75, 1.0])
    cells = []
    for i in range(4):
        m = (v >= q[i]) & (v <= q[i+1] if i == 3 else v < q[i+1])
        cells.append(f"{S[m].mean()*100:5.1f}%({m.sum()})")
    print(f"{k:<10} quartile success: " + "  ".join(cells))
env.close()

print("-" * 62)
pe = np.array([r["place_err"] for r in rows], float)
lifted = np.array([r["lift"] for r in rows], bool)
print("effective success if the target tolerance were tighter than 45 mm:")
for tol in [0.045, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010]:
    ok = lifted & (pe <= tol)
    print(f"   tol {tol*1000:5.0f} mm -> {ok.mean()*100:5.1f}%")
