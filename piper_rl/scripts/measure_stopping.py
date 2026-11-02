"""Measure the PiPER's stopping time and distance in the simulator.

The ISO/TS 15066 protective separation distance needs the robot's reaction
time and its stopping time and distance. This script measures them in the
same control stack the policies use (20 Hz commands, action smoothing, the
joint-velocity and acceleration clamps, MuJoCo position actuators), so the
numbers include every lag a policy or a scripted controller is subject to.

Protocol
--------
1. Reset the no-human scene from the standard start pose.
2. Drive the TCP in a fixed Cartesian direction at full command (action
   component +1 or -1, i.e. 0.03 m per 20 Hz step = 0.60 m/s nominal) for
   ``run_steps`` control steps so the arm reaches its steady speed.
3. At the next control step issue the stop command and keep issuing it.
   Two stop commands are measured:
     zero-action   the policy's only stop: action = 0, so the commanded joint
                   target stops advancing and relaxes through the low-pass
                   filter;
     hold-current  a scripted controller's stop: the commanded joint target
                   is set to the measured joint position at the stop tick and
                   held there (bypasses the smoothing filter).
4. Record the TCP position at every physics substep (500 Hz). Stopping time
   is the time from the stop command to the first substep after which the
   TCP speed stays below ``v_stop`` (default 0.02 m/s, the "stopped" speed
   proposed in the plan). Stopping distance is the TCP travel over that time.

Reaction time is reported separately as one control period (0.05 s): the
command issued at tick k takes effect over the substeps of tick k.

Usage: python -m piper_rl.scripts.measure_stopping --out out/stopping.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import HumanAwareEnvConfig


class _RecordingEnv(PiperHumanAwarePickPlaceEnv):
    """Records the TCP position after every physics substep."""

    def _reset_extension_state(self):
        super()._reset_extension_state()
        self.tcp_log: list[np.ndarray] = []
        self.time_log: list[float] = []
        self.hold_q: np.ndarray | None = None

    def _after_physics_substep(self, substep):
        self.tcp_log.append(self.tcp_pos.copy())
        self.time_log.append(float(self.data.time))
        return super()._after_physics_substep(substep)

    def step(self, action):
        if self.hold_q is not None:
            # Scripted stop: hold the measured joint position, bypassing the
            # low-pass filter and the IK. The clamps still apply.
            self._smoothed_q = self.hold_q.copy()
            self._q_cmd = self.hold_q.copy()
        return super().step(action)


DIRECTIONS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
    "+x+y": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    "-x-y": np.array([-1.0, -1.0, 0.0]) / np.sqrt(2),
}


def _speeds(tcp: np.ndarray, t: np.ndarray) -> np.ndarray:
    v = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.diff(t)
    return np.concatenate([[0.0], v])


def measure(direction: str, mode: str, run_steps: int, v_stop: float,
            seed: int = 0, start_action_scale: float = 1.0) -> dict:
    cfg = HumanAwareEnvConfig.from_preset("no_human").eval_variant()
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    env = _RecordingEnv(cfg)
    env.reset(seed=seed)
    d = DIRECTIONS[direction]
    act = np.zeros(5)
    act[:3] = d * start_action_scale
    clamped_run = 0
    for _ in range(run_steps):
        _, _, te, tr, info = env.step(act)
        clamped_run += int(info["safety_clamped"])
        if te or tr:
            break
    n_run = len(env.tcp_log)
    tcp_run = np.array(env.tcp_log)
    t_run = np.array(env.time_log)
    v_run = _speeds(tcp_run, t_run)
    v_before_stop = float(np.mean(v_run[-env._n_substeps:]))   # last control period
    stop_pos = tcp_run[-1].copy()
    t_stop = t_run[-1]

    if mode == "hold-current":
        env.hold_q = env.data.qpos[env.arm_qadr].copy()
    stop_act = np.zeros(5)
    n_stop_steps = 40   # 2 s, long enough for any settling
    for _ in range(n_stop_steps):
        _, _, te, tr, info = env.step(stop_act)
        if te or tr:
            break
    tcp = np.array(env.tcp_log)
    t = np.array(env.time_log)
    v = _speeds(tcp, t)
    after = slice(n_run, None)
    v_after = v[after]
    t_after = t[after] - t_stop
    tcp_after = tcp[after]
    below = v_after < v_stop
    # first index after which speed stays below v_stop for the rest of the log
    stay = np.where(~below)[0]
    idx = int(stay[-1] + 1) if len(stay) else 0
    if idx >= len(v_after):
        idx = len(v_after) - 1
    t_settle = float(t_after[idx])
    dist = float(np.linalg.norm(tcp_after[idx] - stop_pos))
    # travel 0.2 s and 1.0 s after the stop command, for the creep of a zero-action stop
    i02 = int(np.searchsorted(t_after, 0.2)); i10 = int(np.searchsorted(t_after, 1.0))
    d02 = float(np.linalg.norm(tcp_after[min(i02, len(tcp_after)-1)] - stop_pos))
    d10 = float(np.linalg.norm(tcp_after[min(i10, len(tcp_after)-1)] - stop_pos))
    path = float(np.sum(np.linalg.norm(np.diff(tcp_after[: idx + 1], axis=0), axis=1))) if idx > 0 else 0.0
    # time to half speed and 10 % speed, for the deceleration profile
    def time_below(frac):
        thr = frac * v_before_stop
        w = np.where(v_after < thr)[0]
        return float(t_after[w[0]]) if len(w) else float("nan")
    env.close()
    return {
        "direction": direction,
        "stop_mode": mode,
        "steady_speed_m_s": v_before_stop,
        "peak_speed_m_s": float(v_run.max()),
        "clamped_steps_during_run": clamped_run,
        "run_steps": run_steps,
        "stopping_time_s": t_settle,
        "stopping_distance_m": dist,
        "travel_0p2s_after_stop_m": d02,
        "travel_1s_after_stop_m": d10,
        "stopping_path_length_m": path,
        "time_to_half_speed_s": time_below(0.5),
        "time_to_10pct_speed_s": time_below(0.1),
        "v_stop_threshold_m_s": v_stop,
        "start_tcp": tcp_run[0].tolist(),
        "stop_tcp": stop_pos.tolist(),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="out/stopping.json")
    p.add_argument("--run-steps", type=int, default=8,
                   help="control steps of full command before the stop (8 = 0.4 s; "
                        "steady speed is reached after 3 steps, and longer run-ups in "
                        "+x or +z leave the safety envelope)")
    p.add_argument("--v-stop", type=float, default=0.02)
    args = p.parse_args(argv)
    rows = []
    for mode in ("zero-action", "hold-current"):
        for direction in DIRECTIONS:
            rows.append(measure(direction, mode, args.run_steps, args.v_stop))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"{'mode':<13}{'dir':<6}{'v0 m/s':>8}{'t_stop s':>10}{'d_stop mm':>10}{'d@0.2s mm':>10}{'d@1s mm':>9}{'t50 s':>8}{'t10 s':>8}{'clamped':>9}")
    for r in rows:
        print(f"{r['stop_mode']:<13}{r['direction']:<6}{r['steady_speed_m_s']:8.3f}"
              f"{r['stopping_time_s']:10.3f}{r['stopping_distance_m']*1000:10.1f}"
              f"{r['travel_0p2s_after_stop_m']*1000:10.1f}{r['travel_1s_after_stop_m']*1000:9.1f}"
              f"{r['time_to_half_speed_s']:8.3f}{r['time_to_10pct_speed_s']:8.3f}"
              f"{r['clamped_steps_during_run']:9d}")
    for mode in ("zero-action", "hold-current"):
        sel = [r for r in rows if r["stop_mode"] == mode]
        print(f"\n{mode}: steady speed {min(r['steady_speed_m_s'] for r in sel):.3f}-"
              f"{max(r['steady_speed_m_s'] for r in sel):.3f} m/s; stopping time "
              f"{min(r['stopping_time_s'] for r in sel):.3f}-{max(r['stopping_time_s'] for r in sel):.3f} s; "
              f"stopping distance {min(r['stopping_distance_m'] for r in sel)*1000:.0f}-"
              f"{max(r['stopping_distance_m'] for r in sel)*1000:.0f} mm")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
