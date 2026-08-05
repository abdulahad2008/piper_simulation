"""Validate the human-aware MuJoCo scene, motion, API, safety, and rendering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from piper_rl.human_aware_env import PiperHumanAwarePickPlaceEnv
from piper_rl.human_config import (HUMAN_MODEL_PATH, TRAJECTORY_TYPES,
                                   HumanAwareEnvConfig)


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'  ok  ' if condition else ' FAIL '}] {name}" +
          (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _orange_pixels(image: np.ndarray) -> int:
    rgb = image.astype(np.int16)
    return int(np.count_nonzero(
        (rgb[..., 0] > 130) & (rgb[..., 0] > rgb[..., 1] + 35)
        & (rgb[..., 1] > rgb[..., 2] + 10)))


def validate_trajectories(env: PiperHumanAwarePickPlaceEnv) -> None:
    print("\n--- trajectory checks")
    max_seen = 0.0
    continuity = True
    collision_free = True
    all_names = []
    dt = env.model.opt.timestep
    for index, kind in enumerate(TRAJECTORY_TYPES):
        env.reset(seed=100 + index, options={"human_trajectory_type": kind})
        all_names.append(env.human_trajectory.trajectory_type)
        collision_free &= not env._human_collision and not env._robot_human_contacts()[0]
        times = np.arange(0.0, env.human_trajectory.end_time + dt, dt)
        previous = env.human_trajectory.state(times[0])
        for t in times[1:]:
            state = env.human_trajectory.state(t)
            jump = np.linalg.norm(state.hand_position - previous.hand_position)
            continuity &= jump <= env.human_cfg.human_motion.max_hand_speed * dt * 1.02 + 1e-7
            max_seen = max(max_seen, float(np.linalg.norm(state.hand_velocity)))
            previous = state
    check("all trajectory types sampled", tuple(all_names) == TRAJECTORY_TYPES,
          ", ".join(all_names))
    check("minimum-jerk motion is continuous at physics resolution", continuity)
    check("hand speed respects configured limit",
          max_seen <= env.human_cfg.human_motion.max_hand_speed + 1e-6,
          f"peak {max_seen:.3f} m/s")
    check("all initial human poses are robot-contact-free", collision_free)


def validate_reproducibility(env: PiperHumanAwarePickPlaceEnv) -> None:
    print("\n--- reproducibility")
    def rollout(seed: int):
        env.reset(seed=seed)
        traj = env.human_trajectory
        sample_times = np.linspace(0, min(traj.end_time, 8.0), 100)
        positions = np.array([traj.state(t).hand_position for t in sample_times])
        return traj.trajectory_type, traj.appearance_time, positions

    a = rollout(7123)
    b = rollout(7123)
    c = rollout(7124)
    check("same seed reproduces the human trajectory",
          a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2]))
    check("different seeds vary the human trajectory",
          a[0] != c[0] or a[1] != c[1] or not np.array_equal(a[2], c[2]))


def validate_rollout(env: PiperHumanAwarePickPlaceEnv) -> None:
    print("\n--- Gymnasium and random rollout")
    try:
        check_env(env, warn=True)
        check("SB3 Gymnasium environment checker", True,
              f"obs={env.observation_space.shape}, act={env.action_space.shape}")
    except Exception as exc:
        check("SB3 Gymnasium environment checker", False, str(exc)[:160])
    obs, _ = env.reset(seed=9001)
    rng = np.random.default_rng(12345)
    finite = bool(np.all(np.isfinite(obs)))
    steps = 0
    while steps < 100:
        action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(reward))
        steps += 1
        if terminated or truncated:
            obs, _ = env.reset()
    check("100 random actions complete", steps == 100)
    check("observations and rewards remain finite", finite)


def make_trajectory_video(env: PiperHumanAwarePickPlaceEnv, output: Path) -> None:
    print("\n--- rendering")
    import imageio.v2 as imageio
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover
        Image = ImageDraw = None

    frames = []
    manifest = []
    visible = {"forehead": False, "overview": False}
    cfg = env.human_cfg.human_motion
    old_appearance, old_speed = cfg.appearance_time_range, cfg.speed_range
    old_pause = cfg.pause_duration_range
    cfg.appearance_time_range = (0.0, 0.0)
    cfg.speed_range = (cfg.max_hand_speed, cfg.max_hand_speed)
    cfg.pause_duration_range = (0.5, 0.5)
    try:
        for index, kind in enumerate(TRAJECTORY_TYPES):
            env.reset(seed=300 + index, options={"human_trajectory_type": kind})
            traj = env.human_trajectory
            duration = min(traj.end_time + 0.25, 8.0)
            manifest.append(f"{kind}: {duration:.2f} s")
            for t in np.arange(0.0, duration, 1.0 / env.cfg.control_hz):
                env._set_human_state(traj.state(t))
                mujoco.mj_forward(env.model, env.data)
                frame = env.render("overview")
                visible["overview"] |= _orange_pixels(frame) > 20
                if not visible["forehead"]:
                    visible["forehead"] |= _orange_pixels(env.render("forehead")) > 10
                if Image is not None:
                    pil = Image.fromarray(frame)
                    ImageDraw.Draw(pil).rectangle((8, 8, 260, 34), fill=(0, 0, 0))
                    ImageDraw.Draw(pil).text((14, 13), kind, fill=(255, 255, 255))
                    frame = np.asarray(pil)
                frames.append(frame)
    finally:
        cfg.appearance_time_range, cfg.speed_range = old_appearance, old_speed
        cfg.pause_duration_range = old_pause
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=int(env.cfg.control_hz), quality=8)
    print("trajectory manifest:")
    for item in manifest:
        print(f"  {item}")
    check("human is visible from forehead camera", visible["forehead"])
    check("human is visible from overview camera", visible["overview"])
    check("trajectory video generated", output.exists() and output.stat().st_size > 0,
          str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="out/human_trajectory_validation.mp4")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args(argv)

    print("PIPER human-aware environment validation")
    print(f"model: {HUMAN_MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(str(HUMAN_MODEL_PATH))
        check("human-aware MJCF compiles", True,
              f"nq={model.nq}, nmocap={model.nmocap}, ngeom={model.ngeom}")
        for name in ("human_upper_arm", "human_forearm", "human_hand"):
            model.body(name)
        for name in ("human_upper_arm_geom", "human_forearm_geom", "human_hand_geom"):
            model.geom(name)
        check("named human bodies and geoms exist", True)
    except Exception as exc:
        check("human-aware MJCF compiles", False, str(exc))
        return 1

    cfg = HumanAwareEnvConfig.from_preset("randomized")
    cfg.curriculum = False
    cfg.domain_rand.enabled = False
    cfg.noise.enabled = False
    env = PiperHumanAwarePickPlaceEnv(cfg)
    validate_trajectories(env)
    validate_reproducibility(env)
    validate_rollout(env)
    if not args.skip_video:
        make_trajectory_video(env, Path(args.video))
    env.close()

    if FAILURES:
        print("\nRESULT: FAILED: " + "; ".join(FAILURES))
        return 1
    print("\nRESULT: all human-aware validation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
