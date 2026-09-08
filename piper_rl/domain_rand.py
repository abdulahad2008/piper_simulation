"""Domain randomisation for the PIPER pick-and-place scene.

Everything here writes into the *compiled* :class:`mujoco.MjModel`, which is
legal for the fields we touch (geom size/pos/rgba/friction, body mass/inertia,
actuator gains, joint damping, light pose). Values are re-derived from a pristine
copy of the model every reset, so randomisation never compounds.

What is randomised, and why it matters for sim-to-real
------------------------------------------------------
============================  =========================================
Quantity                       Real-world source of error
============================  =========================================
object pose / yaw              perception error, human placement
object size                    you will not always pick the same cup
object mass                    empty vs. full cup: 30 g .. 120 g
object + table friction        surface finish, dust, humidity
actuator kp / kv               servo tuning, temperature, gearbox wear
joint damping / friction       harmonic-drive stiction, cable drag
light position / intensity     time of day, room lighting
object + table colour          different cups, different tables
============================  =========================================
"""

from __future__ import annotations

import numpy as np
import mujoco

from .config import DomainRandConfig


class DomainRandomizer:
    """Applies :class:`DomainRandConfig` to a compiled model, reversibly."""

    def __init__(self, model: mujoco.MjModel, cfg: DomainRandConfig):
        self.m = model
        self.cfg = cfg

        # Named handles ------------------------------------------------- #
        self.obj_body = model.body("obj").id
        self.obj_geom = model.geom("obj_geom").id
        self.table_geom = model.geom("table").id
        self.pad_dst_geom = model.geom("pad_dst").id
        self.pad_src_geom = model.geom("pad_src").id
        self.target_site = model.site("target").id
        self.light_id = model.light("key_light").id
        self.obj_matid = int(model.geom_matid[self.obj_geom])
        self.table_matid = int(model.geom_matid[self.table_geom])

        # Pristine copies of everything we mutate ------------------------ #
        self._nominal = dict(
            geom_size=model.geom_size.copy(),
            geom_pos=model.geom_pos.copy(),
            geom_rgba=model.geom_rgba.copy(),
            geom_friction=model.geom_friction.copy(),
            body_mass=model.body_mass.copy(),
            body_inertia=model.body_inertia.copy(),
            mat_rgba=model.mat_rgba.copy(),
            actuator_gainprm=model.actuator_gainprm.copy(),
            actuator_biasprm=model.actuator_biasprm.copy(),
            dof_damping=model.dof_damping.copy(),
            dof_frictionloss=model.dof_frictionloss.copy(),
            light_pos=model.light_pos.copy(),
            light_diffuse=model.light_diffuse.copy(),
            site_pos=model.site_pos.copy(),
        )

        # Arm DoF indices (exclude the free joint of the object).
        self.arm_dofs = np.array(
            [model.jnt_dofadr[model.joint(f"joint{i}").id] for i in range(1, 9)])

    # ------------------------------------------------------------------ #
    def restore(self) -> None:
        """Reset every randomised field to its value from the XML."""
        for k, v in self._nominal.items():
            getattr(self.m, k)[:] = v

    # ------------------------------------------------------------------ #
    def randomize(self, rng: np.random.Generator) -> dict:
        """Randomise the model in place. Returns the sampled object/target poses.

        Always call :meth:`restore` first (the env does).
        """
        c = self.cfg
        info = {}

        # ---------------- object pose (returned, applied to qpos by env) -- #
        r = rng.uniform(*c.obj_radius)
        th = rng.uniform(*c.obj_azimuth)
        # Oversample the hard corner (outer radius + far azimuth). No-op at the
        # default hard_corner_frac = 0.0.
        if c.hard_corner_frac and rng.random() < c.hard_corner_frac:
            r0, r1 = c.obj_radius
            t0, t1 = c.obj_azimuth
            r = rng.uniform(r0 + c.hard_radius_lo * (r1 - r0), r1)
            th = rng.uniform(t0, t0 + c.hard_azimuth_hi * (t1 - t0))
        obj_xy = np.array([r * np.cos(th), r * np.sin(th)])
        info["obj_r"] = float(r)
        info["obj_th"] = float(th)
        obj_yaw = rng.uniform(*c.obj_yaw)
        info["obj_xy"] = obj_xy
        info["obj_yaw"] = obj_yaw

        rt = rng.uniform(*c.target_radius)
        tht = rng.uniform(*c.target_azimuth)
        tgt_xy = np.array([rt * np.cos(tht), rt * np.sin(tht)])
        info["target_xy"] = tgt_xy

        if not c.enabled:
            info["obj_half_h"] = float(self.m.geom_size[self.obj_geom, 1])
            info["obj_half_w"] = float(self.m.geom_size[self.obj_geom, 0])
            self._place_target(tgt_xy, info["obj_half_h"])
            self._place_src_pad(obj_xy)
            return info

        # ---------------- object geometry & dynamics --------------------- #
        base_size = self._nominal["geom_size"][self.obj_geom].copy()
        half_w = base_size[0] * rng.uniform(*c.obj_radius_scale)
        half_h = base_size[1] * rng.uniform(*c.obj_height_scale)
        # Hard caps, both derived from the gripper geometry:
        #   width  -- inner opening is 2 * ctrl_max = 0.07 m; leave >= 8 mm of
        #             total clearance or the grasp becomes impossible.
        #   height -- link6's collision capsule (the palm) sits ~46 mm above the
        #             TCP; a taller object is hit by the palm before the fingers
        #             can close around it.
        half_w = float(min(half_w, 0.031))
        half_h = float(min(half_h, 0.042))
        self.m.geom_size[self.obj_geom, 0] = half_w
        self.m.geom_size[self.obj_geom, 1] = half_h
        info["obj_half_w"] = half_w
        info["obj_half_h"] = half_h

        mass = rng.uniform(*c.obj_mass)
        self.m.body_mass[self.obj_body] = mass
        # Cylinder inertia about its own principal axes.
        ixx = mass * (3 * half_w ** 2 + (2 * half_h) ** 2) / 12.0
        izz = 0.5 * mass * half_w ** 2
        self.m.body_inertia[self.obj_body] = np.array([ixx, ixx, izz])
        info["obj_mass"] = mass

        mu = rng.uniform(*c.obj_friction)
        self.m.geom_friction[self.obj_geom, 0] = mu
        info["obj_friction"] = mu

        # ---------------- visual appearance ------------------------------ #
        if self.obj_matid >= 0:
            base = self._nominal["mat_rgba"][self.obj_matid].copy()
            base[:3] = np.clip(
                base[:3] + rng.uniform(-c.obj_rgba_jitter, c.obj_rgba_jitter, 3),
                0.05, 1.0)
            self.m.mat_rgba[self.obj_matid] = base
        if self.table_matid >= 0:
            base = self._nominal["mat_rgba"][self.table_matid].copy()
            base[:3] = np.clip(
                base[:3] + rng.uniform(-c.table_rgba_jitter, c.table_rgba_jitter, 3),
                0.05, 1.0)
            self.m.mat_rgba[self.table_matid] = base

        # ---------------- table friction --------------------------------- #
        table_mu = rng.uniform(*c.table_friction)
        self.m.geom_friction[self.table_geom, 0] = table_mu
        info["table_friction"] = float(table_mu)

        # ---------------- lighting --------------------------------------- #
        lp = self._nominal["light_pos"][self.light_id].copy()
        lp[:2] += rng.uniform(-c.light_pos_jitter, c.light_pos_jitter, 2)
        lp[2] += rng.uniform(-0.2, 0.2)
        self.m.light_pos[self.light_id] = lp
        self.m.light_diffuse[self.light_id] = rng.uniform(*c.light_diffuse)

        # ---------------- actuator + joint dynamics ---------------------- #
        # MuJoCo position actuator: gainprm[0] = kp, biasprm[1] = -kp,
        # biasprm[2] = -kv.
        kp_scales, kv_scales = [], []
        for a in range(self.m.nu):
            kp_s = rng.uniform(*c.actuator_kp_scale)
            kv_s = rng.uniform(*c.actuator_kv_scale)
            kp_scales.append(kp_s)
            kv_scales.append(kv_s)
            self.m.actuator_gainprm[a, 0] = \
                self._nominal["actuator_gainprm"][a, 0] * kp_s
            self.m.actuator_biasprm[a, 1] = \
                self._nominal["actuator_biasprm"][a, 1] * kp_s
            self.m.actuator_biasprm[a, 2] = \
                self._nominal["actuator_biasprm"][a, 2] * kv_s

        damping_scales, frictionloss_scales = [], []
        for dof in self.arm_dofs:
            d_s = rng.uniform(*c.joint_damping_scale)
            f_s = rng.uniform(*c.joint_frictionloss_scale)
            damping_scales.append(d_s)
            frictionloss_scales.append(f_s)
            self.m.dof_damping[dof] = self._nominal["dof_damping"][dof] * d_s
            self.m.dof_frictionloss[dof] = (
                self._nominal["dof_frictionloss"][dof] * f_s)

        info["actuator_kp_scale"] = np.asarray(kp_scales, dtype=float)
        info["actuator_kv_scale"] = np.asarray(kv_scales, dtype=float)
        info["joint_damping_scale"] = np.asarray(damping_scales, dtype=float)
        info["joint_frictionloss_scale"] = np.asarray(frictionloss_scales,
                                                      dtype=float)
        # NOTE: the released MJCF sets no joint damping, so dof_damping is 0 and
        # `joint_damping_scale` multiplies zero -- it is a no-op. The scales are
        # logged anyway so the log is honest about what was *drawn* vs applied.
        info["joint_damping_is_noop"] = bool(
            np.allclose(self._nominal["dof_damping"][self.arm_dofs], 0.0))

        # ---------------- markers ---------------------------------------- #
        self._place_target(tgt_xy, half_h)
        self._place_src_pad(obj_xy)
        return info

    # ------------------------------------------------------------------ #
    def _place_target(self, xy, obj_half_h: float) -> None:
        """Move the destination site + green pad to ``xy``."""
        table_top = 0.20
        self.m.site_pos[self.target_site] = [xy[0], xy[1], table_top + obj_half_h]
        self.m.geom_pos[self.pad_dst_geom] = [xy[0], xy[1], table_top + 0.0015]

    def _place_src_pad(self, xy) -> None:
        self.m.geom_pos[self.pad_src_geom] = [xy[0], xy[1], 0.20 + 0.0015]
