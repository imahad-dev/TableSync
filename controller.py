"""
TableSync — Deterministic Scripted IK Controller
================================================
Keyed off `contracts.py` interface schemas (ActionType, TargetObject, Subtask,
PlannerOutput, HandoffState). Translates discrete high-level planner subtasks into
strictly grounded Cartesian waypoint trajectories solved via damped least-squares
inverse kinematics (`so101_nexus.kinematics.ee_ik_delta_q`).

Features:
  - Continuous dual-arm command tracking: avoids droop or uncommanded drift by
    holding the counterpart arm's setpoint on every simulation step.
  - Strict enum-keyed approach sequences (`get_approach_sequence`).
  - Dwell-check synchronization during `HANDOFF_RECEIVE` enforcing `HandoffState`
    and `GRIP_DWELL_FRAMES = 10` before giving arm releases.
  - Post-release contact isolation: confirms zero contacts between giving arm and
    transferred object after handoff.
  - Subtask 5 support: places transferred spoon beside the placed plate on the table.
  - Tool Center Point (TCP) tip-offset compensation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import mujoco
import so101_nexus.kinematics as k

from contracts import (
    ArmID,
    ActionType,
    TargetObject,
    Subtask,
    SubtaskStatus,
    PlannerOutput,
    HandoffState,
    HandoffPhase,
    ExecutionResult,
    GRIP_DWELL_FRAMES,
    HANDOFF_POINT_WORLD,
)


@dataclass
class Waypoint:
    """Atomic Cartesian trajectory waypoint for an arm."""
    name: str
    arm: ArmID
    target_pos: np.ndarray
    target_roll: float = -1.57
    gripper_ctrl: float = 1.2    # 1.2 = open (~100 deg), -0.1745 = clamped
    n_interp: int = 80
    hold_steps: int = 40
    enforce_dwell: bool = False


class TableSyncController:
    """Deterministic bimanual execution controller operating on MuJoCo scene."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Arm A IDs
        self.site_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "armA_gripperframe")
        self.act_a = list(range(5))
        self.dof_a = [model.jnt_dofadr[model.actuator(i).trnid[0]] for i in self.act_a]
        self.qpos_a = [model.jnt_qposadr[model.actuator(i).trnid[0]] for i in self.act_a]
        self.grip_act_a = 5

        # Arm B IDs
        self.site_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "armB_gripperframe")
        self.act_b = list(range(6, 11))
        self.dof_b = [model.jnt_dofadr[model.actuator(i).trnid[0]] for i in self.act_b]
        self.qpos_b = [model.jnt_qposadr[model.actuator(i).trnid[0]] for i in self.act_b]
        self.grip_act_b = 11

        # Scene object IDs
        self.plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
        self.plate_rim_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_rim_edge")
        self.spoon_handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "spoon_handle")

        # Gripper fingertip offsets in world coords when roll=-1.57
        self.offset_arm_a = np.array([0.003, 0.007, 0.010])
        self.offset_arm_b = np.array([-0.0031, 0.0064, 0.0103])

        # Rest joint references
        self.rest_q_a = np.array([self.data.qpos[a] for a in self.qpos_a])
        self.rest_q_b = np.array([self.data.qpos[b] for b in self.qpos_b])

        # Continuous setpoint commands to hold both arms stably on every sim step
        self.cmd_q_a = self.rest_q_a.copy()
        self.cmd_q_b = self.rest_q_b.copy()
        self.cmd_grip_a = 1.2
        self.cmd_grip_b = 1.2

        # Optional continuous physics step callback (e.g. video rendering)
        self.step_callback: Optional[Callable[[], None]] = None

        # Gripper occupancy tracking: {ArmID.A: None, ArmID.B: None}
        self._held_object: dict[ArmID, Optional[TargetObject]] = {
            ArmID.A: None,
            ArmID.B: None,
        }

    def step_sim(self, n_steps: int = 1) -> None:
        """Step simulation while maintaining active commands on all 12 actuators."""
        for _ in range(n_steps):
            for i in range(5):
                self.data.ctrl[self.act_a[i]] = self.cmd_q_a[i]
                self.data.ctrl[self.act_b[i]] = self.cmd_q_b[i]
            self.data.ctrl[self.grip_act_a] = self.cmd_grip_a
            self.data.ctrl[self.grip_act_b] = self.cmd_grip_b
            mujoco.mj_step(self.model, self.data)
            if self.step_callback is not None:
                self.step_callback()

    def settle_scene(self, steps: int = 150) -> None:
        """Let bodies settle under gravity and position actuators."""
        self.cmd_q_a = self.rest_q_a.copy()
        self.cmd_q_b = self.rest_q_b.copy()
        self.cmd_grip_a = 1.2
        self.cmd_grip_b = 1.2
        self._held_object = {ArmID.A: None, ArmID.B: None}
        self.step_sim(steps)

    def solve_ik(self, arm: ArmID, target_pos: np.ndarray, target_roll: float = -1.57, max_iter: int = 300) -> tuple[np.ndarray, bool]:
        """Solve inverse kinematics via DLS (`ee_ik_delta_q`) for the given arm."""
        site_id = self.site_a if arm == ArmID.A else self.site_b
        dofs = self.dof_a if arm == ArmID.A else self.dof_b
        qposes = self.qpos_a if arm == ArmID.A else self.qpos_b
        acts = self.act_a if arm == ArmID.A else self.act_b

        d_sim = mujoco.MjData(self.model)
        d_sim.qpos[:] = self.data.qpos[:]
        # Seed arm joints from nominal rest posture to guarantee optimal lifting headroom and branch selection
        nominal_rest = self.rest_q_a if arm == ArmID.A else self.rest_q_b
        for i in range(5):
            d_sim.qpos[qposes[i]] = nominal_rest[i]
        mujoco.mj_forward(self.model, d_sim)
        quat = np.zeros(4)

        for _ in range(max_iter):
            pos = d_sim.site_xpos[site_id].copy()
            if np.linalg.norm(target_pos - pos) < 0.001:
                q_sol = np.array([d_sim.qpos[qposes[i]] for i in range(5)])
                q_sol[4] = target_roll
                return q_sol, True

            mat = d_sim.site_xmat[site_id].copy()
            mujoco.mju_mat2Quat(quat, mat)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, d_sim, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])

            dq = k.ee_ik_delta_q(jac, pos, quat, target_pos, quat, orientation_weight=0.0, damping=0.02)
            for i in range(5):
                lo, hi = self.model.jnt_range[self.model.actuator(acts[i]).trnid[0]]
                d_sim.qpos[qposes[i]] = np.clip(d_sim.qpos[qposes[i]] + dq[i], lo, hi)
            mujoco.mj_forward(self.model, d_sim)

        q_sol = np.array([d_sim.qpos[qposes[i]] for i in range(5)])
        q_sol[4] = target_roll
        return q_sol, False

    def move_arm(self, arm: ArmID, q_target: np.ndarray, grip_val: float, steps: int = 80, hold_steps: int = 40) -> None:
        """Interpolate joint commands smoothly while holding counterpart arm setpoint."""
        if arm == ArmID.A:
            q_start = self.cmd_q_a.copy()
            self.cmd_grip_a = grip_val
            for alpha in np.linspace(0, 1, steps):
                self.cmd_q_a = (1 - alpha) * q_start + alpha * q_target
                self.step_sim(2)
            self.cmd_q_a = q_target.copy()
            self.step_sim(hold_steps * 2)
        else:
            q_start = self.cmd_q_b.copy()
            self.cmd_grip_b = grip_val
            for alpha in np.linspace(0, 1, steps):
                self.cmd_q_b = (1 - alpha) * q_start + alpha * q_target
                self.step_sim(2)
            self.cmd_q_b = q_target.copy()
            self.step_sim(hold_steps * 2)

    def count_contacts(self, entity1: str, entity2: str) -> int:
        """Count active collision contacts between two entity name substrings."""
        count = 0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1 = self.model.geom(con.geom1).name or ""
            g2 = self.model.geom(con.geom2).name or ""
            if (entity1 in g1 and entity2 in g2) or (entity1 in g2 and entity2 in g1):
                count += 1
        return count

    def check_arm_object_contact(self, arm: ArmID, object_name: str) -> bool:
        """Verify geometric contact between arm geoms and target object."""
        arm_str = "armA" if arm == ArmID.A else "armB"
        return self.count_contacts(arm_str, object_name) > 0

    def get_approach_sequence(self, subtask: Subtask) -> list[Waypoint]:
        """Define explicit Cartesian waypoint sequences keyed off ActionType and TargetObject."""
        action = subtask.action
        target_obj = subtask.target_object
        arm = subtask.arm

        waypoints: list[Waypoint] = []

        if action == ActionType.PICK and target_obj == TargetObject.PLATE:
            rim_pos = self.data.geom_xpos[self.plate_rim_id].copy()
            grasp_pt = rim_pos + self.offset_arm_a
            waypoints.append(Waypoint("pre_grasp", arm, grasp_pt + np.array([0, 0, 0.060]), gripper_ctrl=1.2, n_interp=80))
            waypoints.append(Waypoint("grasp", arm, grasp_pt, gripper_ctrl=1.2, n_interp=80))
            waypoints.append(Waypoint("clamp", arm, grasp_pt, gripper_ctrl=-0.1745, n_interp=1, hold_steps=80))
            waypoints.append(Waypoint("lift", arm, grasp_pt + np.array([0, 0, 0.120]), gripper_ctrl=-0.1745, n_interp=100))

        elif action == ActionType.PLACE and target_obj == TargetObject.PLATE:
            # Place plate at clearly separated location (+4cm x, +8cm y, displacement > 7cm)
            place_rim_target = np.array([-0.085, 0.14, 0.225])
            waypoints.append(Waypoint("place_above", arm, place_rim_target + np.array([0, 0, 0.070]), gripper_ctrl=-0.1745, n_interp=80))
            waypoints.append(Waypoint("place_down", arm, place_rim_target, gripper_ctrl=-0.1745, n_interp=80))
            waypoints.append(Waypoint("release", arm, place_rim_target, gripper_ctrl=1.2, n_interp=1, hold_steps=60))
            waypoints.append(Waypoint("retreat", arm, place_rim_target + np.array([0, 0, 0.090]), gripper_ctrl=1.2, n_interp=60))

        elif action == ActionType.PICK and target_obj == TargetObject.SPOON:
            spoon_pos = self.data.geom_xpos[self.spoon_handle_id].copy()
            grasp_pt = spoon_pos + self.offset_arm_b
            waypoints.append(Waypoint("pre_grasp", arm, grasp_pt + np.array([0, 0, 0.060]), gripper_ctrl=1.2, n_interp=80))
            waypoints.append(Waypoint("grasp", arm, grasp_pt, gripper_ctrl=1.2, n_interp=80))
            waypoints.append(Waypoint("clamp", arm, grasp_pt, gripper_ctrl=-0.1745, n_interp=1, hold_steps=80))
            waypoints.append(Waypoint("lift", arm, grasp_pt + np.array([0, 0, 0.120]), gripper_ctrl=-0.1745, n_interp=100))

        elif action == ActionType.HANDOFF_EXTEND and target_obj == TargetObject.HANDOFF_POINT:
            ho_target = np.array([0.04, 0.15, 0.29])
            waypoints.append(Waypoint("handoff_extend", arm, ho_target, gripper_ctrl=-0.1745, n_interp=100))

        elif action == ActionType.HANDOFF_RECEIVE and target_obj == TargetObject.HANDOFF_POINT:
            spoon_held = self.data.geom_xpos[self.spoon_handle_id].copy()
            grasp_pt = spoon_held + self.offset_arm_a
            # Staging waypoint prevents arm linkages from colliding during approach
            waypoints.append(Waypoint("ho_stage", arm, grasp_pt + np.array([-0.05, 0.0, 0.04]), gripper_ctrl=1.2, n_interp=80))
            waypoints.append(Waypoint("ho_grasp", arm, grasp_pt, gripper_ctrl=1.2, n_interp=60))
            waypoints.append(Waypoint("ho_clamp_dwell", arm, grasp_pt, gripper_ctrl=-0.1745, n_interp=1, hold_steps=40, enforce_dwell=True))
            waypoints.append(Waypoint("ho_lift", arm, grasp_pt + np.array([0, 0, 0.080]), gripper_ctrl=-0.1745, n_interp=80))

        elif action == ActionType.PLACE and target_obj == TargetObject.SPOON:
            # Place spoon beside the placed plate on the table at [0.015, 0.130, 0.225]
            spoon_place_target = np.array([0.015, 0.130, 0.225])
            waypoints.append(Waypoint("spoon_place_above", arm, spoon_place_target + np.array([0, 0, 0.070]), gripper_ctrl=-0.1745, n_interp=80))
            waypoints.append(Waypoint("spoon_place_down", arm, spoon_place_target, gripper_ctrl=-0.1745, n_interp=80))
            waypoints.append(Waypoint("spoon_release", arm, spoon_place_target, gripper_ctrl=1.2, n_interp=1, hold_steps=60))
            waypoints.append(Waypoint("spoon_retreat", arm, spoon_place_target + np.array([0, 0, 0.090]), gripper_ctrl=1.2, n_interp=60))

        elif action == ActionType.RETREAT:
            site_id = self.site_a if arm == ArmID.A else self.site_b
            current_pos = self.data.site_xpos[site_id].copy()
            waypoints.append(Waypoint("retreat_up", arm, current_pos + np.array([0, 0, 0.080]), gripper_ctrl=1.2, n_interp=60))

        return waypoints

    def execute_subtask(self, subtask: Subtask, handoff_state: Optional[HandoffState] = None) -> ExecutionResult:
        """Execute a subtask waypoint-by-waypoint with full physical validation."""
        waypoints = self.get_approach_sequence(subtask)
        if not waypoints:
            return ExecutionResult(step_index=subtask.step_index, status=SubtaskStatus.FAILED, error="No waypoints generated")

        # Safety Interlock: if any waypoint commands the arm's gripper to open,
        # ensure the arm is not currently holding a different object without an explicit PLACE step.
        will_open_gripper = any(wp.gripper_ctrl > 0.0 for wp in waypoints)
        currently_held = self._held_object[subtask.arm]
        if will_open_gripper and currently_held is not None:
            if not (subtask.action == ActionType.PLACE and subtask.target_object == currently_held):
                subtask.status = SubtaskStatus.FAILED
                target_desc = subtask.target_object.value if subtask.target_object else "none"
                return ExecutionResult(
                    step_index=subtask.step_index,
                    status=SubtaskStatus.FAILED,
                    error=(
                        f"Safety Interlock: Arm {subtask.arm.value} is currently holding '{currently_held.value}' "
                        f"and cannot open gripper for '{target_desc}' ({subtask.action.value}) without an explicit PLACE step."
                    ),
                )

        subtask.status = SubtaskStatus.IN_PROGRESS

        for wp in waypoints:
            target_pos = wp.target_pos.copy()
            q_target, ok = self.solve_ik(wp.arm, target_pos, target_roll=wp.target_roll)
            if not ok:
                subtask.status = SubtaskStatus.FAILED
                return ExecutionResult(
                    step_index=subtask.step_index,
                    status=SubtaskStatus.FAILED,
                    error=f"IK solve failed for waypoint '{wp.name}' at {target_pos.tolist()}",
                )

            self.move_arm(wp.arm, q_target, grip_val=wp.gripper_ctrl, steps=wp.n_interp, hold_steps=wp.hold_steps)

            if wp.enforce_dwell and handoff_state is not None:
                handoff_state.phase = HandoffPhase.AWAITING_GRIP
                confirmed = False
                max_consec = 0
                curr_consec = 0
                total_contacts = 0
                for _ in range(60):
                    has_contact = self.check_arm_object_contact(wp.arm, "spoon")
                    if has_contact:
                        total_contacts += 1
                        curr_consec += 1
                        max_consec = max(max_consec, curr_consec)
                    else:
                        curr_consec = 0
                    handoff_state.advance_grip_check(1.0 if has_contact else 0.0)
                    self.step_sim(2)
                    if handoff_state.phase == HandoffPhase.GRIP_CONFIRMED:
                        confirmed = True
                        break

                if not confirmed:
                    subtask.status = SubtaskStatus.FAILED
                    arm_site = self.site_a if wp.arm == ArmID.A else self.site_b
                    ee_pos = self.data.site_xpos[arm_site].copy()
                    sp_pos = self.data.geom_xpos[self.spoon_handle_id].copy()
                    dist_mm = float(np.linalg.norm(ee_pos - sp_pos) * 1000.0)
                    return ExecutionResult(
                        step_index=subtask.step_index,
                        status=SubtaskStatus.FAILED,
                        error=f"Handoff dwell check failed: max_consec={max_consec}/{GRIP_DWELL_FRAMES}, total_contacts={total_contacts}/60, ee_dist={dist_mm:.1f}mm",
                    )

                # Grip confirmed on Arm A -> Arm B releases spoon
                handoff_state.phase = HandoffPhase.RELEASING
                self.cmd_grip_b = 1.2
                self.step_sim(60)

                # Arm B retreats to rest pose
                handoff_state.phase = HandoffPhase.RETREATING
                self.move_arm(ArmID.B, self.rest_q_b, grip_val=1.2, steps=80)

                # CRITICAL VERIFICATION: Confirm zero contacts between Arm B and spoon after release
                b_contacts = self.count_contacts("armB", "spoon")
                if b_contacts > 0:
                    subtask.status = SubtaskStatus.FAILED
                    return ExecutionResult(
                        step_index=subtask.step_index,
                        status=SubtaskStatus.FAILED,
                        error=f"Arm B failed to release spoon: {b_contacts} remaining contacts",
                    )

        # If subtask completes and was a place action, return arm to rest
        if subtask.action == ActionType.PLACE:
            rest_q = self.rest_q_a if subtask.arm == ArmID.A else self.rest_q_b
            self.move_arm(subtask.arm, rest_q, grip_val=1.2, steps=80)

            # Confirm zero contacts with placed object after retreat
            target_str = "plate" if subtask.target_object == TargetObject.PLATE else "spoon"
            remaining = self.count_contacts("armA" if subtask.arm == ArmID.A else "armB", target_str)
            if remaining > 0:
                subtask.status = SubtaskStatus.FAILED
                return ExecutionResult(
                    step_index=subtask.step_index,
                    status=SubtaskStatus.FAILED,
                    error=f"Arm still in contact with {target_str} after retreat: {remaining} contacts",
                )

        # Update occupancy tracking upon successful subtask completion
        if subtask.action == ActionType.PICK:
            target_str = "plate" if subtask.target_object == TargetObject.PLATE else "spoon"
            if self.check_arm_object_contact(subtask.arm, target_str):
                self._held_object[subtask.arm] = subtask.target_object
        elif subtask.action == ActionType.PLACE:
            self._held_object[subtask.arm] = None
        elif subtask.action == ActionType.HANDOFF_RECEIVE:
            self._held_object[ArmID.B] = None
            if not self.check_arm_object_contact(ArmID.A, "spoon"):
                self._held_object[ArmID.A] = None
                subtask.status = SubtaskStatus.FAILED
                return ExecutionResult(
                    step_index=subtask.step_index,
                    status=SubtaskStatus.FAILED,
                    error="Arm A failed to maintain contact with spoon after lift",
                )
            self._held_object[ArmID.A] = TargetObject.SPOON

        subtask.status = SubtaskStatus.COMPLETE
        site_id = self.site_a if subtask.arm == ArmID.A else self.site_b
        return ExecutionResult(
            step_index=subtask.step_index,
            status=SubtaskStatus.COMPLETE,
            final_ee_pose=self.data.site_xpos[site_id].tolist(),
            grasp_state=1.0,
        )

    def run_plan(self, plan: PlannerOutput) -> list[ExecutionResult]:
        """Execute all subtasks in a PlannerOutput sequentially."""
        self.settle_scene(150)
        results: list[ExecutionResult] = []
        handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)

        for st in plan.subtasks:
            res = self.execute_subtask(st, handoff_state=handoff_state)
            results.append(res)
            if res.status != SubtaskStatus.COMPLETE:
                break

        return results
