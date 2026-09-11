"""
TableSync Robustness Evaluation Harness
=======================================
Evaluates the complete 6-subtask bimanual pipeline across 10 distinct seeds.
Perturbation categories per seed:
  1. Object Initial (x, y) Positions:
     - Plate starting xy perturbed within reachable workspace
     - Spoon starting xy perturbed within reachable workspace
  2. Table Lighting:
     - Perturbed light position (dx, dy, dz)
     - Perturbed light intensity/diffuse color
  3. Geom Friction Coefficients:
     - Perturbed sliding and torsional friction on plate and spoon geoms

For each seed, logs:
  - Subtask completion status (0 to 5)
  - Explicit numeric metric checks against identical thresholds
  - Root cause and measured metric if any subtask fails
"""

import mujoco
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contracts import (
    ArmID, ActionType, TargetObject, Subtask, SubtaskStatus,
    PlannerOutput, HandoffState
)
from controller import TableSyncController


@dataclass
class SeedResult:
    seed: int
    plate_init_xy: tuple[float, float]
    spoon_init_xy: tuple[float, float]
    friction_mult: float
    light_intensity: float
    passed_steps: list[int]
    failed_step: Optional[int]
    failure_reason: Optional[str]
    success: bool


def run_single_seed(seed: int, xml_path: Path) -> SeedResult:
    rng = np.random.RandomState(seed)

    # 1. Perturbations
    # Bounded object positions within confirmed reachable kinematic envelope (+/- 5 mm)
    delta_plate_x = rng.uniform(-0.005, 0.005)  # +/- 5 mm
    delta_plate_y = rng.uniform(-0.005, 0.005)  # +/- 5 mm
    delta_spoon_x = rng.uniform(-0.005, 0.005)  # +/- 5 mm
    delta_spoon_y = rng.uniform(-0.005, 0.005)  # +/- 5 mm

    # Friction perturbation: +/- 25% scale
    friction_mult = rng.uniform(0.75, 1.25)

    # Lighting perturbation: position +/- 10cm, intensity 0.6x to 1.3x
    delta_light = rng.uniform(-0.10, 0.10, size=3)
    light_mult = rng.uniform(0.60, 1.30)

    # Load fresh model
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    # Apply lighting perturbation
    light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "scene_light")
    if light_id >= 0:
        model.light_pos[light_id] += delta_light
        model.light_diffuse[light_id] = np.clip(model.light_diffuse[light_id] * light_mult, 0.1, 1.0)

    # Apply friction perturbations
    for g_name in ["plate_dish", "plate_rim_edge", "spoon_base", "spoon_handle"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g_name)
        if gid >= 0:
            model.geom_friction[gid] *= friction_mult

    data = mujoco.MjData(model)

    # Apply object position perturbations to free joints
    plate_jnt = model.joint("plate_free")
    plate_qadr = plate_jnt.qposadr[0]
    spoon_jnt = model.joint("spoon_free")
    spoon_qadr = spoon_jnt.qposadr[0]

    data.qpos[plate_qadr] = -0.08 + delta_plate_x
    data.qpos[plate_qadr + 1] = 0.06 + delta_plate_y
    data.qpos[plate_qadr + 2] = 0.203

    data.qpos[spoon_qadr] = 0.08 + delta_spoon_x
    data.qpos[spoon_qadr + 1] = 0.06 + delta_spoon_y
    data.qpos[spoon_qadr + 2] = 0.205

    controller = TableSyncController(model, data)
    controller.settle_scene(150)

    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")

    init_plate_pos = data.xpos[plate_id].copy()
    init_spoon_pos = data.xpos[spoon_id].copy()

    plan = PlannerOutput(
        raw_instruction="Execute full 6-subtask sequence",
        scene_summary="Perturbed bimanual workspace",
        subtasks=[
            Subtask(step_index=0, arm=ArmID.A, action=ActionType.PICK, target_object=TargetObject.PLATE),
            Subtask(step_index=1, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.PLATE, depends_on=0),
            Subtask(step_index=2, arm=ArmID.B, action=ActionType.PICK, target_object=TargetObject.SPOON),
            Subtask(step_index=3, arm=ArmID.B, action=ActionType.HANDOFF_EXTEND, target_object=TargetObject.HANDOFF_POINT, depends_on=2),
            Subtask(step_index=4, arm=ArmID.A, action=ActionType.HANDOFF_RECEIVE, target_object=TargetObject.HANDOFF_POINT, depends_on=3),
            Subtask(step_index=5, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.SPOON, depends_on=4),
        ]
    )

    handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)
    passed_steps: list[int] = []

    for subtask in plan.subtasks:
        result = controller.execute_subtask(subtask, handoff_state=handoff_state)
        if result.status != SubtaskStatus.COMPLETE:
            return SeedResult(
                seed=seed,
                plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                friction_mult=float(friction_mult),
                light_intensity=float(light_mult),
                passed_steps=passed_steps,
                failed_step=subtask.step_index,
                failure_reason=f"Controller status={result.status.value}: {result.error}",
                success=False
            )

        # Numerical Metric Verifications per Step
        if subtask.step_index == 0:
            plate_pos = data.xpos[plate_id].copy()
            gain_0 = plate_pos[2] - init_plate_pos[2]
            if gain_0 < 0.045:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=0,
                    failure_reason=f"Plate height gain {gain_0:.4f}m < 0.045m",
                    success=False
                )
            passed_steps.append(0)

        elif subtask.step_index == 1:
            plate_pos = data.xpos[plate_id].copy()
            disp_1 = np.linalg.norm(plate_pos[:2] - init_plate_pos[:2])
            a_plate_contacts = controller.count_contacts("armA", "plate")
            if disp_1 < 0.050:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=1,
                    failure_reason=f"Plate displacement {disp_1:.4f}m < 0.050m",
                    success=False
                )
            if plate_pos[2] > 0.205:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=1,
                    failure_reason=f"Plate settled z {plate_pos[2]:.4f}m > 0.205m",
                    success=False
                )
            if a_plate_contacts > 0:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=1,
                    failure_reason=f"Arm A contacts with plate = {a_plate_contacts} > 0",
                    success=False
                )
            passed_steps.append(1)

        elif subtask.step_index == 2:
            spoon_pos = data.xpos[spoon_id].copy()
            gain_2 = spoon_pos[2] - init_spoon_pos[2]
            if gain_2 < 0.050:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=2,
                    failure_reason=f"Spoon height gain {gain_2:.4f}m < 0.050m",
                    success=False
                )
            passed_steps.append(2)

        elif subtask.step_index == 3:
            ee_b = data.site_xpos[controller.site_b].copy()
            target_ho_b = np.array([0.04, 0.15, 0.29])
            err_3 = np.linalg.norm(ee_b - target_ho_b)
            if err_3 > 0.020:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=3,
                    failure_reason=f"Arm B EE error {err_3*1000:.2f}mm > 20.0mm",
                    success=False
                )
            passed_steps.append(3)

        elif subtask.step_index == 4:
            b_spoon_contacts = controller.count_contacts("armB", "spoon")
            spoon_pos = data.xpos[spoon_id].copy()
            gain_4 = spoon_pos[2] - init_spoon_pos[2]
            if handoff_state.grip_confirm_frames < 10:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=4,
                    failure_reason=f"Dwell frames {handoff_state.grip_confirm_frames} < 10",
                    success=False
                )
            if b_spoon_contacts > 0:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=4,
                    failure_reason=f"Arm B contacts after release = {b_spoon_contacts} > 0",
                    success=False
                )
            if gain_4 < 0.050:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=4,
                    failure_reason=f"Transferred spoon gain {gain_4:.4f}m < 0.050m",
                    success=False
                )
            passed_steps.append(4)

        elif subtask.step_index == 5:
            final_plate_pos = data.xpos[plate_id].copy()
            final_spoon_pos = data.xpos[spoon_id].copy()
            a_spoon_contacts = controller.count_contacts("armA", "spoon")
            spoon_plate_contacts = controller.count_contacts("spoon", "plate")
            rel_vector = final_spoon_pos - final_plate_pos
            rel_distance_xy = np.linalg.norm(rel_vector[:2])

            if final_spoon_pos[2] > 0.220:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=5,
                    failure_reason=f"Spoon elevation z {final_spoon_pos[2]:.4f}m > 0.220m",
                    success=False
                )
            if a_spoon_contacts > 0:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=5,
                    failure_reason=f"Arm A contacts after release = {a_spoon_contacts} > 0",
                    success=False
                )
            if spoon_plate_contacts > 0:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=5,
                    failure_reason=f"Spoon contacts with plate = {spoon_plate_contacts} > 0",
                    success=False
                )
            if rel_distance_xy > 0.100:
                return SeedResult(
                    seed=seed, plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
                    spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
                    friction_mult=float(friction_mult), light_intensity=float(light_mult),
                    passed_steps=passed_steps, failed_step=5,
                    failure_reason=f"Spoon-plate distance {rel_distance_xy:.4f}m > 0.100m",
                    success=False
                )
            passed_steps.append(5)

    return SeedResult(
        seed=seed,
        plate_init_xy=(init_plate_pos[0], init_plate_pos[1]),
        spoon_init_xy=(init_spoon_pos[0], init_spoon_pos[1]),
        friction_mult=float(friction_mult),
        light_intensity=float(light_mult),
        passed_steps=passed_steps,
        failed_step=None,
        failure_reason=None,
        success=True
    )


def run_robustness_harness(num_seeds: int = 10):
    xml_path = Path(__file__).resolve().parent.parent / "scene" / "tablesync_scene.xml"
    print("=========================================================================================")
    print(f"TABLESYNC ROBUSTNESS HARNESS ({num_seeds} SEEDS)")
    print("Perturbations: Object XY (reachable bounds), Lighting (pos/intensity), Friction (plate/spoon)")
    print("=========================================================================================")

    results: list[SeedResult] = []
    for s in range(num_seeds):
        seed = 42 + s * 7  # deterministic seed sequence
        print(f"Running Seed {s+1}/{num_seeds} (seed={seed})...", end=" ", flush=True)
        res = run_single_seed(seed, xml_path)
        status_str = "PASS [6/6]" if res.success else f"FAIL (Step {res.failed_step}: {res.failure_reason})"
        print(status_str)
        results.append(res)

    print("\n=========================================================================================")
    print("ROBUSTNESS EVALUATION REPORT (10 SEEDS)")
    print("=========================================================================================")
    print(f"{'Seed':<6} | {'Plate Init XY':<22} | {'Spoon Init XY':<22} | {'Frict':<6} | {'Light':<6} | {'Steps Passed':<14} | {'Status':<6} | {'Failure Mode'}")
    print("-" * 115)

    pass_count = sum(1 for r in results if r.success)
    for r in results:
        p_xy = f"[{r.plate_init_xy[0]:.3f}, {r.plate_init_xy[1]:.3f}]"
        s_xy = f"[{r.spoon_init_xy[0]:.3f}, {r.spoon_init_xy[1]:.3f}]"
        steps_str = f"{len(r.passed_steps)}/6 ({r.passed_steps})"
        status = "PASS" if r.success else "FAIL"
        fail_str = r.failure_reason if r.failure_reason else "-"
        print(f"{r.seed:<6} | {p_xy:<22} | {s_xy:<22} | {r.friction_mult:.2f}x  | {r.light_intensity:.2f}x  | {steps_str:<14} | {status:<6} | {fail_str}")

    print("=" * 115)
    print(f"FINAL ROBUSTNESS SCORE: {pass_count}/{num_seeds} ({pass_count/num_seeds*100:.1f}%) SUCCESSFUL RUNS")
    print("=========================================================================================")


if __name__ == "__main__":
    run_robustness_harness(10)
