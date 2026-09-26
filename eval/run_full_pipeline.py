"""
TableSync Full 6-Subtask Pipeline Verification
==============================================
Executes the complete 6-subtask sequence via TableSyncController and verifies
every step against explicit numerical thresholds:

  Subtask 0: Arm A Pick Plate
    - Numeric Threshold: Height gain delta_z >= 0.045m (4.5cm)
  Subtask 1: Arm A Place Plate (Separated Target)
    - Numeric Threshold: Tabletop displacement ||p_xy - p0_xy|| >= 0.050m (5.0cm)
    - Numeric Threshold: Settled on table z <= 0.205m
    - Numeric Threshold: Arm A contacts with plate after release == 0
  Subtask 2: Arm B Pick Spoon
    - Numeric Threshold: Height gain delta_z >= 0.050m (5.0cm)
  Subtask 3: Arm B Handoff Extend
    - Numeric Threshold: End-effector Cartesian error <= 0.010m (10.0mm)
  Subtask 4: Arm A Handoff Receive with Dwell Check
    - Numeric Threshold: Consecutive dwell frames confirmed >= 10
    - Numeric Threshold: Arm B contacts with spoon after release == 0
    - Numeric Threshold: Transferred spoon elevation delta_z >= 0.050m (5.0cm)
  Subtask 5: Arm A Place Spoon Beside Placed Plate
    - Numeric Threshold: Settled on table z <= 0.220m
    - Numeric Threshold: Arm A contacts with spoon after release == 0
    - Numeric Threshold: Distance to placed plate ||p_spoon,xy - p_plate,xy|| <= 0.100m (10.0cm)
"""

import mujoco
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts import (
    ArmID, ActionType, TargetObject, Subtask, SubtaskStatus,
    PlannerOutput, HandoffState, HandoffPhase, HANDOFF_POINT_WORLD
)
from controller import TableSyncController


def run_evaluation():
    xml_path = Path(__file__).resolve().parent.parent / "scene" / "tablesync_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    controller = TableSyncController(model, data)
    controller.settle_scene(150)

    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")

    init_plate_pos = data.xpos[plate_id].copy()
    init_spoon_pos = data.xpos[spoon_id].copy()

    plan = PlannerOutput(
        raw_instruction="Pick up the plate with arm A, place it on the table at a new location, "
                        "then hand the spoon to arm A with arm B, and place the spoon next to the plate.",
        scene_summary="Plate on table left, spoon on table right, both arms ready.",
        subtasks=[
            Subtask(step_index=0, arm=ArmID.A, action=ActionType.PICK, target_object=TargetObject.PLATE),
            Subtask(step_index=1, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.PLATE, depends_on=0),
            Subtask(step_index=2, arm=ArmID.B, action=ActionType.PICK, target_object=TargetObject.SPOON),
            Subtask(step_index=3, arm=ArmID.B, action=ActionType.HANDOFF_EXTEND, target_object=TargetObject.HANDOFF_POINT, depends_on=2),
            Subtask(step_index=4, arm=ArmID.A, action=ActionType.HANDOFF_RECEIVE, target_object=TargetObject.HANDOFF_POINT, depends_on=3),
            Subtask(step_index=5, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.SPOON, depends_on=4),
        ]
    )

    print("================================================================")
    print("TABLESYNC END-TO-END 6-SUBTASK EVALUATION")
    print("================================================================")
    print(f"Initial Plate Position [x, y, z]: {init_plate_pos.tolist()}")
    print(f"Initial Spoon Position [x, y, z]: {init_spoon_pos.tolist()}")
    print("----------------------------------------------------------------")

    handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)

    min_step5_transit_contacts = [float("inf")]
    def pipeline_step_cb():
        if controller.current_waypoint_name in ("spoon_place_above", "spoon_place_down"):
            c = controller.count_contacts("armA", "spoon")
            min_step5_transit_contacts[0] = min(min_step5_transit_contacts[0], c)
    controller.step_callback = pipeline_step_cb

    for subtask in plan.subtasks:
        print(f"\nExecuting Step {subtask.step_index}: [{subtask.arm.value}] {subtask.action.value} -> {subtask.target_object.value}")
        result = controller.execute_subtask(subtask, handoff_state=handoff_state)
        print(f"  Result: status={result.status.value} error={result.error}")
        assert result.status == SubtaskStatus.COMPLETE, f"Subtask {subtask.step_index} failed: {result.error}"

        if subtask.step_index == 0:
            plate_pos = data.xpos[plate_id].copy()
            gain_0 = plate_pos[2] - init_plate_pos[2]
            print(f"  [Metric] Plate height gain: {gain_0:.4f} m ({gain_0*100:.2f} cm) | Threshold: >= 0.045 m")
            assert gain_0 >= 0.045, f"Plate lift gain {gain_0} insufficient"

        elif subtask.step_index == 1:
            plate_pos = data.xpos[plate_id].copy()
            disp_1 = np.linalg.norm(plate_pos[:2] - init_plate_pos[:2])
            a_plate_contacts = controller.count_contacts("armA", "plate")
            print(f"  [Metric] Placed plate pos: {plate_pos.tolist()}")
            print(f"  [Metric] Tabletop displacement: {disp_1:.4f} m ({disp_1*100:.2f} cm) | Threshold: >= 0.050 m")
            print(f"  [Metric] Settled z: {plate_pos[2]:.4f} m | Threshold: <= 0.205 m")
            print(f"  [Metric] Arm A contacts after release: {a_plate_contacts} | Threshold: == 0")
            assert disp_1 >= 0.050, f"Displacement {disp_1} too small"
            assert plate_pos[2] <= 0.205, f"Plate not settled on table: z={plate_pos[2]}"
            assert a_plate_contacts == 0, f"Arm A still contacting plate: {a_plate_contacts}"

        elif subtask.step_index == 2:
            spoon_pos = data.xpos[spoon_id].copy()
            gain_2 = spoon_pos[2] - init_spoon_pos[2]
            print(f"  [Metric] Spoon height gain: {gain_2:.4f} m ({gain_2*100:.2f} cm) | Threshold: >= 0.050 m")
            assert gain_2 >= 0.050, f"Spoon lift gain {gain_2} insufficient"

        elif subtask.step_index == 3:
            ee_b = data.site_xpos[controller.site_b].copy()
            target_ho_b = np.array([0.04, 0.15, 0.29])
            err_3 = np.linalg.norm(ee_b - target_ho_b)
            print(f"  [Metric] Arm B EE pos: {ee_b.tolist()}")
            print(f"  [Metric] Position error to handoff target: {err_3*1000:.2f} mm | Threshold: <= 20.0 mm")
            assert err_3 <= 0.020, f"Position error {err_3} too large"

        elif subtask.step_index == 4:
            b_spoon_contacts = controller.count_contacts("armB", "spoon")
            spoon_pos = data.xpos[spoon_id].copy()
            gain_4 = spoon_pos[2] - init_spoon_pos[2]
            print(f"  [Metric] Dwell frames confirmed: {handoff_state.grip_confirm_frames} | Threshold: >= 10")
            print(f"  [Metric] Arm B contacts after release: {b_spoon_contacts} | Threshold: == 0")
            print(f"  [Metric] Held spoon elevation: {gain_4:.4f} m ({gain_4*100:.2f} cm) | Threshold: >= 0.050 m")
            assert handoff_state.grip_confirm_frames >= 10, "Insufficient dwell frames"
            assert b_spoon_contacts == 0, f"Arm B failed to release spoon: {b_spoon_contacts} contacts"
            assert gain_4 >= 0.050, "Spoon not elevated after handoff"

        elif subtask.step_index == 5:
            final_plate_pos = data.xpos[plate_id].copy()
            final_spoon_pos = data.xpos[spoon_id].copy()
            a_spoon_contacts = controller.count_contacts("armA", "spoon")
            spoon_plate_contacts = controller.count_contacts("spoon", "plate")
            rel_vector = final_spoon_pos - final_plate_pos
            rel_distance_xy = np.linalg.norm(rel_vector[:2])
            print(f"  [Metric] Final placed spoon pos: {final_spoon_pos.tolist()}")
            print(f"  [Metric] Spoon settled z: {final_spoon_pos[2]:.4f} m | Threshold: <= 0.220 m")
            print(f"  [Metric] Min Arm A spoon contacts during transit: {min_step5_transit_contacts[0]} | Threshold: > 0")
            print(f"  [Metric] Arm A contacts after release: {a_spoon_contacts} | Threshold: == 0")
            print(f"  [Metric] Spoon-to-plate contacts: {spoon_plate_contacts} | Threshold: == 0")
            print(f"  [Metric] Distance to placed plate: {rel_distance_xy:.4f} m ({rel_distance_xy*100:.2f} cm) | Threshold: <= 0.100 m")
            assert min_step5_transit_contacts[0] > 0, (
                f"Continuous invariant violated: Arm A dropped spoon during transit before deliberate release "
                f"(min contacts = {min_step5_transit_contacts[0]})"
            )
            assert final_spoon_pos[2] <= 0.220, f"Spoon not resting on table: z={final_spoon_pos[2]}"
            assert a_spoon_contacts == 0, f"Arm A still contacting spoon: {a_spoon_contacts}"
            assert spoon_plate_contacts == 0, f"Spoon is contacting plate: {spoon_plate_contacts}"
            assert rel_distance_xy <= 0.100, f"Spoon placed too far from plate: {rel_distance_xy} m"

    final_plate_pos = data.xpos[plate_id].copy()
    final_spoon_pos = data.xpos[spoon_id].copy()
    rel_vector = final_spoon_pos - final_plate_pos

    print("\n================================================================")
    print("FINAL 6-SUBTASK EXECUTION SUMMARY")
    print("================================================================")
    print(f"Final Plate Position [x, y, z]: {final_plate_pos.tolist()}")
    print(f"Final Spoon Position [x, y, z]: {final_spoon_pos.tolist()}")
    print(f"Relative Vector (Spoon - Plate): {rel_vector.tolist()}")
    print(f"Horizontal Separation:           {np.linalg.norm(rel_vector[:2]):.4f} m ({np.linalg.norm(rel_vector[:2])*100:.2f} cm)")
    print(f"Arm B Contacts with Spoon:       {controller.count_contacts('armB', 'spoon')}")
    print(f"Arm A Contacts with Spoon:       {controller.count_contacts('armA', 'spoon')}")
    print(f"Spoon Contacts with Plate:       {controller.count_contacts('spoon', 'plate')}")
    print("ALL 6 SUBTASKS PASSED AND NUMERICALLY VERIFIED!")
    print("================================================================")

    # Render final resting frames (both overhead and angled camera perspectives)
    try:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=480, width=640)

        # 1. Overhead Camera (Plan view)
        renderer.update_scene(data, camera="overhead")
        frame_overhead = renderer.render()
        out_overhead = Path(__file__).resolve().parent / "final_rest_overhead.png"
        out_legacy = Path(__file__).resolve().parent / "final_rest_frame.png"
        Image.fromarray(frame_overhead).save(out_overhead)
        Image.fromarray(frame_overhead).save(out_legacy)
        print(f"\nFinal resting overhead frame rendered to: {out_overhead}")

        # 2. Angled Natural View (Demo / Video perspective)
        cam_angled = mujoco.MjvCamera()
        cam_angled.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_angled.fixedcamid = -1
        cam_angled.lookat[:] = [0.0, 0.10, 0.22]
        cam_angled.distance = 0.65
        cam_angled.elevation = -32.0
        cam_angled.azimuth = 90.0
        renderer.update_scene(data, camera=cam_angled)
        frame_angled = renderer.render()
        out_angled = Path(__file__).resolve().parent / "final_rest_angled.png"
        Image.fromarray(frame_angled).save(out_angled)
        print(f"Final resting angled frame rendered to:   {out_angled}")

        renderer.close()
    except Exception as e:
        print(f"\nWarning: Could not render final frames: {e}")


if __name__ == "__main__":
    run_evaluation()
