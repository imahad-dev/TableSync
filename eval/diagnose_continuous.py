"""
eval/diagnose_continuous.py
Systematic continuous telemetry and dense frame extraction for:
  Bug 1: Arm-on-arm jostling / collision during handoff (steps 3-4)
  Bug 2: Premature spoon drop in transit before place motion (steps 4-5)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contracts import (
    ArmID, ActionType, TargetObject, Subtask, SubtaskStatus,
    HandoffState, HandoffPhase
)
from controller import TableSyncController

def run_diagnostics():
    xml_path = PROJECT_ROOT / "scene" / "tablesync_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    controller = TableSyncController(model, data)
    controller.settle_scene(150)

    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    site_a = controller.site_a
    site_b = controller.site_b

    # Telemetry storage
    telemetry = []
    
    # State tracking
    curr_subtask = [-1]
    curr_waypoint = ["init"]
    curr_phase_detail = ["init"]
    dwell_confirmed_step = [-1]
    dwell_confirmed_time = [-1.0]
    st4_complete_step = [-1]
    st4_complete_time = [-1.0]
    st5_start_step = [-1]
    st5_start_time = [-1.0]
    handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)

    # Video frame tracking aligned with 30fps VideoRecorder
    video_fps = 30.0
    frame_interval = 1.0 / video_fps
    last_render_time = [data.time]
    video_frame_count = [0]
    sim_step_idx = [0]

    def per_step_hook():
        # Check if a video frame tick happened
        if (data.time - last_render_time[0]) >= (frame_interval - 1e-6):
            video_frame_count[0] += 1
            last_render_time[0] = data.time

        vf = video_frame_count[0]
        step = sim_step_idx[0]
        t = data.time

        # Contact counts
        a_sp = controller.count_contacts("armA", "spoon")
        b_sp = controller.count_contacts("armB", "spoon")

        # ArmA - ArmB collisions
        ab_collisions = []
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = model.geom(con.geom1).name or ""
            g2 = model.geom(con.geom2).name or ""
            if ("armA" in g1 and "armB" in g2) or ("armB" in g1 and "armA" in g2):
                ab_collisions.append((g1, g2))

        # Positions & Distances
        ee_a = data.site_xpos[site_a].copy()
        ee_b = data.site_xpos[site_b].copy()
        inter_ee_dist = float(np.linalg.norm(ee_a - ee_b))
        sp_pos = data.xpos[spoon_id].copy()

        # Waypoint and dwell confirmation tracking
        wp = controller.current_waypoint_name
        if handoff_state.phase == HandoffPhase.GRIP_CONFIRMED and dwell_confirmed_step[0] == -1:
            dwell_confirmed_step[0] = step
            dwell_confirmed_time[0] = t

        telemetry.append({
            "step": step,
            "time": t,
            "video_frame": vf,
            "subtask": curr_subtask[0],
            "waypoint": wp,
            "phase": curr_phase_detail[0],
            "a_sp": a_sp,
            "b_sp": b_sp,
            "ab_count": len(ab_collisions),
            "ab_pairs": ab_collisions,
            "inter_ee_dist": inter_ee_dist,
            "spoon_pos": sp_pos,
        })
        sim_step_idx[0] += 1

    # Patch move_arm to give fine-grained progress logging
    def instrumented_move_arm(arm: ArmID, q_target: np.ndarray, grip_val: float, steps: int = 80, hold_steps: int = 40):
        if arm == ArmID.A:
            q_start = controller.cmd_q_a.copy()
            controller.cmd_grip_a = grip_val
            for idx, alpha in enumerate(np.linspace(0, 1, steps)):
                curr_phase_detail[0] = f"interp_{idx}/{steps}"
                controller.cmd_q_a = (1 - alpha) * q_start + alpha * q_target
                controller.step_sim(2)
            controller.cmd_q_a = q_target.copy()
            for idx in range(hold_steps * 2):
                curr_phase_detail[0] = f"hold_{idx}/{hold_steps*2}"
                controller.step_sim(1)
        else:
            q_start = controller.cmd_q_b.copy()
            controller.cmd_grip_b = grip_val
            for idx, alpha in enumerate(np.linspace(0, 1, steps)):
                curr_phase_detail[0] = f"interp_{idx}/{steps}"
                controller.cmd_q_b = (1 - alpha) * q_start + alpha * q_target
                controller.step_sim(2)
            controller.cmd_q_b = q_target.copy()
            for idx in range(hold_steps * 2):
                curr_phase_detail[0] = f"hold_{idx}/{hold_steps*2}"
                controller.step_sim(1)

    controller.move_arm = instrumented_move_arm
    controller.step_callback = per_step_hook

    # 1.0 second initial settling
    curr_phase_detail[0] = "initial_settle"
    controller.step_sim(int(1.0 / model.opt.timestep))

    handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)

    plan = [
        Subtask(step_index=0, arm=ArmID.A, action=ActionType.PICK, target_object=TargetObject.PLATE),
        Subtask(step_index=1, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.PLATE, depends_on=0),
        Subtask(step_index=2, arm=ArmID.B, action=ActionType.PICK, target_object=TargetObject.SPOON),
        Subtask(step_index=3, arm=ArmID.B, action=ActionType.HANDOFF_EXTEND, target_object=TargetObject.HANDOFF_POINT, depends_on=2),
        Subtask(step_index=4, arm=ArmID.A, action=ActionType.HANDOFF_RECEIVE, target_object=TargetObject.HANDOFF_POINT, depends_on=3),
        Subtask(step_index=5, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.SPOON, depends_on=4),
    ]

    for st in plan:
        curr_subtask[0] = st.step_index
        if st.step_index == 5:
            st5_start_step[0] = sim_step_idx[0]
            st5_start_time[0] = data.time

        prev_phase = handoff_state.phase
        res = controller.execute_subtask(st, handoff_state=handoff_state)
        assert res.status == SubtaskStatus.COMPLETE, f"Subtask {st.step_index} failed: {res.error}"

        if st.step_index == 4:
            st4_complete_step[0] = sim_step_idx[0]
            st4_complete_time[0] = data.time

    # 2.0s final hold
    curr_subtask[0] = 6
    controller.current_waypoint_name = "final_hold"
    curr_phase_detail[0] = "final_hold"
    controller.step_sim(int(2.0 / model.opt.timestep))

    print(f"Total simulation steps logged: {len(telemetry)}")
    print(f"Total video frame ticks: {video_frame_count[0]}")

    # =========================================================================
    # STEP 2 ANSWERS & NUMERICAL TELEMETRY
    # =========================================================================
    print("\n" + "=" * 75)
    print("EMPIRICAL TELEMETRY REPORT")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # BUG 1 ANALYSIS: Inter-gripper distance and arm-on-arm collisions
    # -------------------------------------------------------------------------
    handoff_recs = [r for r in telemetry if r["subtask"] in (3, 4)]
    min_dist_rec = min(handoff_recs, key=lambda r: r["inter_ee_dist"])
    
    print("\n--- BUG 1: INTER-GRIPPER DISTANCE & TRANSIENT ARM-ON-ARM CONTACT ---")
    print(f"Handoff window spans sim steps {handoff_recs[0]['step']} to {handoff_recs[-1]['step']}")
    print(f"Handoff time: t = {handoff_recs[0]['time']:.3f}s to {handoff_recs[-1]['time']:.3f}s")
    print(f"Handoff video frames: frame {handoff_recs[0]['video_frame']} to {handoff_recs[-1]['video_frame']}")
    print(f"Minimum inter-gripper distance: {min_dist_rec['inter_ee_dist']*1000.0:.2f} mm")
    print(f"  Occurred at sim step {min_dist_rec['step']} (t={min_dist_rec['time']:.3f}s, video_frame={min_dist_rec['video_frame']})")
    print(f"  Subtask: {min_dist_rec['subtask']}, Waypoint: {min_dist_rec['waypoint']}, Phase: {min_dist_rec['phase']}")

    ab_collisions = [r for r in handoff_recs if r["ab_count"] > 0]
    print(f"\nArmA-ArmB Direct Contact Check:")
    print(f"  Total simulation steps with direct Arm A - Arm B collision: {len(ab_collisions)}")
    if ab_collisions:
        first_c = ab_collisions[0]
        last_c = ab_collisions[-1]
        max_c = max(ab_collisions, key=lambda r: r["ab_count"])
        print(f"  First collision: step {first_c['step']} (t={first_c['time']:.3f}s, frame={first_c['video_frame']}) in wp '{first_c['waypoint']}' ({first_c['phase']})")
        print(f"  Last collision:  step {last_c['step']} (t={last_c['time']:.3f}s, frame={last_c['video_frame']}) in wp '{last_c['waypoint']}' ({last_c['phase']})")
        print(f"  Peak collision contacts: {max_c['ab_count']} simultaneous contacts at step {max_c['step']}")
        
        # Unique geom pairs
        pair_counts = {}
        for r in ab_collisions:
            for p in r["ab_pairs"]:
                pair_key = tuple(sorted(p))
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        print(f"  Colliding geom pairs and duration (step counts):")
        for pair, cnt in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {pair[0]} <---> {pair[1]}: {cnt} steps ({cnt * model.opt.timestep * 1000:.1f} ms)")
    else:
        print("  NO direct arm-on-arm contact occurred.")

    # -------------------------------------------------------------------------
    # BUG 2 ANALYSIS: Timing of armA-spoon contact drop & spoon drop
    # -------------------------------------------------------------------------
    print("\n--- BUG 2: SPOON DROP TIMING & CONTACT ANALYSIS ---")
    dwell_step = dwell_confirmed_step[0]
    dwell_t = dwell_confirmed_time[0]
    st4_end = st4_complete_step[0]
    st4_end_t = st4_complete_time[0]
    st5_start = st5_start_step[0]
    st5_start_t = st5_start_time[0]

    print(f"Step 4 Dwell Confirmed at: sim step {dwell_step} (t={dwell_t:.3f}s, video_frame={telemetry[dwell_step]['video_frame']})")
    print(f"Step 4 (Handoff Receive) Completed at: sim step {st4_end} (t={st4_end_t:.3f}s, video_frame={telemetry[st4_end]['video_frame']})")
    print(f"Step 5 (Place Spoon) Initiated at:    sim step {st5_start} (t={st5_start_t:.3f}s, video_frame={telemetry[st5_start]['video_frame']})")

    # Trace armA-spoon contacts starting from dwell confirmation
    post_dwell_recs = [r for r in telemetry if r["step"] >= dwell_step]
    
    # When did armB release complete (b_sp == 0)?
    b_release_recs = [r for r in post_dwell_recs if r["b_sp"] == 0]
    if b_release_recs:
        print(f"Arm B spoon contacts reached 0 at sim step {b_release_recs[0]['step']} (t={b_release_recs[0]['time']:.3f}s, video_frame={b_release_recs[0]['video_frame']})")

    # When did armA spoon contacts drop to 0 after dwell?
    drop_recs = [r for r in post_dwell_recs if r["a_sp"] == 0]
    if drop_recs:
        first_drop = drop_recs[0]
        steps_since_dwell = first_drop["step"] - dwell_step
        time_since_dwell = first_drop["time"] - dwell_t
        print(f"\nArm A spoon contact FIRST DROPPED TO 0 at:")
        print(f"  Simulation step: {first_drop['step']}")
        print(f"  Simulation time: t = {first_drop['time']:.3f}s")
        print(f"  Video frame:     {first_drop['video_frame']}")
        print(f"  Subtask:         {first_drop['subtask']}")
        print(f"  Waypoint:        '{first_drop['waypoint']}'")
        print(f"  Phase Detail:    '{first_drop['phase']}'")
        print(f"  Spoon pos:       [{first_drop['spoon_pos'][0]:.4f}, {first_drop['spoon_pos'][1]:.4f}, {first_drop['spoon_pos'][2]:.4f}]")
        print(f"  Steps elapsed since dwell confirmed: {steps_since_dwell} steps ({time_since_dwell:.3f}s)")

        if first_drop["step"] < st5_start:
            print(f"  --> VERDICT: Contact dropped BEFORE Step 5 even started! (Occurred in Step {first_drop['subtask']})")
            print(f"  --> Steps before Step 5 start: {st5_start - first_drop['step']} steps")
        else:
            print(f"  --> VERDICT: Contact dropped during Step 5 in waypoint '{first_drop['waypoint']}'")

        # Compare against 40-step slip threshold
        print(f"\nComparison with documented ~40-step slip threshold:")
        print(f"  Steps from dwell confirmation to armB release complete: {b_release_recs[0]['step'] - dwell_step if b_release_recs else 'N/A'}")
        print(f"  Steps from dwell confirmation to first a_sp drop:     {steps_since_dwell} steps")
        print(f"  Ratio to 40-step slip threshold:                      {steps_since_dwell / 40.0:.2f}x")
    else:
        print("\nArm A spoon contact NEVER dropped to 0 after dwell confirmation!")

    # Check spoon height trajectory through Step 4 and Step 5
    print("\nSpoon z trajectory breakdown across waypoints:")
    for wp in ["ho_clamp_dwell", "ho_lift", "spoon_place_above", "spoon_place_down", "spoon_release"]:
        w_recs = [r for r in telemetry if r["waypoint"] == wp]
        if w_recs:
            z_init = w_recs[0]["spoon_pos"][2]
            z_final = w_recs[-1]["spoon_pos"][2]
            z_min = min(r["spoon_pos"][2] for r in w_recs)
            z_max = max(r["spoon_pos"][2] for r in w_recs)
            a_sp_min = min(r["a_sp"] for r in w_recs)
            a_sp_max = max(r["a_sp"] for r in w_recs)
            print(f"  Waypoint '{wp}': steps {w_recs[0]['step']}-{w_recs[-1]['step']} (t={w_recs[0]['time']:.2f}-{w_recs[-1]['time']:.2f}s, frames {w_recs[0]['video_frame']}-{w_recs[-1]['video_frame']})")
            print(f"    z range: [{z_min:.4f}, {z_max:.4f}], start: {z_init:.4f}, end: {z_final:.4f}, delta: {z_final - z_init:+.4f}m")
            print(f"    Arm A contacts range: [{a_sp_min}, {a_sp_max}]")

    # =========================================================================
    # STEP 1: DENSE FRAME EXTRACTION FROM eval/tablesync_e2e_demo.mp4
    # =========================================================================
    print("\n" + "=" * 75)
    print("STEP 1: EXTRACTING DENSE FRAMES FROM VIDEO")
    print("=" * 75)

    video_path = PROJECT_ROOT / "eval" / "tablesync_e2e_demo.mp4"
    assert video_path.exists(), f"Video file not found at {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened: {video_path} ({total_frames} frames)")

    out_dir_handoff = PROJECT_ROOT / "eval" / "diagnostics" / "dense_handoff"
    out_dir_place = PROJECT_ROOT / "eval" / "diagnostics" / "dense_transit_place"
    out_dir_handoff.mkdir(parents=True, exist_ok=True)
    out_dir_place.mkdir(parents=True, exist_ok=True)

    # Window 1: Handoff window (steps 3-4): frames 425 to 565
    print(f"\nExtracting Window 1 (Handoff Steps 3-4, frames 425 to 565)...")
    w1_count = 0
    for f_idx in range(425, 566, 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            out_p = out_dir_handoff / f"frame_{f_idx:03d}.png"
            cv2.imwrite(str(out_p), frame)
            w1_count += 1
    print(f"  Extracted {w1_count} dense frames to {out_dir_handoff}")

    # Window 2: Transition from Step 4 completion to Step 5 place motion: frames 555 to 650
    print(f"\nExtracting Window 2 (Transit & Place, frames 555 to 650)...")
    w2_count = 0
    for f_idx in range(555, 651, 1): # Every single frame!
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            out_p = out_dir_place / f"frame_{f_idx:03d}.png"
            cv2.imwrite(str(out_p), frame)
            w2_count += 1
    print(f"  Extracted {w2_count} dense frames to {out_dir_place}")

    cap.release()

    # Save summary telemetry to csv for comprehensive record
    csv_path = PROJECT_ROOT / "eval" / "diagnostics" / "continuous_telemetry.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,time,video_frame,subtask,waypoint,phase,a_sp,b_sp,ab_count,inter_ee_dist_mm,spoon_x,spoon_y,spoon_z\n")
        for r in telemetry:
            if r["subtask"] in (3, 4, 5): # focus on windows of interest
                f.write(f"{r['step']},{r['time']:.4f},{r['video_frame']},{r['subtask']},{r['waypoint']},{r['phase']},"
                        f"{r['a_sp']},{r['b_sp']},{r['ab_count']},{r['inter_ee_dist']*1000.0:.2f},"
                        f"{r['spoon_pos'][0]:.4f},{r['spoon_pos'][1]:.4f},{r['spoon_pos'][2]:.4f}\n")
    print(f"\nSaved CSV telemetry: {csv_path}")

if __name__ == "__main__":
    run_diagnostics()
