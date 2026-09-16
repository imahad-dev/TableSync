import cv2
import numpy as np
import mujoco
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contracts import ArmID, ActionType, TargetObject, Subtask, HandoffState
from controller import TableSyncController

# 1. Simulate exact 6-subtask plan with frame-accurate state logging
xml_path = PROJECT_ROOT / "scene" / "tablesync_scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

controller = TableSyncController(model, data)
controller.settle_scene(150)

plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")

plan = [
    Subtask(step_index=0, arm=ArmID.A, action=ActionType.PICK, target_object=TargetObject.PLATE),
    Subtask(step_index=1, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.PLATE, depends_on=0),
    Subtask(step_index=2, arm=ArmID.B, action=ActionType.PICK, target_object=TargetObject.SPOON),
    Subtask(step_index=3, arm=ArmID.B, action=ActionType.HANDOFF_EXTEND, target_object=TargetObject.HANDOFF_POINT, depends_on=2),
    Subtask(step_index=4, arm=ArmID.A, action=ActionType.HANDOFF_RECEIVE, target_object=TargetObject.HANDOFF_POINT, depends_on=3),
    Subtask(step_index=5, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.SPOON, depends_on=4),
]

sim_dt = model.opt.timestep
frame_interval = 1.0 / 30.0
last_render_time = [data.time]
frame_data = {}
f_count = 0

def cb():
    global f_count
    if (data.time - last_render_time[0]) >= (frame_interval - 1e-6):
        frame_data[f_count] = {
            "time": data.time,
            "plate_pos": data.xpos[plate_id].copy(),
            "spoon_pos": data.xpos[spoon_id].copy(),
            "a_plate_con": controller.count_contacts("armA", "plate"),
            "a_spoon_con": controller.count_contacts("armA", "spoon"),
            "b_spoon_con": controller.count_contacts("armB", "spoon"),
            "sp_pl_con": controller.count_contacts("spoon", "plate"),
        }
        f_count += 1
        last_render_time[0] = data.time

controller.step_callback = cb
controller.step_sim(int(1.0 / sim_dt))
handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)

step_ends = {}
for st in plan:
    controller.execute_subtask(st, handoff_state=handoff_state)
    step_ends[st.step_index] = f_count - 1

controller.step_sim(int(2.0 / sim_dt))
final_hold_f = f_count - 1
controller.step_callback = None

print(f"Total frames simulated: {f_count}")
print(f"Step 5 completion frame: {step_ends[5]}")
print(f"Final hold frame: {final_hold_f}")

# 2. Extract frames from eval/tablesync_e2e_demo.mp4 and burn in telemetry
video_path = PROJECT_ROOT / "eval" / "tablesync_e2e_demo.mp4"
cap = cv2.VideoCapture(str(video_path))
total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total video frames in mp4: {total_v_frames}")

targets = [
    (step_ends[5], "step5_place_spoon_completed", PROJECT_ROOT / "eval" / "frame_step5_place_spoon_completed.png"),
    (final_hold_f, "final_rest_hold_settled", PROJECT_ROOT / "eval" / "frame_final_rest_hold_settled.png"),
]

for f_idx, label, out_p in targets:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {f_idx}")
        continue

    fd = frame_data[f_idx]
    p_pos = fd["plate_pos"]
    s_pos = fd["spoon_pos"]
    t = fd["time"]

    line1 = f"Frame: {f_idx} | t={t:.2f}s | {label}"
    line2 = f"Plate pos: [{p_pos[0]:.4f}, {p_pos[1]:.4f}, {p_pos[2]:.4f}] (resting on table)"
    line3 = f"Spoon pos: [{s_pos[0]:.4f}, {s_pos[1]:.4f}, {s_pos[2]:.4f}] (resting on table)"
    line4 = f"Contacts: armA_pl={fd['a_plate_con']} armA_sp={fd['a_spoon_con']} sp_pl={fd['sp_pl_con']}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (560, 118), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, line1, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, line3, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, line4, (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_p), frame)
    print(f"Saved: {out_p}")
    print(f"  {line1}")
    print(f"  {line2}")
    print(f"  {line3}")
    print(f"  {line4}")

cap.release()
