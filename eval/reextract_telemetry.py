import cv2
import numpy as np
import mujoco
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contracts import (
    ArmID, ActionType, TargetObject, Subtask, SubtaskStatus,
    PlannerOutput, HandoffState
)
from controller import TableSyncController

# 1. Run exact deterministic simulation with frame-accurate logging
xml_path = Path("scene/tablesync_scene.xml")
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
]

sim_dt = model.opt.timestep
frame_interval = 1.0 / 30.0
last_render_time = [data.time]
frame_positions = {}
frame_count = 0

def render_cb():
    global frame_count
    if (data.time - last_render_time[0]) >= (frame_interval - 1e-6):
        sp = data.xpos[spoon_id].copy()
        pl = data.xpos[plate_id].copy()
        a_con = controller.count_contacts("armA", "spoon")
        b_con = controller.count_contacts("armB", "spoon")
        frame_positions[frame_count] = {
            "time": data.time,
            "spoon_pos": sp,
            "plate_pos": pl,
            "a_con": a_con,
            "b_con": b_con,
        }
        frame_count += 1
        last_render_time[0] = data.time

controller.step_callback = render_cb
controller.step_sim(int(1.0 / sim_dt))
handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)

for subtask in plan:
    controller.execute_subtask(subtask, handoff_state=handoff_state)

controller.step_sim(int(2.0 / sim_dt))
controller.step_callback = None

print(f"Total simulated frames logged: {frame_count}")

# 2. Extract frames from eval/tablesync_e2e_demo.mp4 using cv2
video_path = Path("eval/tablesync_e2e_demo.mp4")
cap = cv2.VideoCapture(str(video_path))
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total video frames in {video_path}: {total_video_frames}")

target_indices = [
    (460, "step2", "eval/frame460_step2_recheck.png"),
    (540, "step3", "eval/frame540_step3_recheck.png"),
    (620, "step4", "eval/frame620_step4_recheck.png"),
]

for frame_idx, step_label, out_path in target_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: could not read frame {frame_idx}")
        continue

    sim_data = frame_positions.get(frame_idx, None)
    if sim_data:
        sp = sim_data["spoon_pos"]
        a_con = sim_data["a_con"]
        b_con = sim_data["b_con"]
        t = sim_data["time"]
        line1 = f"Frame: {frame_idx} | t={t:.2f}s | {step_label}"
        line2 = f"Spoon pos: [{sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f}]"
        line3 = f"Contacts: armA={a_con} armB={b_con}"
    else:
        line1 = f"Frame: {frame_idx} | {step_label}"
        line2 = "Spoon pos: N/A"
        line3 = "Contacts: N/A"

    # Burn-in overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    cv2.putText(frame, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, line3, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, frame)
    print(f"Saved {out_path} with burned-in telemetry:")
    print(f"  {line1}")
    print(f"  {line2}")
    print(f"  {line3}")

cap.release()
