"""
TableSync -- True End-to-End Pipeline
======================================
The single real entry point that wires every proven piece together:

  1. Speechmatics voice transcription (voice_transcriber.py)
  2. MuJoCo scene rendering (overhead camera frame)
  3. Gemini multimodal planning (gemini_planner.py)
  4. Deterministic IK execution (controller.py)
  5. Full-sequence video recording (.mp4)

Usage:
  python run_tablesync_e2e.py path/to/command.wav
  python run_tablesync_e2e.py   (uses microphone)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import cv2

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
def _load_dotenv(env_path: Path | None = None) -> None:
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

import mujoco
from PIL import Image

from contracts import (
    ArmID, ActionType, TargetObject, Subtask, SubtaskStatus,
    PlannerOutput, HandoffState, HandoffPhase, HANDOFF_POINT_WORLD,
)
from controller import TableSyncController
from gemini_planner import generate_plan
from voice_transcriber import transcribe_spoken_command

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "eval"
VIDEO_FPS = 30
RENDER_WIDTH = 640
RENDER_HEIGHT = 480


def render_overhead_frame(model, data) -> np.ndarray:
    """Render a single overhead camera frame from the current scene state."""
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    renderer.update_scene(data, camera="overhead")
    frame = renderer.render().copy()
    renderer.close()
    return frame


class AngledRenderer:
    """Persistent renderer for the angled demo camera view."""

    def __init__(self, model, width: int = RENDER_WIDTH, height: int = RENDER_HEIGHT):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        self.cam.lookat[:] = [0.0, 0.10, 0.22]
        self.cam.distance = 0.65
        self.cam.elevation = -32.0
        self.cam.azimuth = 90.0

    def render(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.cam)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def render_angled_frame(model, data) -> np.ndarray:
    """Render a single frame from the demo/video perspective (convenience wrapper)."""
    ar = AngledRenderer(model)
    frame = ar.render(data)
    ar.close()
    return frame


# ---------------------------------------------------------------------------
# Video recorder
# ---------------------------------------------------------------------------

class VideoRecorder:
    """Records MuJoCo frames into an MP4 video using Pillow (GIF fallback)."""

    def __init__(self, output_path: Path, fps: int = 30):
        self.output_path = output_path
        self.fps = fps
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def save(self) -> Path:
        """Save collected frames as video. Uses cv2 for MP4 if available, else GIF."""
        if not self.frames:
            raise RuntimeError("No frames recorded")

        # Try OpenCV for proper MP4
        try:
            import cv2
            mp4_path = self.output_path.with_suffix(".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w, _ = self.frames[0].shape
            writer = cv2.VideoWriter(str(mp4_path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter failed to open for {mp4_path}")
            try:
                for f in self.frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            print(f"[VideoRecorder] Saved MP4: {mp4_path} ({len(self.frames)} frames, {len(self.frames)/self.fps:.1f}s)")
            return mp4_path
        except (ImportError, Exception) as e:
            print(f"[VideoRecorder] cv2/MP4 unavailable ({e}), trying imageio/GIF")

        # Try imageio for proper MP4
        try:
            import imageio.v3 as iio
            mp4_path = self.output_path.with_suffix(".mp4")
            iio.imwrite(
                str(mp4_path),
                np.stack(self.frames),
                fps=self.fps,
                codec="libx264",
            )
            print(f"[VideoRecorder] Saved MP4: {mp4_path} ({len(self.frames)} frames)")
            return mp4_path
        except (ImportError, Exception) as e:
            print(f"[VideoRecorder] imageio/MP4 unavailable ({e}), falling back to GIF")

        # Fallback: save as animated GIF via Pillow
        gif_path = self.output_path.with_suffix(".gif")
        pil_frames = [Image.fromarray(f) for f in self.frames]
        duration_ms = int(1000 / self.fps)
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[VideoRecorder] Saved GIF: {gif_path} ({len(self.frames)} frames, {len(self.frames)/self.fps:.1f}s)")
        return gif_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_e2e(audio_path: str | Path | None = None):
    """Run the full TableSync end-to-end pipeline."""

    print("=" * 70)
    print("TABLESYNC END-TO-END PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Voice transcription
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Voice Transcription ---")
    if audio_path is not None:
        print(f"Audio file: {audio_path}")
    else:
        print("Mode: Live microphone (10s timeout)")

    instruction_text = asyncio.run(
        transcribe_spoken_command(audio_path=audio_path, duration_s=10.0)
    )
    print(f"Transcribed instruction: \"{instruction_text}\"")

    # ------------------------------------------------------------------
    # Step 2: Initialize MuJoCo scene and render camera frame
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Scene Initialization & Camera Render ---")
    xml_path = PROJECT_ROOT / "scene" / "tablesync_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    controller = TableSyncController(model, data)
    controller.settle_scene(150)

    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    spoon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
    init_plate_pos = data.xpos[plate_id].copy()
    init_spoon_pos = data.xpos[spoon_id].copy()

    print(f"Initial plate position: {init_plate_pos.tolist()}")
    print(f"Initial spoon position: {init_spoon_pos.tolist()}")

    camera_frame = render_overhead_frame(model, data)
    overhead_path = OUTPUT_DIR / "e2e_initial_overhead.png"
    Image.fromarray(camera_frame).save(str(overhead_path))
    print(f"Overhead camera frame saved: {overhead_path}")
    print(f"Frame shape: {camera_frame.shape}")

    # ------------------------------------------------------------------
    # Step 3: Gemini planning (REAL multimodal call)
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Gemini Multimodal Planning ---")
    plan = generate_plan(instruction_text, camera_frame)

    # CHECKPOINT: Print full plan before any execution
    print("\n" + "=" * 70)
    print("CHECKPOINT: Gemini PlannerOutput (pre-execution review)")
    print("=" * 70)
    print(plan.model_dump_json(indent=2))
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 4: Execute plan with video recording
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Execution with Video Recording ---")

    recorder = VideoRecorder(OUTPUT_DIR / "tablesync_e2e_demo", fps=VIDEO_FPS)
    angled_renderer = AngledRenderer(model)

    sim_dt = model.opt.timestep
    frame_interval = 1.0 / VIDEO_FPS
    last_render_time = [data.time]
    current_phase = ["Init: Settling scene"]

    def render_callback():
        if (data.time - last_render_time[0]) >= (frame_interval - 1e-6):
            raw_frame = angled_renderer.render(data)
            f_idx = len(recorder.frames)
            t = data.time
            p_pos = data.xpos[plate_id]
            s_pos = data.xpos[spoon_id]
            a_plate_con = controller.count_contacts("armA", "plate")
            a_spoon_con = controller.count_contacts("armA", "spoon")
            b_spoon_con = controller.count_contacts("armB", "spoon")
            sp_pl_con = controller.count_contacts("spoon", "plate")

            line1 = f"Frame: {f_idx:03d} | t={t:.2f}s | {current_phase[0]}"
            line2 = f"Plate pos: [{p_pos[0]:.4f}, {p_pos[1]:.4f}, {p_pos[2]:.4f}]"
            line3 = f"Spoon pos: [{s_pos[0]:.4f}, {s_pos[1]:.4f}, {s_pos[2]:.4f}]"
            line4 = f"Contacts: armA_pl={a_plate_con} armB_sp={b_spoon_con} armA_sp={a_spoon_con} sp_pl={sp_pl_con}"

            frame = raw_frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (560, 118), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            cv2.putText(frame, line1, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, line2, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, line3, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, line4, (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

            recorder.add_frame(frame)
            last_render_time[0] = data.time

    # Record initial state (1 second)
    controller.step_callback = render_callback
    controller.step_sim(int(1.0 / sim_dt))

    handoff_state = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A)

    step_end_frames = {}
    for subtask in plan.subtasks:
        current_phase[0] = f"Step {subtask.step_index}: [{subtask.arm.value}] {subtask.action.value} -> {subtask.target_object.value}"
        print(f"\nExecuting Step {subtask.step_index}: "
              f"[{subtask.arm.value}] {subtask.action.value} -> {subtask.target_object.value}")

        # The controller steps physics internally and triggers render_callback
        # roughly every 1/30th of a simulated second
        result = controller.execute_subtask(subtask, handoff_state=handoff_state)
        step_end_frames[subtask.step_index] = len(recorder.frames) - 1
        print(f"  [EventTrace] Step {subtask.step_index} completed at video frame {step_end_frames[subtask.step_index]}")

        print(f"  Result: status={result.status.value} error={result.error}")
        if result.status != SubtaskStatus.COMPLETE:
            raise RuntimeError(f"Subtask {subtask.step_index} execution failed: {result.error}")

        # Local physical threshold assertions matching run_full_pipeline.py
        if subtask.action == ActionType.PICK and subtask.target_object == TargetObject.PLATE:
            plate_pos = data.xpos[plate_id].copy()
            gain_0 = plate_pos[2] - init_plate_pos[2]
            print(f"  [Metric] Plate height gain: {gain_0:.4f} m ({gain_0*100:.2f} cm) | Threshold: >= 0.045 m")
            assert gain_0 >= 0.045, f"Plate lift gain {gain_0} insufficient (< 0.045m)"

        elif subtask.action == ActionType.PLACE and subtask.target_object == TargetObject.PLATE:
            plate_pos = data.xpos[plate_id].copy()
            disp_1 = np.linalg.norm(plate_pos[:2] - init_plate_pos[:2])
            a_plate_contacts = controller.count_contacts("armA", "plate")
            print(f"  [Metric] Placed plate pos: {plate_pos.tolist()}")
            print(f"  [Metric] Tabletop displacement: {disp_1:.4f} m ({disp_1*100:.2f} cm) | Threshold: >= 0.050 m")
            print(f"  [Metric] Settled z: {plate_pos[2]:.4f} m | Threshold: <= 0.205 m")
            print(f"  [Metric] Arm A contacts after release: {a_plate_contacts} | Threshold: == 0")
            assert disp_1 >= 0.050, f"Displacement {disp_1} too small (< 0.050m)"
            assert plate_pos[2] <= 0.205, f"Plate not settled on table: z={plate_pos[2]} (> 0.205m)"
            assert a_plate_contacts == 0, f"Arm A still contacting plate: {a_plate_contacts}"

        elif subtask.action == ActionType.PICK and subtask.target_object == TargetObject.SPOON:
            spoon_pos = data.xpos[spoon_id].copy()
            gain_2 = spoon_pos[2] - init_spoon_pos[2]
            print(f"  [Metric] Spoon height gain: {gain_2:.4f} m ({gain_2*100:.2f} cm) | Threshold: >= 0.050 m")
            assert gain_2 >= 0.050, f"Spoon lift gain {gain_2} insufficient (< 0.050m)"

        elif subtask.action == ActionType.HANDOFF_EXTEND and subtask.target_object == TargetObject.HANDOFF_POINT:
            ee_b = data.site_xpos[controller.site_b].copy()
            target_ho_b = np.array([0.04, 0.15, 0.29])
            err_3 = np.linalg.norm(ee_b - target_ho_b)
            print(f"  [Metric] Arm B EE pos: {ee_b.tolist()}")
            print(f"  [Metric] Position error to handoff target: {err_3*1000:.2f} mm | Threshold: <= 20.0 mm")
            assert err_3 <= 0.020, f"Position error {err_3} too large (> 20.0mm)"

        elif subtask.action == ActionType.HANDOFF_RECEIVE and subtask.target_object == TargetObject.HANDOFF_POINT:
            b_spoon_contacts = controller.count_contacts("armB", "spoon")
            spoon_pos = data.xpos[spoon_id].copy()
            gain_4 = spoon_pos[2] - init_spoon_pos[2]
            print(f"  [Metric] Dwell frames confirmed: {handoff_state.grip_confirm_frames} | Threshold: >= 10")
            print(f"  [Metric] Arm B contacts after release: {b_spoon_contacts} | Threshold: == 0")
            print(f"  [Metric] Held spoon elevation: {gain_4:.4f} m ({gain_4*100:.2f} cm) | Threshold: >= 0.050 m")
            assert handoff_state.grip_confirm_frames >= 10, f"Insufficient dwell frames ({handoff_state.grip_confirm_frames} < 10)"
            assert b_spoon_contacts == 0, f"Arm B failed to release spoon: {b_spoon_contacts} contacts"
            assert gain_4 >= 0.050, f"Spoon not elevated after handoff: gain {gain_4} (< 0.050m)"

        elif subtask.action == ActionType.PLACE and subtask.target_object == TargetObject.SPOON:
            final_plate_pos = data.xpos[plate_id].copy()
            final_spoon_pos = data.xpos[spoon_id].copy()
            a_spoon_contacts = controller.count_contacts("armA", "spoon")
            spoon_plate_contacts = controller.count_contacts("spoon", "plate")
            rel_vector = final_spoon_pos - final_plate_pos
            rel_distance_xy = np.linalg.norm(rel_vector[:2])
            print(f"  [Metric] Final placed spoon pos: {final_spoon_pos.tolist()}")
            print(f"  [Metric] Spoon settled z: {final_spoon_pos[2]:.4f} m | Threshold: <= 0.220 m")
            print(f"  [Metric] Arm A contacts after release: {a_spoon_contacts} | Threshold: == 0")
            print(f"  [Metric] Spoon-to-plate contacts: {spoon_plate_contacts} | Threshold: == 0")
            print(f"  [Metric] Distance to placed plate: {rel_distance_xy:.4f} m ({rel_distance_xy*100:.2f} cm) | Threshold: <= 0.100 m")
            assert final_spoon_pos[2] <= 0.220, f"Spoon not resting on table: z={final_spoon_pos[2]}"
            assert a_spoon_contacts == 0, f"Arm A still contacting spoon: {a_spoon_contacts}"
            assert spoon_plate_contacts == 0, f"Spoon is contacting plate: {spoon_plate_contacts}"
            assert rel_distance_xy <= 0.100, f"Spoon placed too far from plate: {rel_distance_xy} m"

    # Record final resting state (2 seconds)
    current_phase[0] = "Final Hold: Settled on Tabletop"
    controller.step_sim(int(2.0 / sim_dt))
    final_hold_frame = len(recorder.frames) - 1
    print(f"\n[EventTrace] Final 2-second hold completed at video frame {final_hold_frame}")
    controller.step_callback = None
    angled_renderer.close()

    # Save video
    video_path = recorder.save()

    # Save final resting frame
    final_frame = render_overhead_frame(model, data)
    final_path = OUTPUT_DIR / "e2e_final_overhead.png"
    Image.fromarray(final_frame).save(str(final_path))
    print(f"Final overhead frame saved: {final_path}")

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    final_plate_pos = data.xpos[plate_id].copy()
    final_spoon_pos = data.xpos[spoon_id].copy()

    print("\n" + "=" * 70)
    print("TABLESYNC E2E PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Instruction: \"{instruction_text}\"")
    print(f"Gemini plan: {len(plan.subtasks)} subtasks")
    print(f"Scene summary: \"{plan.scene_summary}\"")
    print(f"Final plate pos: {final_plate_pos.tolist()}")
    print(f"Final spoon pos: {final_spoon_pos.tolist()}")
    print(f"Video saved: {video_path}")
    print(f"Initial frame: {overhead_path}")
    print(f"Final frame: {final_path}")
    print("=" * 70)

    return plan, video_path


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    run_e2e(audio_path=audio)
