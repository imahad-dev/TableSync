"""
TableSync -- Gemini Multimodal Planner
======================================
Sends instruction text + camera frame to Gemini's multimodal API.
Returns a validated PlannerOutput using structured-output mode
(response_schema), not free-text parsing.

Environment:
  GEMINI_API_KEY -- your Gemini API key (or set in .env)
"""

from __future__ import annotations

import base64
import io
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Load .env before anything reads os.environ
def _load_dotenv(env_path: Path | None = None) -> None:
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
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

import google.generativeai as genai
from PIL import Image

from contracts import (
    ArmID,
    ActionType,
    TargetObject,
    SubtaskStatus,
    Subtask,
    PlannerOutput,
    HANDOFF_POINT_WORLD,
)

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """\
You are the planning module of TableSync, a bimanual robotic table-clearing system.
You control two SO-101 robot arms: Arm A (left) and Arm B (right).

WORKSPACE RULES:
- The table has a plate (near Arm A) and a spoon (near Arm B).
- Each arm can PICK, PLACE, HANDOFF_EXTEND, HANDOFF_RECEIVE, RETREAT, or WAIT.
- PHYSICAL CONSTRAINTS:
  * Each arm has EXACTLY ONE gripper and can hold at most ONE object at a time.
  * A gripper CANNOT hold two objects simultaneously. Opening an occupied gripper drops the object!
  * If an arm has picked an object (e.g. Arm A picks plate), it MUST place it on the table (action: "place") to free its gripper before participating in any subsequent pick or handoff_receive.
  * When a command involves both objects (plate and spoon), always ensure an occupied arm places its first object before receiving or picking a second object.
- Inter-arm object transfer MUST use the handoff protocol:
  1. The carrying arm does HANDOFF_EXTEND to the handoff_point.
  2. The receiving arm does HANDOFF_RECEIVE at the handoff_point (depends_on the extend step).
- target_object for handoff steps is always "handoff_point".
- arm values: "arm_a" or "arm_b".
- action values: "pick", "place", "handoff_extend", "handoff_receive", "retreat", "wait".
- target_object values: "plate", "spoon", "handoff_point".
- status must always be "pending" for new plans.
- step_index is 0-based and sequential.
- depends_on links to the step_index this step must wait for (null if independent).

OUTPUT: Return a JSON object matching the PlannerOutput schema exactly.
Include raw_instruction (echo the user's command), scene_summary (one sentence
describing what you see in the camera image), and subtasks (ordered list).

CRITICAL: Do NOT invent objects or actions outside the enum values above.
"""

# The JSON schema Gemini must conform to (structured output mode)
PLANNER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "raw_instruction": {"type": "string"},
        "scene_summary": {"type": "string"},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_index": {"type": "integer"},
                    "arm": {"type": "string", "enum": ["arm_a", "arm_b"]},
                    "action": {"type": "string", "enum": [
                        "pick", "place", "handoff_extend",
                        "handoff_receive", "retreat", "wait"
                    ]},
                    "target_object": {"type": "string", "enum": [
                        "plate", "spoon", "handoff_point"
                    ]},
                    "status": {"type": "string", "enum": ["pending"]},
                    "depends_on": {"type": "integer", "nullable": True},
                },
                "required": ["step_index", "arm", "action", "target_object", "status"],
            },
        },
    },
    "required": ["raw_instruction", "scene_summary", "subtasks"],
}


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a MuJoCo-rendered RGB numpy array to PIL Image."""
    return Image.fromarray(frame.astype(np.uint8))


def generate_plan(
    instruction_text: str,
    camera_frame: np.ndarray,
    api_key: Optional[str] = None,
) -> PlannerOutput:
    """Send instruction + camera frame to Gemini, return validated PlannerOutput.

    Uses Gemini's structured-output / response_schema mode to constrain
    the response to match PlannerOutput's schema exactly.

    Raises:
        RuntimeError: If Gemini returns invalid JSON or schema mismatch.
        ValueError: If the response doesn't validate against PlannerOutput.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY in .env or environment."
        )

    genai.configure(api_key=key)

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=PLANNER_OUTPUT_SCHEMA,
            temperature=0.1,
        ),
    )

    pil_image = _frame_to_pil(camera_frame)

    user_prompt = (
        f"The operator said: \"{instruction_text}\"\n\n"
        "Look at the attached camera image of the current workspace. "
        "Generate the execution plan as a PlannerOutput JSON."
    )

    print("[GeminiPlanner] Sending to Gemini...")
    print(f"  Model: {GEMINI_MODEL}")
    print(f"  Instruction: \"{instruction_text}\"")
    print(f"  Image size: {pil_image.size}")

    response = model.generate_content([user_prompt, pil_image])

    raw_text = response.text
    print(f"\n[GeminiPlanner] Raw Gemini response:\n{raw_text}")

    # Parse and validate against Pydantic model
    try:
        raw_dict = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Gemini returned invalid JSON: {e}\nRaw response:\n{raw_text}"
        ) from e

    try:
        plan = PlannerOutput.model_validate(raw_dict)
    except Exception as e:
        raise ValueError(
            f"Gemini response does not match PlannerOutput schema: {e}\n"
            f"Raw dict:\n{json.dumps(raw_dict, indent=2)}"
        ) from e

    print(f"\n[GeminiPlanner] Validated PlannerOutput:")
    print(f"  raw_instruction: \"{plan.raw_instruction}\"")
    print(f"  scene_summary: \"{plan.scene_summary}\"")
    print(f"  subtasks ({len(plan.subtasks)}):")
    for st in plan.subtasks:
        dep = f" (depends_on={st.depends_on})" if st.depends_on is not None else ""
        print(f"    [{st.step_index}] {st.arm.value} -> {st.action.value} -> {st.target_object.value}{dep}")

    return plan


if __name__ == "__main__":
    # Quick smoke test with a synthetic image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[100:380, 100:540] = [80, 60, 40]  # table-ish rectangle
    plan = generate_plan(
        "Pick up the plate with Arm A and hand the spoon to Arm A with Arm B",
        dummy_frame,
    )
    print("\n" + plan.model_dump_json(indent=2))
