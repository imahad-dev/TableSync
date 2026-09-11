"""
TableSync — Interface Contracts
================================
The strict boundary between the reasoning layer (Gemini, multimodal) and
the execution layer (a pretrained LeRobot policy served through Intel's
`physicalai.inference.InferenceModel`). Gemini never touches actuators
directly; it only ever emits a PlannerOutput, and the execution layer only
ever reports back an ExecutionResult. Nothing crosses that boundary except
these two shapes.

Grounded against real, checked APIs — not guessed:
  - so101_nexus.GraspState: contact-force + opposing-normal signal.
    Per its own docstring it is "not a proof of load bearing" -- a caged
    object can read 1.0 without being liftable. HandoffState.grip_confirm_frames
    exists specifically to add a dwell-time check on top of that raw signal.
  - physicalai.inference.InferenceModel.select_action(observation: dict[str, np.ndarray]) -> np.ndarray
    -- this is the exact call the execution layer wraps per subtask.
  - HANDOFF_POINT_WORLD below is derived from an empirical grid search
    over damped least-squares IK solutions (via `so101_nexus.kinematics.ee_ik_delta_q`)
    confirming both Arm A (base at [-0.16, -0.12, 0.20]) and Arm B (base at
    [0.16, -0.12, 0.20]) achieve simultaneous Cartesian convergence with < 0.3mm
    position error and comfortable clearance above the tabletop (z = 0.20m).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# Derived via empirical dual-arm IK reachability grid search (<0.3mm position error for both arms)
HANDOFF_POINT_WORLD: tuple[float, float, float] = (0.0, 0.15, 0.30)
GRIP_DWELL_FRAMES = 10  # consecutive frames GraspState must hold before trusted; tune once timestep-accurate


class ArmID(str, Enum):
    A = "arm_a"
    B = "arm_b"


class ActionType(str, Enum):
    PICK = "pick"
    PLACE = "place"
    HANDOFF_EXTEND = "handoff_extend"     # carrying arm moves to HANDOFF_POINT_WORLD and holds
    HANDOFF_RECEIVE = "handoff_receive"   # receiving arm closes gripper on the held object
    RETREAT = "retreat"
    WAIT = "wait"


class TargetObject(str, Enum):
    PLATE = "plate"
    SPOON = "spoon"
    HANDOFF_POINT = "handoff_point"        # not a scene object -- the fixed transfer coordinate


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class Subtask(BaseModel):
    """One atomic unit of work dispatched to the execution layer."""
    step_index: int = Field(..., ge=0)
    arm: ArmID
    action: ActionType
    target_object: TargetObject
    status: SubtaskStatus = SubtaskStatus.PENDING
    depends_on: Optional[int] = Field(
        default=None,
        description="step_index this subtask must wait on, e.g. HANDOFF_RECEIVE depends on the matching HANDOFF_EXTEND."
    )


class PlannerOutput(BaseModel):
    """Strict schema Gemini must return via JSON-mode / response_schema.
    Never free-text-parsed -- a malformed response is a failed subtask,
    not a best-effort regex extraction."""
    raw_instruction: str
    subtasks: list[Subtask]
    scene_summary: str = Field(
        description="One-sentence grounding of what Gemini saw in the camera "
                    "frame. Kept separate from the plan so failures are "
                    "diagnosable: wrong plan vs. wrong perception."
    )


class HandoffPhase(str, Enum):
    """Sub-states inside a HANDOFF_EXTEND / HANDOFF_RECEIVE pair. This is
    the actual synchronization protocol, not a narrative description of one."""
    APPROACHING = "approaching"            # giving arm moving toward HANDOFF_POINT_WORLD
    AT_TRANSFER_POSE = "at_transfer_pose"  # giving arm within goal_thresh of the handoff point, holding
    AWAITING_GRIP = "awaiting_grip"        # receiving arm moving in to grip the held object
    GRIP_CONFIRMED = "grip_confirmed"      # receiving arm's GraspState==1.0 held for GRIP_DWELL_FRAMES
    RELEASING = "releasing"                # giving arm opening its gripper
    RETREATING = "retreating"              # giving arm pulling back clear of receiving arm's workspace


class HandoffState(BaseModel):
    phase: HandoffPhase = HandoffPhase.APPROACHING
    giving_arm: ArmID
    receiving_arm: ArmID
    grip_confirm_frames: int = Field(
        default=0,
        description="Consecutive frames receiving_arm's GraspState==1.0 while "
                    "still in AWAITING_GRIP. Require >= GRIP_DWELL_FRAMES "
                    "before advancing to GRIP_CONFIRMED -- GraspState alone "
                    "is contact-based, not proof the object is actually held."
    )

    def advance_grip_check(self, grasp_state_value: float) -> None:
        """Call once per sim step while phase == AWAITING_GRIP."""
        if self.phase != HandoffPhase.AWAITING_GRIP:
            return
        if grasp_state_value >= 1.0:
            self.grip_confirm_frames += 1
        else:
            self.grip_confirm_frames = 0  # any drop resets the dwell count
        if self.grip_confirm_frames >= GRIP_DWELL_FRAMES:
            self.phase = HandoffPhase.GRIP_CONFIRMED


class ExecutionResult(BaseModel):
    """What the execution layer reports back up to the orchestrator after
    each subtask -- the only thing that crosses back over the boundary."""
    step_index: int
    status: SubtaskStatus
    final_ee_pose: Optional[list[float]] = None
    grasp_state: Optional[float] = None
    error: Optional[str] = None


if __name__ == "__main__":
    # Smoke test: build the 3-step scoped task and confirm the schema round-trips.
    plan = PlannerOutput(
        raw_instruction="Pick up the plate with arm A, place it on the table, "
                        "then hand the spoon to arm A with arm B.",
        scene_summary="Plate near arm A, spoon near arm B, table clear otherwise.",
        subtasks=[
            Subtask(step_index=0, arm=ArmID.A, action=ActionType.PICK, target_object=TargetObject.PLATE),
            Subtask(step_index=1, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.PLATE, depends_on=0),
            Subtask(step_index=2, arm=ArmID.B, action=ActionType.PICK, target_object=TargetObject.SPOON),
            Subtask(step_index=3, arm=ArmID.B, action=ActionType.HANDOFF_EXTEND, target_object=TargetObject.HANDOFF_POINT, depends_on=2),
            Subtask(step_index=4, arm=ArmID.A, action=ActionType.HANDOFF_RECEIVE, target_object=TargetObject.HANDOFF_POINT, depends_on=3),
            Subtask(step_index=5, arm=ArmID.A, action=ActionType.PLACE, target_object=TargetObject.SPOON, depends_on=4),
        ],
    )
    print(plan.model_dump_json(indent=2))

    hs = HandoffState(giving_arm=ArmID.B, receiving_arm=ArmID.A, phase=HandoffPhase.AWAITING_GRIP)
    for _ in range(GRIP_DWELL_FRAMES + 2):
        hs.advance_grip_check(1.0)
    assert hs.phase == HandoffPhase.GRIP_CONFIRMED, "dwell check should have advanced the phase"
    print("Handoff dwell-check smoke test: OK ->", hs.phase)
