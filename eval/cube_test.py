"""
Phase 2 + Phase 3: ACT Cube Pick-Place Test

Self-contained script that:
1. Loads the MuJoCo scene with the test cube
2. Loads the ACT policy with manual normalization from dataset stats
3. Runs a 500-step episode (10 seconds at 50Hz control)
4. Logs every action clamp event
5. Reports cube displacement (did it move? did it lift?)
6. Saves a frame sequence for visual inspection

This is the actual cube test — no more pre-checks.
"""
import json
import logging
import numpy as np
import torch
import mujoco
from pathlib import Path
from huggingface_hub import hf_hub_download

# ================================================================
# Configuration
# ================================================================
SCENE_XML = Path(r"c:\Users\MSI\Source Code\TableSync\scene\tablesync_scene.xml")
REPO_ID = "legalaspro/act-so101-pick-place-cube-50hz-v1"
DATASET_REPO = "legalaspro/so101-pick-and-place-cube-lerobot-50hz"

CONTROL_HZ = 50       # Policy runs at 50Hz
PHYSICS_HZ = 200       # MuJoCo at dt=0.005
ACTION_REPEAT = PHYSICS_HZ // CONTROL_HZ  # = 4
EPISODE_STEPS = 500    # 10 seconds at 50Hz
IMG_HEIGHT = 480
IMG_WIDTH = 640

# Output directory for frame captures
OUT_DIR = Path(r"C:\Users\MSI\.gemini\antigravity-ide\brain\920d8617-a6e0-49a6-94f3-590eb96bce19\scratch\cube_test_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Joint names (Arm A only, matching dataset order)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# ================================================================
# Setup logging
# ================================================================
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("cube_test")

# ================================================================
# Load dataset normalization stats
# ================================================================
log.info("Loading dataset normalization stats...")
stats_path = hf_hub_download(repo_id=DATASET_REPO, filename="meta/stats.json", repo_type="dataset")
with open(stats_path) as f:
    raw_stats = json.load(f)

state_mean = torch.tensor(raw_stats["observation.state"]["mean"], dtype=torch.float32)
state_std = torch.tensor(raw_stats["observation.state"]["std"], dtype=torch.float32)
action_mean = torch.tensor(raw_stats["action"]["mean"], dtype=torch.float32)
action_std = torch.tensor(raw_stats["action"]["std"], dtype=torch.float32)

# Image normalization: use dataset stats as-is (NOT ImageNet).
# The z-score range will be wild on sim renders, but this is what the
# policy was trained with. Switching to ImageNet stats is an untested
# experiment — noted but not applied.
img_top_mean = torch.tensor([raw_stats["observation.images.top"]["mean"][c][0][0] for c in range(3)])
img_top_std = torch.tensor([raw_stats["observation.images.top"]["std"][c][0][0] for c in range(3)])
img_wrist_mean = torch.tensor([raw_stats["observation.images.wrist"]["mean"][c][0][0] for c in range(3)])
img_wrist_std = torch.tensor([raw_stats["observation.images.wrist"]["std"][c][0][0] for c in range(3)])

log.info(f"  State mean: {state_mean.tolist()}")
log.info(f"  Action mean: {action_mean.tolist()}")
log.info(f"  Image top mean (RGB): {img_top_mean.tolist()}")

# ================================================================
# Load MuJoCo scene
# ================================================================
log.info("\nLoading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Identify actuator IDs for Arm A
arm_a_act_ids = []
arm_a_ctrlranges = []
for i in range(model.nu):
    name = model.actuator(i).name
    if name.startswith("armA_"):
        arm_a_act_ids.append(i)
        arm_a_ctrlranges.append((model.actuator(i).ctrlrange[0], model.actuator(i).ctrlrange[1]))

# Identify joint qpos addresses for Arm A state reading
arm_a_qpos_addrs = []
for act_id in arm_a_act_ids:
    joint_id = model.actuator(act_id).trnid[0]
    arm_a_qpos_addrs.append(model.jnt_qposadr[joint_id])

# Identify actuator IDs and joint qpos addresses for Arm B
arm_b_act_ids = []
arm_b_qpos_addrs = []
for i in range(model.nu):
    name = model.actuator(i).name
    if name.startswith("armB_"):
        arm_b_act_ids.append(i)
        joint_id = model.actuator(i).trnid[0]
        arm_b_qpos_addrs.append(model.jnt_qposadr[joint_id])

def hold_arm_b():
    """Hold Arm B joint positions as ctrl every step so it genuinely stays at rest."""
    for act_id, qpos_addr in zip(arm_b_act_ids, arm_b_qpos_addrs):
        data.ctrl[act_id] = data.qpos[qpos_addr]

# Initialize Arm B ctrl immediately to rest qpos
hold_arm_b()

# Camera IDs
cam_overhead = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
cam_wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "armA_wrist_cam")

# Cube and spoon bodies for tracking
cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cube")
spoon_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spoon")

# Create renderers
renderer_top = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
renderer_wrist = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

log.info(f"  Arm A actuator IDs: {arm_a_act_ids}")
log.info(f"  Arm A qpos addrs: {arm_a_qpos_addrs}")
log.info(f"  Arm B actuator IDs: {arm_b_act_ids}")
log.info(f"  Arm B qpos addrs: {arm_b_qpos_addrs}")
log.info(f"  Camera overhead ID: {cam_overhead}, wrist ID: {cam_wrist}")
log.info(f"  Cube body ID: {cube_body_id}")
log.info(f"  Spoon body ID: {spoon_body_id}")

# Record initial positions
cube_pos_initial = data.xpos[cube_body_id].copy()
spoon_pos_initial = data.xpos[spoon_body_id].copy()
log.info(f"  Cube initial pos:  {cube_pos_initial.tolist()}")
log.info(f"  Spoon initial pos: {spoon_pos_initial.tolist()}")

# ================================================================
# Load ACT Policy
# ================================================================
log.info("\nLoading ACT policy...")
from lerobot.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained(REPO_ID, device="cpu")
policy.eval()
log.info("  Policy loaded on CPU.")

# ================================================================
# Helper functions
# ================================================================
def get_arm_a_state():
    """Read Arm A joint positions (6-dim, radians)."""
    return np.array([data.qpos[addr] for addr in arm_a_qpos_addrs], dtype=np.float32)

def render_camera(renderer, cam_id):
    """Render camera, return float32 tensor (3, H, W) in [0, 1]."""
    renderer.update_scene(data, camera=cam_id)
    frame = renderer.render()  # uint8 (H, W, 3)
    frame_float = frame.astype(np.float32) / 255.0
    frame_chw = np.transpose(frame_float, (2, 0, 1))  # (3, H, W)
    return torch.from_numpy(frame_chw)

def normalize_image(img_chw, mean, std):
    """Per-channel normalization: (img - mean) / std."""
    # mean/std are (3,), img is (3, H, W)
    return (img_chw - mean.view(3, 1, 1)) / std.view(3, 1, 1)

def normalize_state(state_raw):
    """z-score normalize state: (x - mean) / std."""
    return (torch.from_numpy(state_raw) - state_mean) / state_std

def denormalize_action(action_z):
    """Denormalize z-score action to radians: x * std + mean."""
    return action_z * action_std + action_mean

def apply_action_with_clamping(action_rad, step_idx):
    """Apply 6-dim action to Arm A actuators, logging any clamping."""
    clamp_events = []
    for j in range(6):
        val = action_rad[j].item()
        lo, hi = arm_a_ctrlranges[j]
        clamped = np.clip(val, lo, hi)
        if abs(clamped - val) > 1e-6:
            clamp_events.append({
                "step": step_idx,
                "joint": JOINT_NAMES[j],
                "commanded": val,
                "clamped": clamped,
                "delta": abs(val - clamped)
            })
        data.ctrl[arm_a_act_ids[j]] = clamped
    return clamp_events

# ================================================================
# Pre-episode: log raw select_action output range for dummy obs
# (Phase 3 requirement: action range sanity check)
# ================================================================
log.info("\n--- Pre-Episode: Action Range Sanity Check ---")
state_raw = get_arm_a_state()
state_z = normalize_state(state_raw)
img_top = render_camera(renderer_top, cam_overhead)
img_wrist = render_camera(renderer_wrist, cam_wrist)
img_top_norm = normalize_image(img_top, img_top_mean, img_top_std)
img_wrist_norm = normalize_image(img_wrist, img_wrist_mean, img_wrist_std)

obs = {
    "observation.state": state_z.unsqueeze(0),
    "observation.images.top": img_top_norm.unsqueeze(0),
    "observation.images.wrist": img_wrist_norm.unsqueeze(0),
}

with torch.inference_mode():
    policy.reset()
    action_z = policy.select_action(obs)

action_rad = denormalize_action(action_z[0])
log.info(f"  Raw z-score output: {action_z[0].tolist()}")
log.info(f"  Denormalized (rad): {action_rad.tolist()}")
log.info(f"  z-score range: [{action_z.min().item():.3f}, {action_z.max().item():.3f}]")
log.info(f"  Radian range:  [{action_rad.min().item():.3f}, {action_rad.max().item():.3f}]")

for j in range(6):
    lo, hi = arm_a_ctrlranges[j]
    v = action_rad[j].item()
    status = "OK" if lo <= v <= hi else "CLAMP"
    log.info(f"    {JOINT_NAMES[j]:<18} val={v:>8.4f}  ctrl=[{lo:.4f}, {hi:.4f}]  {status}")

# Save initial frame
import PIL.Image
frame_init = renderer_top.render()
PIL.Image.fromarray(frame_init).save(str(OUT_DIR / "frame_000_initial.png"))
log.info(f"\n  Saved initial frame to {OUT_DIR / 'frame_000_initial.png'}")

# ================================================================
# Run Episode
# ================================================================
log.info("\n" + "=" * 65)
log.info("RUNNING CUBE PICK-PLACE EPISODE")
log.info(f"  {EPISODE_STEPS} control steps at {CONTROL_HZ}Hz")
log.info(f"  {ACTION_REPEAT} physics steps per control step at {PHYSICS_HZ}Hz")
log.info("=" * 65)

all_clamp_events = []
cube_positions = [cube_pos_initial.copy()]
max_cube_height = cube_pos_initial[2]

# Reset policy state
with torch.inference_mode():
    policy.reset()
    
    for step in range(EPISODE_STEPS):
        # 1. Get observation
        state_raw = get_arm_a_state()
        state_z = normalize_state(state_raw)
        img_top = render_camera(renderer_top, cam_overhead)
        img_wrist = render_camera(renderer_wrist, cam_wrist)
        img_top_norm = normalize_image(img_top, img_top_mean, img_top_std)
        img_wrist_norm = normalize_image(img_wrist, img_wrist_mean, img_wrist_std)

        obs = {
            "observation.state": state_z.unsqueeze(0),
            "observation.images.top": img_top_norm.unsqueeze(0),
            "observation.images.wrist": img_wrist_norm.unsqueeze(0),
        }

        # 2. Get action from policy
        action_z = policy.select_action(obs)
        action_rad = denormalize_action(action_z[0])

        # 3. Apply action with clamp logging
        clamp_events = apply_action_with_clamping(action_rad, step)
        all_clamp_events.extend(clamp_events)

        # 4. Step physics (action repeat) while holding Arm B at its rest qpos
        for _ in range(ACTION_REPEAT):
            hold_arm_b()
            mujoco.mj_step(model, data)

        # 5. Track cube and spoon positions
        cube_pos = data.xpos[cube_body_id].copy()
        cube_positions.append(cube_pos)
        max_cube_height = max(max_cube_height, cube_pos[2])

        spoon_pos = data.xpos[spoon_body_id].copy()
        spoon_disp = np.linalg.norm(spoon_pos - spoon_pos_initial)

        # 6. Periodic logging
        if step % 50 == 0 or step == EPISODE_STEPS - 1:
            cube_disp = np.linalg.norm(cube_pos - cube_pos_initial)
            log.info(f"  Step {step:>4d}: cube_pos={cube_pos.tolist()}, disp={cube_disp:.4f}m, height={cube_pos[2]:.4f}m | spoon_disp={spoon_disp:.4f}m")
            # Save frame
            renderer_top.update_scene(data, camera=cam_overhead)
            frame = renderer_top.render()
            PIL.Image.fromarray(frame).save(str(OUT_DIR / f"frame_{step:03d}.png"))

# ================================================================
# Results
# ================================================================
cube_pos_final = data.xpos[cube_body_id].copy()
cube_displacement = np.linalg.norm(cube_pos_final - cube_pos_initial)
cube_lifted = max_cube_height > cube_pos_initial[2] + 0.01  # >1cm lift

spoon_pos_final = data.xpos[spoon_body_id].copy()
spoon_displacement = np.linalg.norm(spoon_pos_final - spoon_pos_initial)

log.info("\n" + "=" * 65)
log.info("CUBE TEST RESULTS")
log.info("=" * 65)
log.info(f"  Initial cube pos:  {cube_pos_initial.tolist()}")
log.info(f"  Final cube pos:    {cube_pos_final.tolist()}")
log.info(f"  Cube displacement: {cube_displacement:.4f} m")
log.info(f"  Max cube height:   {max_cube_height:.4f} m (initial={cube_pos_initial[2]:.4f})")
log.info(f"  Cube lifted >1cm:  {'YES' if cube_lifted else 'NO'}")
log.info(f"  Initial spoon pos: {spoon_pos_initial.tolist()}")
log.info(f"  Final spoon pos:   {spoon_pos_final.tolist()}")
log.info(f"  Spoon displacement:{spoon_displacement:.4f} m (undisturbed: {'YES' if spoon_displacement < 0.01 else 'NO'})")

# Clamp summary
log.info(f"\n  Total clamp events: {len(all_clamp_events)}")
if all_clamp_events:
    # Group by joint
    from collections import Counter
    joint_counts = Counter(e["joint"] for e in all_clamp_events)
    log.info(f"  Clamp events by joint:")
    for joint, count in joint_counts.most_common():
        max_delta = max(e["delta"] for e in all_clamp_events if e["joint"] == joint)
        log.info(f"    {joint}: {count} events, max_delta={max_delta:.4f} rad")
    
    # Check if clamping correlated with gripper actions (grasp attempts)
    gripper_clamps = [e for e in all_clamp_events if e["joint"] == "gripper"]
    if gripper_clamps:
        log.info(f"\n  !! Gripper clamping at steps: {[e['step'] for e in gripper_clamps[:10]]}")
        log.info(f"     This may have prevented successful grasping.")

# Verdict
if cube_lifted:
    log.info("  VERDICT: CUBE LIFTED — pipeline plumbing works. Proceed to plate test.")
elif cube_displacement > 0.01:
    log.info("  VERDICT: CUBE MOVED but not lifted — policy is producing non-trivial actions.")
    log.info("  Likely issues: camera framing mismatch, normalization domain gap,")
    log.info("  or insufficient episode length for this checkpoint's strategy.")
else:
    log.info("  VERDICT: CUBE DID NOT MOVE — policy output is not producing meaningful motion.")
    log.info("  This is a plumbing/domain-gap failure, not object generalization.")
    log.info("  Possible causes:")
    log.info("    1. Visual domain gap (sim renders vs real camera, z-scores up to ±22)")
    log.info("    2. Camera framing mismatch (overhead position/FOV)")
    log.info("    3. Normalization not correctly wired")

# Save final frame
renderer_top.update_scene(data, camera=cam_overhead)
frame_final = renderer_top.render()
PIL.Image.fromarray(frame_final).save(str(OUT_DIR / "frame_final.png"))

# Cleanup
renderer_top.close()
renderer_wrist.close()

log.info(f"\n  Frames saved to: {OUT_DIR}")
log.info("=== Cube Test Complete ===")
