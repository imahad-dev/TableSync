"""
TableSync -- Static Hero / Cover Shot Renderer
================================================
Renders a purely cosmetic, static cover image for publications / LinkedIn.
No physics, no dynamic bodies, no controller. Just the wooden table,
the realistic ceramic plate, and the realistic spoon (v2) lying naturally
flat on the tabletop with its bowl fully visible.
"""

from pathlib import Path
import mujoco
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_DIR = PROJECT_ROOT / "scene"
OUTPUT_DIR = PROJECT_ROOT / "eval"

def render_cover_shot(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = OUTPUT_DIR / "tablesync_cover_hero.png"

    xml_content = """
    <mujoco model="tablesync_cover_shot">
      <compiler angle="radian" meshdir="assets" texturedir="assets" />
      <visual>
        <global offwidth="1920" offheight="1080" />
        <quality shadowsize="4096" />
      </visual>
      <asset>
        <mesh name="plate_mesh" file="plate_realistic.obj" />
        <mesh name="spoon_mesh" file="spoon_realistic_v2.obj" />
        <texture name="skybox" type="skybox" builtin="gradient"
                 rgb1="0.80 0.82 0.85" rgb2="0.30 0.32 0.38" width="512" height="512" />
        <texture name="wood_table" type="2d" file="wood_table_texture.png" />
        <material name="wood_table_mat" texture="wood_table" texrepeat="3 3"
                  specular="0.3" shininess="0.2" reflectance="0.05" />
        <material name="ceramic_plate_mat" rgba="0.97 0.96 0.93 1"
                  specular="0.7" shininess="0.6" reflectance="0.1" />
        <material name="steel_spoon_mat" rgba="0.82 0.83 0.85 1"
                  specular="0.75" shininess="0.6" reflectance="0.3" />
      </asset>
      <worldbody>
        <light pos="0.0 -0.15 0.95" dir="0 0.15 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3" directional="false" />
        <light pos="-0.4 0.1 0.8" dir="0.4 -0.1 -1" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1" directional="false" />
        
        <!-- Table -->
        <body name="table" pos="0 0.06 0">
          <geom type="box" size="0.32 0.22 0.02" pos="0 0 0.18" material="wood_table_mat" />
        </body>

        <!-- Ceramic Plate resting naturally on tabletop (surface z=0.20m + 1mm offset) -->
        <geom name="plate_visual" type="mesh" mesh="plate_mesh"
              pos="-0.08 0.06 0.201" material="ceramic_plate_mat" />

        <!-- Spoon resting naturally flat on tabletop beside plate (bowl clearly visible) -->
        <geom name="spoon_visual" type="mesh" mesh="spoon_mesh"
              pos="0.08 0.08 0.205" euler="0.10 -0.05 2.35" material="steel_spoon_mat" />
      </worldbody>
    </mujoco>
    """

    tmp_xml = SCENE_DIR / "_cover_shot_temp.xml"
    tmp_xml.write_text(xml_content, encoding="utf-8")

    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=1080, width=1920)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat[:] = [0.0, 0.06, 0.20]
        cam.distance = 0.58
        cam.elevation = -32.0
        cam.azimuth = 90.0

        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        img.save(str(output_path))
        renderer.close()
        print(f"Cover shot rendered to: {output_path} ({img.size[0]}x{img.size[1]})")
    finally:
        if tmp_xml.exists():
            tmp_xml.unlink()

    return output_path

if __name__ == "__main__":
    render_cover_shot()
