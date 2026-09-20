"""
TableSync — Dual-Arm Scene Composer
====================================
so101_nexus ships a single-arm SO-101 MuJoCo model (EnvironmentConfig only
accepts one RobotConfig — verified against the installed package). There is
no bimanual env out of the box, so this script builds one: it takes the
stock so101.xml, deep-copies the arm body/joint/site/geom/actuator tree
twice with a per-arm name prefix (armA_ / armB_), places each copy at a
mirrored base pose facing a shared table, and adds a plate + spoon as
free-jointed bodies plus an overhead camera.

Grounded facts used here (checked directly against the installed package,
not assumed):
  - source model: so101_nexus.get_so101_mujoco_model_path() -> so101.xml
  - named elements needing a prefix: body, joint, site, geom, and each
    <position> actuator's `name` + `joint` attrs (actuators are <position>
    tags inside <actuator>, not a tag literally called "actuator")
  - meshes carry only a `file` attribute, no `name` -> the <asset> block
    is shared as-is, unmodified, between both arm copies
  - top-level body is a single <body name="base"> per arm

This produces tablesync_scene.xml and load-tests it in MuJoCo. Base-pose
offsets below are a first guess (SO-101 is a short-reach desktop arm on
STS3215 servos) and are meant to be tuned once we can visualize the scene
render — not treated as final.
"""

from __future__ import annotations
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

import so101_nexus as s101

SOURCE_XML = s101.get_so101_mujoco_model_path()
SCENE_DIR = Path(__file__).resolve().parent
OUT_XML = SCENE_DIR / "tablesync_scene.xml"

# Realistic visual asset paths
PLATE_OBJ = (SCENE_DIR / "plate_realistic.obj").resolve()
SPOON_OBJ = (SCENE_DIR / "spoon_realistic_v2.obj").resolve()
TABLE_TEX = (SCENE_DIR / "wood_table_texture.png").resolve()

# First-guess base poses. SO-101 is a short-reach desktop arm; bases are
# placed close together so the handoff point sits inside both workspaces.
# TUNE THESE once we can render + check reach — do not treat as final.
ARM_A_POS = (-0.16, -0.12, 0.20)
ARM_A_EULER = (0, 0, 1.5708)   # facing +y, toward the shared table
ARM_B_POS = (0.16, -0.12, 0.20)
ARM_B_EULER = (0, 0, 1.5708)   # SAME yaw as A -- both arms sit side-by-side
                               # facing the same table, clamped to tabletop level

NAMED_TAGS = {"body", "joint", "site", "geom", "camera"}


def prefix_names(elem: ET.Element, prefix: str) -> None:
    """Recursively prefix name= on body/joint/site/geom, in place."""
    for e in elem.iter():
        if e.tag in NAMED_TAGS and "name" in e.attrib:
            e.attrib["name"] = prefix + e.attrib["name"]


def prefix_actuators(actuator_section: ET.Element, prefix: str) -> ET.Element:
    """Return a deep-copied <actuator> section with name/joint prefixed."""
    new_section = copy.deepcopy(actuator_section)
    for a in new_section:
        if "name" in a.attrib:
            a.attrib["name"] = prefix + a.attrib["name"]
        if "joint" in a.attrib:
            a.attrib["joint"] = prefix + a.attrib["joint"]
    return new_section


def build_scene() -> ET.ElementTree:
    src_tree = ET.parse(SOURCE_XML)
    src_root = src_tree.getroot()

    out_root = ET.Element("mujoco", {"model": "tablesync_dual_arm"})

    # Ensure scene/assets directory has all mesh and texture assets for portability
    assets_dir = SCENE_DIR / "assets"
    assets_dir.mkdir(exist_ok=True)
    src_assets_dir = Path(SOURCE_XML).parent / "assets"
    if src_assets_dir.exists():
        for stl in src_assets_dir.glob("*.stl"):
            dest_stl = assets_dir / stl.name
            if not dest_stl.exists():
                shutil.copy2(stl, dest_stl)

    for custom_asset in (PLATE_OBJ, SPOON_OBJ, TABLE_TEX):
        dest_custom = assets_dir / custom_asset.name
        if not dest_custom.exists():
            shutil.copy2(custom_asset, dest_custom)

    # Shared sections: compiler / option / visual / default / asset
    for tag in ("compiler", "option", "visual", "default", "asset"):
        section = src_root.find(tag)
        if section is not None:
            section_copy = copy.deepcopy(section)
            if tag == "compiler":
                # Use portable relative paths inside scene/
                section_copy.attrib["meshdir"] = "assets"
                section_copy.attrib["texturedir"] = "assets"
            out_root.append(section_copy)

    # Augment asset section with realistic visual meshes, textures, materials, and skybox
    asset_section = out_root.find("asset")
    if asset_section is not None:
        ET.SubElement(asset_section, "mesh", {"name": "plate_mesh", "file": "plate_realistic.obj"})
        ET.SubElement(asset_section, "mesh", {"name": "spoon_mesh", "file": "spoon_realistic_v2.obj"})
        ET.SubElement(asset_section, "texture", {
            "name": "skybox", "type": "skybox", "builtin": "gradient",
            "rgb1": "0.55 0.6 0.68", "rgb2": "0.12 0.13 0.18", "width": "512", "height": "512"
        })
        ET.SubElement(asset_section, "texture", {
            "name": "wood_table", "type": "2d", "file": "wood_table_texture.png"
        })
        ET.SubElement(asset_section, "material", {
            "name": "wood_table_mat", "texture": "wood_table", "texrepeat": "3 3",
            "specular": "0.3", "shininess": "0.2", "reflectance": "0.05"
        })
        ET.SubElement(asset_section, "material", {
            "name": "ceramic_plate_mat", "rgba": "0.97 0.96 0.93 1",
            "specular": "0.7", "shininess": "0.6", "reflectance": "0.1"
        })
        ET.SubElement(asset_section, "material", {
            "name": "steel_spoon_mat", "rgba": "0.72 0.73 0.75 1",
            "specular": "0.55", "shininess": "0.5", "reflectance": "0.25"
        })

    worldbody = ET.SubElement(out_root, "worldbody")

    # Table (fixed) — sized with clearance so front edge does not collide with arm bases
    table = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0 0.06 0"})
    ET.SubElement(table, "geom", {
        "type": "box", "size": "0.32 0.22 0.02", "pos": "0 0 0.18",
        "material": "wood_table_mat",
    })

    # Plate — free body with a 10mm graspable rim lip
    plate = ET.SubElement(worldbody, "body", {"name": "plate", "pos": "-0.08 0.06 0.203"})
    ET.SubElement(plate, "joint", {"name": "plate_free", "type": "free"})
    # Existing collision geoms kept identical for physics, set to invisible rgba="1 1 1 0"
    ET.SubElement(plate, "geom", {
        "name": "plate_dish", "type": "cylinder", "size": "0.050 0.003", "pos": "0 0 0.003",
        "rgba": "1 1 1 0", "mass": "0.05", "friction": "1.5 0.01 0.001",
    })
    ET.SubElement(plate, "geom", {
        "name": "plate_rim_edge", "type": "box", "size": "0.005 0.015 0.015", "pos": "-0.045 0 0.015",
        "rgba": "1 1 1 0", "mass": "0.02", "condim": "6", "friction": "3.0 0.05 0.005",
    })
    # Non-colliding visual mesh offset 1mm above bottom surface (z=0.000) with explicit mass="0"
    ET.SubElement(plate, "geom", {
        "name": "plate_visual", "type": "mesh", "mesh": "plate_mesh",
        "pos": "0 0 0.001", "material": "ceramic_plate_mat",
        "contype": "0", "conaffinity": "0", "mass": "0",
    })

    # Spoon — free body, stable base with 16mm diameter handle
    spoon = ET.SubElement(worldbody, "body", {"name": "spoon", "pos": "0.08 0.06 0.203"})
    ET.SubElement(spoon, "joint", {"name": "spoon_free", "type": "free"})
    # Existing collision geoms kept identical for physics, set to invisible rgba="1 1 1 0"
    ET.SubElement(spoon, "geom", {
        "name": "spoon_base", "type": "cylinder", "size": "0.018 0.003", "pos": "0 0 0.003",
        "rgba": "1 1 1 0", "mass": "0.04", "friction": "2.0 0.01 0.001",
    })
    ET.SubElement(spoon, "geom", {
        "name": "spoon_handle", "type": "capsule", "size": "0.008 0.030", "pos": "0 0 0.033",
        "rgba": "1 1 1 0", "mass": "0.02", "condim": "6", "friction": "3.0 0.05 0.005",
    })
    # Non-colliding visual mesh offset 1mm above bottom surface (z=0.000) with explicit mass="0"
    # euler="0 -1.5708 0" aligns the handle vertically with the bowl at the top
    ET.SubElement(spoon, "geom", {
        "name": "spoon_visual", "type": "mesh", "mesh": "spoon_mesh",
        "pos": "0 0 0.001", "euler": "0 -1.5708 0", "material": "steel_spoon_mat",
        "contype": "0", "conaffinity": "0", "mass": "0",
    })

    # Main scene illumination — can be perturbed in robustness harness
    ET.SubElement(worldbody, "light", {
        "name": "scene_light", "pos": "0.0 -0.05 0.95", "dir": "0 0.05 -1",
        "diffuse": "0.8 0.8 0.8", "specular": "0.2 0.2 0.2", "directional": "false",
    })

    # Overhead camera — feeds both the reasoning layer and the policy
    ET.SubElement(worldbody, "camera", {
        "name": "overhead", "pos": "0 -0.05 0.85", "euler": "0 0 0",
        "fovy": "60", "mode": "fixed",
    })
    # Angled camera — natural 3D demo perspective
    ET.SubElement(worldbody, "camera", {
        "name": "angled", "pos": "0.0 -0.45 0.56", "euler": "0.56 0 0",
        "fovy": "55", "mode": "fixed",
    })

    # Two arm copies
    src_worldbody = src_root.find("worldbody")
    src_actuator = src_root.find("actuator")
    for prefix, pos, euler in (
        ("armA_", ARM_A_POS, ARM_A_EULER),
        ("armB_", ARM_B_POS, ARM_B_EULER),
    ):
        arm_body = copy.deepcopy(src_worldbody.find("body"))  # the single <body name="base">
        prefix_names(arm_body, prefix)
        # Make camera boxes visual-only so wrist movement is not blocked by collision
        for geom in arm_body.iter("geom"):
            g_name = geom.attrib.get("name", "")
            if "camera_box" in g_name:
                geom.attrib["contype"] = "0"
                geom.attrib["conaffinity"] = "0"
        mount = ET.SubElement(worldbody, "body", {
            "name": f"{prefix}mount",
            "pos": f"{pos[0]} {pos[1]} {pos[2]}",
            "euler": f"{euler[0]} {euler[1]} {euler[2]}",
        })
        mount.append(arm_body)

    if src_actuator is not None:
        out_root.append(prefix_actuators(src_actuator, "armA_"))
        out_root.append(prefix_actuators(src_actuator, "armB_"))

    return ET.ElementTree(out_root)


if __name__ == "__main__":
    tree = build_scene()
    ET.indent(tree, space="  ")
    tree.write(OUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUT_XML}")

    # Load-test: does MuJoCo actually accept this composed model?
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(OUT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"LOAD OK — nq={model.nq} nu={model.nu} nbody={model.nbody}")
    print("Actuator names:", [model.actuator(i).name for i in range(model.nu)])
    print("Camera names:", [model.camera(i).name for i in range(model.ncam)])
