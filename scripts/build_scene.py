"""Generate the D07 scene and a minimally adapted copy of the official robot."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from simple_arm import replace_arm

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "scenes"


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attrs.items()})


def save(tree, path):
    ET.indent(tree, space="  ")
    ET.ElementTree(tree).write(path, encoding="utf-8", xml_declaration=True)


def main():
    source = ROOT / "models/stanford_tidybot/tidybot.xml"
    if not source.exists():
        raise SystemExit("Run python scripts/fetch_tidybot.py first")
    SCENES.mkdir(exist_ok=True)
    robot = ET.parse(source).getroot()
    robot.remove(robot.find("keyframe"))
    robot.remove(robot.find("compiler"))
    for mesh in robot.findall("asset/mesh"):
        mesh.set("file", "../models/stanford_tidybot/assets/" + mesh.get("file"))
    replace_arm(robot, mount_height=0.16)
    # Scale BOTH visual and collision meshes. MuJoCo recomputes mass properties from these meshes.
    used_meshes = {geom.get("mesh") for geom in robot.findall(".//geom") if geom.get("mesh")}
    for mesh in list(robot.find("asset")):
        if mesh.tag == "mesh":
            name = mesh.get("name", Path(mesh.get("file")).stem)
            if name not in used_meshes:
                robot.find("asset").remove(mesh)
            else:
                mesh.set("scale", "1 1 0.5")
    # Provisional sensor mounting aimed at the protruding handle.
    base_camera = robot.find(".//camera[@name='base']")
    base_camera.attrib.clear()
    base_camera.attrib.update(name="base", pos="0.28 0 0.24", xyaxes="0 -1 0 -0.11 0 0.994",
                              fovy="60", resolution="640 480")
    base = robot.find("worldbody/body[@name='base_link']")
    element(base, "geom", name="camera_post", type="box", pos="0.24 0 0.20", size="0.009 0.009 0.035",
            mass="0.02", rgba="0.15 0.20 0.23 1")
    element(base, "geom", name="camera_housing", type="box", pos="0.25 0 0.24", size="0.025 0.035 0.02",
            mass="0.05", rgba="0.15 0.20 0.23 1")
    element(base, "geom", name="camera_lens", type="sphere", pos="0.277 0 0.24", size="0.008",
            mass="0", rgba="0.15 0.50 0.75 1", contype="0", conaffinity="0")
    robot_qpos = ["0"] * 7 + ["0.02", "0.02"]
    robot_ctrl = "0 0 0 0 0 0 0 0.02"
    save(robot, SCENES / "tidybot_robot.xml")

    scene = ET.Element("mujoco", model="FishingBot simple arm and toy bin")
    element(scene, "include", file="tidybot_robot.xml")
    element(scene, "compiler", angle="radian", autolimits="true")
    element(scene, "option", timestep="0.001", gravity="0 0 -9.81", iterations="100")
    element(scene, "statistic", center="0.35 0 0.25", extent="1.8")
    visual = element(scene, "visual")
    element(visual, "global", offwidth="1280", offheight="960", azimuth="135", elevation="-25")
    element(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.65 0.65 0.65")
    asset = element(scene, "asset")
    element(asset, "texture", type="skybox", builtin="gradient", rgb1="0.16 0.20 0.26", rgb2="0.34 0.40 0.47",
            width="512", height="3072")
    element(asset, "texture", name="floor_texture", type="2d", builtin="checker", width="512", height="512",
            rgb1="0.88 0.90 0.92", rgb2="0.76 0.80 0.84")
    element(asset, "material", name="floor_material", texture="floor_texture", texrepeat="6 6", texuniform="true")
    world = element(scene, "worldbody")
    element(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    element(world, "camera", name="overview", pos="1.3 -1.6 1.1", xyaxes="0.872 0.490 0 -0.196 0.349 0.918")
    # A finite floor makes the 3 x 3 m work area visible; the blue boundary is visual only.
    element(world, "geom", name="floor", type="box", pos="0 0 -0.05", size="1.5 1.5 0.05",
            material="floor_material", friction="0.8 0.01 0.001")
    for name, pos, size in (
        ("north", "0 1.49 0.002", "1.49 0.008 0.002"),
        ("south", "0 -1.49 0.002", "1.49 0.008 0.002"),
        ("east", "1.49 0 0.002", "0.008 1.49 0.002"),
        ("west", "-1.49 0 0.002", "0.008 1.49 0.002")):
        element(world, "geom", name="boundary_" + name, type="box", pos=pos, size=size,
                rgba="0.1 0.45 0.8 1", contype="0", conaffinity="0")

    # Open-top bin: floor and 24 separate panels, rather than a solid cylinder.
    bin_body = element(world, "body", name="toy_bin", pos="0.75 0 0")
    element(bin_body, "geom", name="bin_bottom", type="cylinder", pos="0 0 0.02", size="0.075 0.02", rgba="0.20 0.29 0.34 1")
    for index in range(24):
        angle = index * 2 * np.pi / 24
        element(bin_body, "geom", name=f"bin_wall_{index:02d}", type="box",
                pos=f"{0.0703*np.cos(angle):.8g} {0.0703*np.sin(angle):.8g} 0.15",
                size="0.004 0.0095 0.11", euler=f"0 0 {angle:.8g}", rgba="0.26 0.39 0.46 1")
    # A passive internal sleeve supports the upright shaft; the rod still lifts out freely.
    for name, pos, size in (("xp", "0.011 0 0.11", "0.005 0.016 0.07"),
                            ("xn", "-0.011 0 0.11", "0.005 0.016 0.07"),
                            ("yp", "0 0.011 0.15", "0.006 0.005 0.03"),
                            ("yn", "0 -0.011 0.11", "0.006 0.005 0.07")):
        element(bin_body, "geom", name=f"bin_guide_{name}", type="box", pos=pos, size=size, rgba="0.20 0.29 0.34 1")

    rod_quat = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0, 0])
    rod = element(world, "body", name="fishing_rod", pos="0.75 0 0.29", quat=" ".join(map(str, rod_quat)))
    element(rod, "freejoint", name="rod_free")
    element(rod, "geom", name="rod_handle", type="cylinder", fromto="0 -0.05 0 0 0.05 0", size="0.015",
            mass="0.045", rgba="0.95 0.32 0.12 1", friction="1 0.01 0.001")
    element(rod, "geom", name="rod_shaft", type="cylinder", fromto="0 0.05 0 0 0.25 0", size="0.004",
            mass="0.025", rgba="0.18 0.24 0.30 1")
    # Overlay marker shares the handle surface, has no mass or collision geometry.
    element(rod, "geom", name="handle_marker", type="cylinder", fromto="0 -0.015 0 0 0.015 0", size="0.0151",
            mass="0", rgba="0.10 0.85 0.20 1", contype="0", conaffinity="0")
    element(rod, "site", name="handle_grasp", pos="0 0 0", size="0.003", rgba="1 1 0 0.7")

    # shortcut: rigid 2 cm links approximate string bending, not real cord elasticity or knotting.
    parent = rod
    parent_quat = rod_quat
    for index in range(10):
        # Start the string folded UP inside the bin; gravity lets it settle naturally.
        if index < 2:
            direction = np.array([0, 1, 1], dtype=float) / np.sqrt(2)
        else:
            # Fold the same 20 cm string into a narrow rising spiral inside the 15 cm bin.
            radius, step = 0.02 * np.sqrt(2), 0.43
            angle = (index - 2) * step
            dx = radius * (np.sin(angle + step) - np.sin(angle))
            dy = radius * (np.cos(angle + step) - np.cos(angle))
            direction = np.array([dx, dy, np.sqrt(0.02**2 - dx**2 - dy**2)]) / 0.02
        world_quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(world_quat, -direction)
        inverse = parent_quat.copy(); inverse[1:] *= -1
        relative = np.zeros(4)
        mujoco.mju_mulQuat(relative, inverse, world_quat)
        parent = element(parent, "body", name=f"string_{index:02d}", pos="0 0.25 0" if index == 0 else "0 0 -0.02",
                         quat=" ".join(map(str, relative)))
        parent_quat = world_quat
        element(parent, "joint", name=f"string_joint_{index:02d}", type="ball", damping="0.00002")
        element(parent, "geom", name=f"string_geom_{index:02d}", type="capsule", fromto="0 0 0 0 0 -0.02", size="0.0015",
                mass="0.001", rgba="0.85 0.70 0.20 1", friction="0.5 0.005 0.0001",
                priority="1", solref="0.004 1", solimp="0.99 0.999 0.001")
    toy = element(parent, "body", name="toy_ball", pos="0 0 -0.02")
    element(toy, "geom", name="toy_ball_geom", type="sphere", pos="0 0 -0.025", size="0.025", mass="0.020",
            rgba="0.85 0.18 0.42 1", friction="0.7 0.01 0.001", priority="1", solref="0.004 1")
    element(toy, "site", name="toy_center", pos="0 0 -0.025", size="0.002")

    sensors = element(scene, "sensor")
    for side in ("left", "right"):
        element(sensors, "touch", name=f"{side}_grip_force", site=f"{side}_grip_touch")
    for name, objtype, objname in (("handle_world", "site", "handle_grasp"), ("toy_world", "site", "toy_center")):
        element(sensors, "framepos", name=name, objtype=objtype, objname=objname)
    # Runtime uses named finger contacts to check both sides; no forced attachment/weld.
    custom = element(scene, "custom")
    element(custom, "numeric", name="workspace_xy", data="-1.5 1.5 -1.5 1.5")
    element(custom, "numeric", name="rod_mass_target_kg", data="0.10")
    element(custom, "text", name="layout_status", data="Half-height base, custom 4-axis arm, open toy bin; grasp reach not validated")
    element(custom, "text", name="position_sensors", data="Ground truth for diagnostics, not camera perception")
    keys = element(scene, "keyframe")
    rod_qpos = ["0.75", "0", "0.29"] + list(map(str, rod_quat))
    string_qpos = ["1", "0", "0", "0"] * 10
    element(keys, "key", name="home", qpos=" ".join(robot_qpos + rod_qpos + string_qpos), ctrl=robot_ctrl)
    save(scene, SCENES / "fishingbot.xml")
    print("Generated scenes/fishingbot.xml and scenes/tidybot_robot.xml")


if __name__ == "__main__":
    main()
