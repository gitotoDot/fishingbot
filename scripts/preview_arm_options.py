"""Render two design alternatives without changing the current FishingBot scene."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/arm-options"
KEEP = {"joint_1", "joint_2", "joint_4", "joint_7"}


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def save(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def prepare_robot():
    robot = ET.parse(ROOT / "models/stanford_tidybot/tidybot.xml").getroot()
    robot.remove(robot.find("compiler"))
    robot.remove(robot.find("keyframe"))
    for side in ("left", "right"):
        add(robot.find(f".//body[@name='{side}_pad']"), "site", name=f"{side}_grip_touch", type="box", size="0.02 0.02 0.04", rgba="0 0 0 0")
    for mesh in robot.findall("asset/mesh"):
        mesh.set("file", (ROOT / "models/stanford_tidybot/assets" / mesh.get("file")).resolve().as_posix())
    return robot


def lock_joints(robot, original, original_data):
    for body in robot.findall(".//body"):
        for joint in list(body.findall("joint")):
            name = joint.get("name", "")
            if not name.startswith("joint_") or name in KEEP or name in {"joint_x", "joint_y", "joint_th"}:
                continue
            angle = original_data.qpos[original.joint(name).qposadr[0]]
            quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
            quat /= np.linalg.norm(quat)
            rotation = np.zeros(4)
            mujoco.mju_axisAngle2Quat(rotation, np.fromstring(joint.get("axis", "0 0 1"), sep=" "), angle)
            result = np.zeros(4)
            mujoco.mju_mulQuat(result, quat, rotation)
            body.set("quat", " ".join(map(str, result)))
            body.remove(joint)
            add(body, "geom", name=f"locked_marker_{name}", type="sphere", size="0.045", mass="0",
                rgba="1 0.65 0.08 0.95", contype="0", conaffinity="0")
    for actuator in list(robot.find("actuator")):
        name = actuator.get("joint", "")
        if name.startswith("joint_") and name not in KEEP | {"joint_x", "joint_y", "joint_th"}:
            robot.find("actuator").remove(actuator)


from simple_arm import replace_arm


def render_variant(name, robot, original, original_data):
    robot_path = OUT / f"{name}_robot.xml"
    save(robot, robot_path)
    scene = ET.parse(ROOT / "scenes/fishingbot.xml").getroot()
    scene.find("include").set("file", robot_path.name)
    scene.remove(scene.find("keyframe"))
    if name == "simple":
        sensors = scene.find("sensor")
        for sensor in list(sensors):
            if sensor.tag == "touch":
                sensors.remove(sensor)
    path = OUT / f"{name}.xml"
    save(scene, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    # Match the current home pose by name; removed joints have their rotation baked into the bodies.
    for index in range(model.njnt):
        joint = model.joint(index)
        if joint.name.startswith("simple_"):
            if joint.name.endswith("_slide"):
                data.qpos[joint.qposadr[0]] = 0.02
            continue
        old = original.joint(joint.name)
        width = 7 if joint.type == mujoco.mjtJoint.mjJNT_FREE else 4 if joint.type == mujoco.mjtJoint.mjJNT_BALL else 1
        data.qpos[joint.qposadr[0]:joint.qposadr[0]+width] = original_data.qpos[old.qposadr[0]:old.qposadr[0]+width]
    mujoco.mj_forward(model, data)
    if name == "locked":
        assert np.allclose(data.site("pinch_site").xpos, original_data.site("pinch_site").xpos, atol=1e-7)
        assert sum(model.joint(i).name in KEEP for i in range(model.njnt)) == 4
    else:
        assert sum(model.joint(i).name in {"simple_yaw_joint", "simple_shoulder_joint", "simple_elbow_joint", "simple_wrist_joint"} for i in range(model.njnt)) == 4
    # Save the preview pose for inspection in Studio as well as for this static render.
    keys = add(scene, "keyframe")
    add(keys, "key", name="preview", qpos=" ".join(map(str, data.qpos)))
    save(scene, path)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.42, 0, 0.56]
    camera.distance = 1.95
    camera.azimuth = 135
    camera.elevation = -20
    with mujoco.Renderer(model, height=720, width=800) as renderer:
        renderer.update_scene(data, camera=camera)
        result = Image.fromarray(renderer.render())
        result.save(OUT / f"{name}.png")
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference_robot = OUT / "original_robot.xml"
    save(prepare_robot(), reference_robot)
    reference = ET.parse(ROOT / "scenes/fishingbot.xml").getroot()
    reference.find("include").set("file", reference_robot.name)
    home = reference.find("keyframe/key")
    original_home = ET.parse(ROOT / "models/stanford_tidybot/tidybot.xml").find("keyframe/key")
    home.set("qpos", original_home.get("qpos") + " " + " ".join(home.get("qpos").split()[9:]))
    home.set("ctrl", original_home.get("ctrl"))
    reference_path = OUT / "original_reference.xml"
    save(reference, reference_path)
    original = mujoco.MjModel.from_xml_path(str(reference_path))
    data = mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original, data, original.key("home").id)
    mujoco.mj_forward(original, data)
    simple_robot = prepare_robot()
    replace_arm(simple_robot)
    simple = render_variant("simple", simple_robot, original, data)
    locked_robot = prepare_robot()
    lock_joints(locked_robot, original, data)
    locked = render_variant("locked", locked_robot, original, data)
    canvas = Image.new("RGB", (1600, 910), "#f3f5f7")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 28)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)
    canvas.paste(simple, (0, 100))
    canvas.paste(locked, (800, 100))
    draw.text((32, 24), "A. 단순한 4관절 팔 + 단순 집게", font=font, fill="#152e36")
    draw.text((832, 24), "B. 기존 팔 유지 + 3개 관절 고정", font=font, fill="#152e36")
    draw.text((32, 66), "청록색: 움직이는 팔 관절 4개", font=small, fill="#176c72")
    draw.text((832, 66), "노란색: 고정한 관절 3개 / 나머지 4개 사용", font=small, fill="#8d5f00")
    draw.text((32, 838), "MuJoCo 실제 모델 렌더링 · 설계 비교용 정지 자세", font=small, fill="#334550")
    draw.text((32, 872), "팔 길이·고정 각도는 예시이며 도달성·파지·물리 안정성은 아직 검증하지 않았습니다.", font=small, fill="#334550")
    canvas.save(OUT / "comparison.png")
    print("Rendered both 4-joint options; locked variant preserves original home gripper pose.")


if __name__ == "__main__":
    main()
