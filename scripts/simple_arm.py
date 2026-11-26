"""Simple four-axis FishingBot arm shared by scene and design previews."""
import xml.etree.ElementTree as ET


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def replace_arm(robot, mount_height=0.335):
    base = robot.find("worldbody/body[@name='base_link']")
    base.remove(base.find("body"))
    for tag in ("contact", "tendon", "equality"):
        node = robot.find(tag)
        if node is not None:
            robot.remove(node)
    actuators = robot.find("actuator")
    for actuator in list(actuators):
        if actuator.get("joint") not in {"joint_x", "joint_y", "joint_th"}:
            actuators.remove(actuator)

    def hinge(body, name, axis):
        add(body, "joint", name=name, type="hinge", axis=axis, range="-2.6 2.6", damping="2")
        add(actuators, "position", name=name, joint=name, kp="250", kv="20")

    def geom(body, name, shape, **attrs):
        add(body, "geom", name=name, type=shape, rgba="0.70 0.76 0.79 1", **attrs)

    def joint_marker(body, name, radius):
        add(body, "geom", name=name, type="sphere", size=radius, mass="0.15", rgba="0.05 0.65 0.68 1")

    yaw = add(base, "body", name="simple_yaw", pos=f"0 0 {mount_height}")
    hinge(yaw, "simple_yaw_joint", "0 0 1")
    geom(yaw, "simple_pedestal", "cylinder", fromto="0 0 0 0 0 0.10", size="0.055", mass="0.8")
    joint_marker(yaw, "yaw_marker", "0.06")
    shoulder = add(yaw, "body", name="simple_shoulder", pos="0 0 0.10")
    hinge(shoulder, "simple_shoulder_joint", "0 1 0")
    joint_marker(shoulder, "shoulder_marker", "0.045")
    geom(shoulder, "simple_upper", "capsule", fromto="0 0 0 0.18 0 0.24", size="0.03", mass="0.6")
    elbow = add(shoulder, "body", name="simple_elbow", pos="0.18 0 0.24")
    hinge(elbow, "simple_elbow_joint", "0 1 0")
    joint_marker(elbow, "elbow_marker", "0.04")
    geom(elbow, "simple_forearm", "capsule", fromto="0 0 0 0.28 0 -0.12", size="0.026", mass="0.45")
    wrist = add(elbow, "body", name="simple_wrist", pos="0.28 0 -0.12")
    hinge(wrist, "simple_wrist_joint", "1 0 0")
    joint_marker(wrist, "wrist_marker", "0.035")
    geom(wrist, "simple_palm", "box", pos="0.045 0 0", size="0.035 0.045 0.025", mass="0.2")
    for side, sign in (("left", 1), ("right", -1)):
        finger = add(wrist, "body", name=f"simple_{side}_finger", pos=f"0.09 {sign*0.018} 0")
        add(finger, "joint", name=f"simple_{side}_slide", type="slide", axis=f"0 {sign} 0", range="0 0.025", damping="1")
        add(finger, "geom", type="box", pos="0.015 0 0", size="0.04 0.008 0.022", mass="0.05", rgba="0.12 0.16 0.19 1")
        add(finger, "site", name=f"{side}_grip_touch", type="box", pos="0.015 0 0", size="0.041 0.009 0.023", rgba="0 0 0 0")
    add(wrist, "site", name="pinch_site", pos="0.105 0 0", size="0.003", rgba="1 1 0 0.5")
    equality = add(robot, "equality")
    add(equality, "joint", joint1="simple_left_slide", joint2="simple_right_slide", polycoef="0 1 0 0 0")
    add(actuators, "position", name="simple_grip", joint="simple_left_slide", kp="100", kv="5", ctrlrange="0 0.025")
