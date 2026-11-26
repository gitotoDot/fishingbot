"""Check/render the D07 scene, or open it at the prepared home keyframe."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def check(model, data):
    rod = model.body("fishing_rod").id
    descendants = {rod}
    for body in range(rod + 1, model.nbody):
        if int(model.body_parentid[body]) in descendants:
            descendants.add(body)
    mass = float(sum(model.body_mass[list(descendants)]))
    assert np.isclose(mass, 0.1, atol=1e-9), f"Rod set mass: {mass}"
    assert np.allclose(model.geom("floor").size[:2] * 2, [3, 3])
    assert np.isclose(model.geom("bin_bottom").size[0] * 2, 0.15)
    for name in ("rod_handle", "rod_shaft", "toy_ball_geom", "bin_bottom"):
        assert model.geom(name).id >= 0
    assert model.body("fishing_rod").parentid == 0, "Rod must be independent of robot"
    assert np.isclose(model.geom("rod_handle").size[0], 0.015)
    assert np.isclose(model.geom("rod_handle").size[1] * 2, 0.1)
    assert np.isclose(model.geom("rod_shaft").size[1] * 2, 0.2)
    assert np.isclose(model.geom("toy_ball_geom").size[0] * 2, 0.05)
    arm_joints = [model.joint(i).name for i in range(model.njnt)
                  if model.joint(i).name.startswith("simple_") and model.joint(i).name.endswith("_joint")]
    assert len(arm_joints) == 4
    assert np.allclose(model.body("simple_yaw").pos, [0, 0, 0.16])
    for index in range(10):
        assert np.isclose(model.geom(f"string_geom_{index:02d}").size[1] * 2, 0.02)

    start = data.site("handle_grasp").xpos.copy()
    max_penetration = 0.0
    for _ in range(round(3 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert data.time > 0, "Simulation reset unexpectedly"
        for contact in data.contact:
            if (model.geom_bodyid[contact.geom1] in descendants
                    or model.geom_bodyid[contact.geom2] in descendants):
                max_penetration = max(max_penetration, -float(contact.dist))
    assert np.isclose(data.time, 3, atol=0.002), "Simulation reset or time drift"
    assert not any(w.number for w in data.warning), "MuJoCo physics warnings"
    end = data.site("handle_grasp").xpos.copy()
    assert np.linalg.norm(end - start) < 0.015, f"Rod shifted in bin: {start} -> {end}"
    assert max_penetration < 0.003, f"Rod penetration: {max_penetration}"
    assert end[2] - 0.0151 > 0.26, "Marker must remain fully above bin rim"
    assert data.site("toy_center").xpos[2] + 0.025 < 0.26, "Toy ball must remain inside bin"

    output = ROOT / "output/scene"
    output.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera="overview")
        Image.fromarray(renderer.render()).save(output / "overview.png")
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="base")
        rgb = renderer.render().copy()
        Image.fromarray(rgb).save(output / "base_rgb.png")
        green = (rgb[:, :, 1] > 100) & (rgb[:, :, 1] > rgb[:, :, 0].astype(float) * 1.6) & (rgb[:, :, 1] > rgb[:, :, 2].astype(float) * 1.6)
        assert green.sum() >= 10, "Handle marker not visible to base camera"
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="base")
        depth = renderer.render().copy()
        assert np.isfinite(depth).all() and (depth > 0).all()
        np.save(output / "base_depth_m.npy", depth)
        Image.fromarray((255 * (1 - np.clip(depth / 3, 0, 1))).astype(np.uint8)).save(output / "base_depth.png")

    report = {"mujoco_version": mujoco.__version__, "simulation_s": float(data.time),
              "rod_set_mass_kg": mass, "handle_initial_m": start.tolist(), "handle_final_m": end.tolist(),
              "max_rod_penetration_m": max_penetration, "marker_green_pixels": int(green.sum()),
              "physics_warnings": 0, "arm_active_joints": len(arm_joints),
              "base_height_scale": 0.5, "bin_rim_height_m": 0.26, "bin_outer_diameter_m": 0.15,
              "grasp_and_reach_validated": False,
              "limits": "Scene stability only; linked string approximation; no perception or grasp controller"}
    (output / "check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Open interactive viewer instead of checking/rendering")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes/fishingbot.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    if args.view:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.45, 0, 0.45]
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            while viewer.is_running():
                started = time.perf_counter()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.perf_counter() - started)))
    else:
        check(model, data)


if __name__ == "__main__":
    main()
