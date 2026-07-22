import json
import shutil
from pathlib import Path


ROOT = Path("/workspace_whz")
SRC_DIR = ROOT / "data/output/geometric_joint_estimate_sam3door_sam3d_fitted"
OUT_DIR = SRC_DIR / "prismatic_existing_fitted_visible"

BASE_OPEN = SRC_DIR / "base_sam3d_fitted_camera_decimated.obj"
DRAWER_OPEN = SRC_DIR / "door_sam3d_fitted_open_camera_decimated.obj"
PROJ_JSON = SRC_DIR / "prismatic_projection_aligned/projection_fit_targets.json"


def parse_vertex(line: str) -> tuple[float, float, float]:
    parts = line.split()
    return float(parts[1]), float(parts[2]), float(parts[3])


def format_vertex(x: float, y: float, z: float) -> str:
    return f"v {x:.9f} {y:.9f} {z:.9f}\n"


def transform_obj_vertices(src: Path, dst: Path, origin: list[float], axis: list[float], distance: float) -> None:
    ox, oy, oz = origin
    ax, ay, az = axis
    with src.open("r", encoding="utf-8", errors="ignore") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("v "):
                x, y, z = parse_vertex(line)
                # qpos=0 child mesh: take the observed open drawer, slide it back to closed,
                # then express it in the joint child frame. Translation only; shape/topology is untouched.
                x = x - ax * distance - ox
                y = y - ay * distance - oy
                z = z - az * distance - oz
                fout.write(format_vertex(x, y, z))
            else:
                fout.write(line)


def write_urdf(path: Path, origin: list[float], axis: list[float], distance: float) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<robot name="cabinet_drawer_prismatic_existing_fitted">
  <link name="base">
    <visual><geometry><mesh filename="base_open_visible.obj"/></geometry></visual>
    <collision><geometry><mesh filename="base_open_visible.obj"/></geometry></collision>
  </link>
  <link name="drawer">
    <visual><geometry><mesh filename="drawer_closed_child_prismatic.obj"/></geometry></visual>
    <collision><geometry><mesh filename="drawer_closed_child_prismatic.obj"/></geometry></collision>
  </link>
  <joint name="drawer_slide" type="prismatic">
    <parent link="base"/>
    <child link="drawer"/>
    <origin xyz="{origin[0]:.9f} {origin[1]:.9f} {origin[2]:.9f}" rpy="0 0 0"/>
    <axis xyz="{axis[0]:.9f} {axis[1]:.9f} {axis[2]:.9f}"/>
    <limit lower="0" upper="{distance:.9f}" effort="20" velocity="0.5"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joint = json.loads(PROJ_JSON.read_text(encoding="utf-8"))["prismatic_guess"]
    axis = joint["axis_camera"]
    origin = joint["origin_camera_m"]
    distance = float(joint["open_distance_m"])

    shutil.copy2(BASE_OPEN, OUT_DIR / "base_open_visible.obj")
    shutil.copy2(DRAWER_OPEN, OUT_DIR / "drawer_open_visible.obj")
    transform_obj_vertices(
        DRAWER_OPEN,
        OUT_DIR / "drawer_closed_child_prismatic.obj",
        origin=origin,
        axis=axis,
        distance=distance,
    )
    write_urdf(OUT_DIR / "cabinet_drawer_prismatic_existing_fitted.urdf", origin, axis, distance)

    manifest = {
        "method": "Uses the existing fitted meshes without any new scaling, rotation fitting, or decimation. drawer_open_visible.obj is the observed qpos1 drawer. drawer_closed_child_prismatic.obj is derived by translation only for a prismatic URDF child frame.",
        "inputs": {
            "base_open": str(BASE_OPEN),
            "drawer_open": str(DRAWER_OPEN),
        },
        "joint": {
            "type": "prismatic",
            "origin_camera_m": origin,
            "axis_camera": axis,
            "open_distance_m": distance,
        },
        "outputs": {
            "base_open_obj": str(OUT_DIR / "base_open_visible.obj"),
            "drawer_open_obj": str(OUT_DIR / "drawer_open_visible.obj"),
            "drawer_closed_child_obj": str(OUT_DIR / "drawer_closed_child_prismatic.obj"),
            "urdf": str(OUT_DIR / "cabinet_drawer_prismatic_existing_fitted.urdf"),
        },
    }
    (OUT_DIR / "existing_fitted_prismatic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
