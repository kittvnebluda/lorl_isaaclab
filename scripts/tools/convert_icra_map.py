"""Convert the ICRA 2024 Quadruped Challenge arena (ROS URDF) into MuJoCo deploy scenes.

Source: ``ICRA2024_QRC_Simulation_Map/urdf/map_{flat,sloped}.urdf`` (+ OBJ meshes).
Output: ``unitree_aliengo/scene_icra_{flat,sloped}.xml`` and ``assets/icra/*.obj``.

Pipeline per variant:
  1. Rewrite the ``package://`` mesh prefix to the real on-disk path so ``urdf2mjcf``
     can resolve the OBJs (it joins ``urdf_dir / filename``).
  2. Run ``urdf2mjcf`` with ``floating_base=False`` so the whole arena (all joints are
     ``type="fixed"``) bakes into one *static* body tree -- no freejoint, no actuators.
  3. Lift the baked body tree out of the generated MJCF into a hand-built scene that
     ``<include>``s ``aliengo.xml``. To stay decoupled from aliengo's own ``visual`` /
     ``collision`` default classes and ``collision_material`` (same names!), every geom
     gets explicit ``rgba`` / contype / group attributes instead of class references.

Run inside the ``isaaclab`` pyenv (which has ``urdf2mjcf``):
    python scripts/tools/convert_icra_map.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import trimesh

ICRA_REPO_URL = "https://github.com/teamgrit-lab/ICRA2024_Quadruped_Robot_Challenges.git"
PKG_PREFIX = "package://ICRA2024_Quadruped_Competition/"
VARIANTS = {"flat": "map_flat.urdf", "sloped": "map_sloped.urdf"}

START_OFFSET = (-6.0, 4.0, 0.08)
DEFAULT_RGBA = "0.92 0.93 0.90 1"  # the unnamed URDF material (board off-white)

# --- paths -------------------------------------------------------------------
ICRA_REPO_ROOT = Path("/tmp/ICRA2024_Quadruped_Robot_Challenges")
MAP_ROOT = ICRA_REPO_ROOT / "ICRA2024_QRC_Simulation_Map"

ALIENGO_DIR_REL = Path("source/legged_obstacle_rl/legged_obstacle_rl/tasks/mujoco/unitree_aliengo")
ALIENGO_DIR = Path(__file__).resolve().parents[2] / ALIENGO_DIR_REL
ASSET_ICRA = ALIENGO_DIR / "assets" / "icra"

# CoACD convex-decomposition threshold (concavity). Lower -> more parts, closer to the
# true concave shape; higher -> fewer parts, coarser. 0.05 keeps lane lips / rail gaps.
COACD_THRESHOLD = 0.05
DECOMP_SUBDIR = "decomp"  # under assets/icra/

# --- scene template (mirrors unitree_aliengo/scene.xml) ----------------------
SCENE_TEMPLATE = """<mujoco model="aliengo icra {variant} scene">
  <include file="aliengo.xml" />

  <statistic center="0 0 0.1" extent="0.8" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="120" elevation="-20" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
{meshes}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
{map_body}
  </worldbody>
</mujoco>
"""


def rewrite_urdf(src: Path, dst: Path, map_root: Path) -> None:
    dst.write_text(src.read_text().replace(PKG_PREFIX, str(map_root) + "/"))


def run_urdf2mjcf(urdf: Path, out: Path, meta: Path) -> None:
    meta.write_text(json.dumps({"floating_base": False, "freejoint": False}))
    subprocess.run(["urdf2mjcf", str(urdf), "--output", str(out), "--metadata-file", str(meta)], check=True)


def material_rgba(raw_root: ET.Element) -> dict[str, str]:
    """Map material name -> rgba string from the generated MJCF <asset>."""
    out = {}
    for mat in raw_root.findall("./asset/material"):
        out[mat.get("name", "")] = mat.get("rgba", DEFAULT_RGBA)
    return out


def make_visual_geom(geom: ET.Element, rgba: dict[str, str]) -> None:
    """Bake explicit attrs onto a visual geom: non-colliding, group 2, colored via rgba."""
    mat = geom.get("material", "")
    keep: dict[str, str] = {k: geom.get(k) for k in ("name", "pos", "quat", "type", "mesh") if geom.get(k) is not None}  # pyright: ignore[reportAssignmentType]
    geom.attrib.clear()
    geom.attrib.update(keep)
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("group", "2")
    geom.set("rgba", rgba.get(mat, DEFAULT_RGBA))


def make_collision_geoms(geom: ET.Element, decomp: dict[str, list[str]]) -> list[ET.Element]:
    """Expand one mesh collision geom into one geom per convex-decomposition part.

    Aliengo collision geoms are contype=0 conaffinity=1 (collide only with the floor's
    contype=1), so the map exposes contype=1 (robot collides with it); conaffinity=0 skips
    pointless static map-vs-map contacts. Each part is a convex hull -> their union
    approximates the original concave shape.
    """
    pos = geom.get("pos", "0 0 0")
    quat = geom.get("quat", "1 0 0 0")
    base_name = geom.get("name", "icra")
    parts = decomp[geom.get("mesh")]  # pyright: ignore[reportArgumentType]
    out = []
    for i, part in enumerate(parts):
        out.append(
            ET.Element(
                "geom",
                {
                    "name": f"{base_name}_{i:02d}",
                    "pos": pos,
                    "quat": quat,
                    "type": "mesh",
                    "mesh": part,
                    "contype": "1",
                    "conaffinity": "0",
                    "condim": "3",
                    "friction": "1 0.01 0.01",
                    "group": "3",
                },
            )
        )
    return out


def build_map_body(raw_root: ET.Element, rgba: dict[str, str], decomp: dict[str, list[str]], offset) -> str:
    """Lift the baked arena body tree into a static <body name="icra_map">."""
    # The single child of <worldbody> is the baked root body (all fixed joints merged in).
    root_body = raw_root.find("./worldbody/body")
    assert root_body is not None, "no baked body in generated MJCF"
    root_body.attrib.pop("childclass", None)
    root_body.set("name", "icra_pieces")
    # Drop inertials (ignored for static bodies); rewrite visual geoms and expand
    # collision geoms into convex-decomposition parts. Rebuild each body's geom list.
    for body in root_body.iter("body"):
        body.attrib.pop("childclass", None)
        new_children = []
        for child in list(body):
            if child.tag == "inertial":
                body.remove(child)
            elif child.tag == "geom" and child.get("class") == "collision":
                body.remove(child)
                new_children.extend(make_collision_geoms(child, decomp))
            elif child.tag == "geom":
                child.attrib.pop("class", None)
                make_visual_geom(child, rgba)
        for g in new_children:
            body.append(g)

    wrapper = ET.Element("body", {"name": "icra_map", "pos": "{} {} {}".format(*offset)})
    wrapper.append(root_body)
    ET.indent(wrapper, space="  ")
    body_xml = ET.tostring(wrapper, encoding="unicode")
    return "\n".join("    " + line for line in body_xml.splitlines())


def get_map_root() -> Path:
    if MAP_ROOT.exists():
        return MAP_ROOT

    completed_proc = subprocess.run(["git", "clone", ICRA_REPO_URL, ICRA_REPO_ROOT])
    if completed_proc.returncode != 0:
        raise RuntimeError(f"Could not clone ICRA repository: {completed_proc.stderr}")

    return MAP_ROOT


def decompose_meshes(map_root: Path) -> dict[str, list[str]]:
    """Copy source OBJs and CoACD-decompose each into convex parts.

    Returns a map ``"<name>.obj" -> ["<name>_col00", ...]`` of MJCF collision-part mesh
    names. The visual mesh keeps the original ``"<name>.obj"`` mesh name.
    """
    ASSET_ICRA.mkdir(parents=True, exist_ok=True)
    (ASSET_ICRA / DECOMP_SUBDIR).mkdir(parents=True, exist_ok=True)
    coacd.set_log_level("error")
    decomp: dict[str, list[str]] = {}
    for obj in sorted((map_root / "meshes" / "visual").glob("*.obj")):
        shutil.copy2(obj, ASSET_ICRA / obj.name)  # original, used by the visual geom
        mesh = trimesh.load(str(obj), force="mesh")
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=COACD_THRESHOLD)  # pyright: ignore[reportAttributeAccessIssue]
        names = []
        for i, (verts, faces) in enumerate(parts):
            part_name = f"{obj.stem}_col{i:02d}"
            trimesh.Trimesh(vertices=verts, faces=faces).export(ASSET_ICRA / DECOMP_SUBDIR / f"{part_name}.obj")
            names.append(part_name)
        decomp[obj.name] = names
        print(f"  {obj.name}: {len(names)} convex parts")
    return decomp


def mesh_asset_block(raw_root: ET.Element, decomp: dict[str, list[str]]) -> str:
    """Emit <mesh> assets: original meshes (for visuals) + all collision parts."""
    lines = []
    for mesh in raw_root.findall("./asset/mesh"):
        name = mesh.get("name")
        if name is None:
            raise RuntimeError("Wrong element attribute")
        lines.append(f'    <mesh name="{name}" file="icra/{Path(name).name}" />')
        for part in decomp.get(name, []):
            lines.append(f'    <mesh name="{part}" file="icra/{DECOMP_SUBDIR}/{part}.obj" />')
    return "\n".join(lines)


def convert(variant: str, urdf_name: str, decomp: dict[str, list[str]], map_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        urdf = tmp / urdf_name
        rewrite_urdf(map_root / "urdf" / urdf_name, urdf, map_root)
        raw = tmp / f"icra_{variant}_raw.xml"
        run_urdf2mjcf(urdf, raw, tmp / "meta.json")

        raw_root = ET.parse(raw).getroot()
        rgba = material_rgba(raw_root)
        scene = SCENE_TEMPLATE.format(
            variant=variant,
            meshes=mesh_asset_block(raw_root, decomp),
            map_body=build_map_body(raw_root, rgba, decomp, START_OFFSET),
        )
        out = ALIENGO_DIR / f"scene_icra_{variant}.xml"
        out.write_text(scene)
        print(f"wrote {out}")


def main() -> None:
    map_root = get_map_root()
    print(f"Decomposing meshes -> {ASSET_ICRA}")
    decomp = decompose_meshes(map_root)
    for variant, urdf_name in VARIANTS.items():
        convert(variant, urdf_name, decomp, map_root)


if __name__ == "__main__":
    main()
