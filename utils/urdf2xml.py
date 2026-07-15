#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a URDF file to MuJoCo MJCF XML.

Version: mesh-path-fix-keep-visual-auto-motor-required-base-wrap-freejoint-default-classes-expanded-attrs-light-floor-v3-compatible-2026-06-17

Usage:
    python urdf2xml.py --path=/path/to/robot.urdf --base=base_link

Default output:
    /path/to/robot.xml

Dependencies:
    pip install mujoco

Path handling policy:
    - Mesh paths are resolved before MuJoCo compiles the URDF.
    - The script first searches `meshes/` in the same folder as the URDF.
    - Then it searches `../meshes/`, i.e. a meshes folder in the parent folder.
    - package:// paths are also supported through auto package discovery and --package-map.
    - The temporary URDF uses absolute mesh paths, so relative paths will not break after
      the URDF is copied into /tmp.
    - By default, MuJoCo URDF visual meshes are preserved by injecting
      <mujoco><compiler discardvisual="false"/></mujoco> into the temporary URDF.
    - The --base argument is required. The generated MJCF body named by --base gets
      a <freejoint/>. If that body is absent after MuJoCo URDF compilation, the script wraps direct <worldbody> children into a new body with this name.
    - After MJCF export, a robot default-class tree is added:
          robot / motor / visual / collision.
    - Visual mesh geoms are assigned class="visual". Collision geoms are assigned
      class="collision". The class defaults carry the MuJoCo contact, friction,
      density, group, and material settings.
    - The generated MJCF gets a checker floor material, a non-shadow-casting directional light, and a plane floor in group=0.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET


MESH_EXTENSIONS = {".stl", ".dae", ".obj", ".ply", ".mesh"}


def parse_package_map(items: Optional[Iterable[str]]) -> Dict[str, Path]:
    """Parse --package-map entries like pkg_name=/abs/or/relative/path."""
    result: Dict[str, Path] = {}
    if not items:
        return result

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --package-map '{item}'. Expected format: package_name=/path/to/package")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --package-map '{item}'. Expected format: package_name=/path/to/package")
        result[name] = Path(path).expanduser().resolve()
    return result


def build_auto_package_map(urdf_path: Path, package_roots: Optional[Iterable[str]]) -> Dict[str, Path]:
    """
    Build a best-effort package map.

    Supports common layouts:
      1. URDF folder itself is the package directory:
            /path/ZB26RD02B-URDF1.1/robot.urdf
            package://ZB26RD02B-URDF1.1/meshes/base.STL
      2. Parent directory contains package directories:
            /path/ZB26RD02B-URDF1.1/urdf/robot.urdf
            /path/ZB26RD02B-URDF1.1/meshes/base.STL
      3. User-specified --package-root directories.
    """
    candidates: List[Path] = []

    # URDF directory and its parents are common package roots.
    candidates.append(urdf_path.parent)
    candidates.append(urdf_path.parent.parent)
    candidates.extend(urdf_path.parents)

    if package_roots:
        candidates.extend(Path(p).expanduser().resolve() for p in package_roots)

    package_map: Dict[str, Path] = {}
    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue

        # Case: root itself is a package folder.
        package_map.setdefault(root.name, root.resolve())

        # Case: root contains multiple package folders.
        try:
            for child in root.iterdir():
                if child.is_dir():
                    package_map.setdefault(child.name, child.resolve())
        except PermissionError:
            continue

    return package_map


def _split_package_url(value: str) -> Optional[Tuple[str, str]]:
    """Return (package_name, relative_path) for package:// URLs."""
    if not value.startswith("package://"):
        return None
    rest = value[len("package://") :]
    if "/" not in rest:
        return rest, ""
    pkg_name, rel_path = rest.split("/", 1)
    return pkg_name, rel_path


def _mesh_tail(path_text: str) -> Path:
    """
    Return the path part inside/after a meshes directory.

    Examples:
        ../meshes/IMU_Link.STL        -> IMU_Link.STL
        meshes/sub/a.stl              -> sub/a.stl
        package://pkg/meshes/sub/a.stl -> sub/a.stl
        visual/a.stl                  -> a.stl
    """
    normalized = path_text.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in ("", ".", "..")]
    lower_parts = [p.lower() for p in parts]
    if "meshes" in lower_parts:
        idx = lower_parts.index("meshes")
        tail_parts = parts[idx + 1 :]
        if tail_parts:
            return Path(*tail_parts)
    return Path(parts[-1]) if parts else Path(path_text).name # type: ignore


def _mesh_search_candidates(raw_value: str, urdf_path: Path, package_map: Dict[str, Path]) -> List[Path]:
    """Generate mesh path candidates in priority order."""
    candidates: List[Path] = []
    urdf_dir = urdf_path.parent
    parent_dir = urdf_dir.parent

    package_info = _split_package_url(raw_value)
    if package_info is not None:
        pkg_name, rel_path = package_info
        tail = _mesh_tail(rel_path)

        # User-specific policy: meshes next to URDF first, then one level above.
        candidates.append(urdf_dir / "meshes" / tail)
        candidates.append(parent_dir / "meshes" / tail)

        pkg_dir = package_map.get(pkg_name)
        if pkg_dir is not None and rel_path:
            candidates.append(pkg_dir / rel_path)
            candidates.append(pkg_dir / "meshes" / tail)
        return candidates

    expanded = os.path.expandvars(os.path.expanduser(raw_value))
    raw_path = Path(expanded)
    tail = _mesh_tail(expanded)

    # User-specific policy: meshes next to URDF first, then one level above.
    candidates.append(urdf_dir / "meshes" / tail)
    candidates.append(parent_dir / "meshes" / tail)

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        # Resolve normal relative paths against the original URDF directory, not /tmp.
        candidates.append(urdf_dir / raw_path)
        candidates.append(Path.cwd() / raw_path)

    return candidates


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    """Keep path order while removing duplicates."""
    result: List[Path] = []
    seen = set()
    for path in paths:
        try:
            key = str(path.expanduser().resolve(strict=False))
        except OSError:
            key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def resolve_mesh_filename(raw_value: str, urdf_path: Path, package_map: Dict[str, Path]) -> Tuple[str, bool, List[Path]]:
    """
    Resolve one mesh filename.

    Returns:
        resolved_value: absolute path when found; otherwise best-effort absolute path
        found: whether the file exists
        candidates: paths tried
    """
    candidates = _dedupe_paths(_mesh_search_candidates(raw_value, urdf_path, package_map))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve()), True, candidates

    # Not found. Return the most informative absolute path so MuJoCo errors are readable.
    # Priority remains same-folder meshes, then parent meshes.
    fallback = candidates[0] if candidates else (urdf_path.parent / raw_value)
    return str(fallback.expanduser().resolve(strict=False)), False, candidates


def preserve_urdf_visuals(text: str) -> str:
    """
    Make MuJoCo keep URDF <visual> meshes.

    MuJoCo's default for URDF is compiler discardvisual="true", which removes
    purely visual meshes during compilation. We add or modify the MuJoCo URDF
    extension so the compiled MJCF contains visual geoms/assets.
    """
    # Case 1: a <compiler .../> already exists inside a <mujoco> extension.
    compiler_pattern = re.compile(r"(<mujoco\b[^>]*>.*?<compiler\b)([^>]*?)(/?>)", re.IGNORECASE | re.DOTALL)
    match = compiler_pattern.search(text)
    if match:
        attrs = match.group(2)
        if re.search(r"\bdiscardvisual\s*=", attrs, flags=re.IGNORECASE):
            attrs = re.sub(
                r"\bdiscardvisual\s*=\s*(['\"])(.*?)\1",
                'discardvisual="false"',
                attrs,
                flags=re.IGNORECASE,
            )
        else:
            attrs = attrs.rstrip() + ' discardvisual="false"'
        return text[: match.start(2)] + attrs + text[match.end(2) :]

    # Case 2: a <mujoco> extension exists, but no compiler child.
    mujoco_open_pattern = re.compile(r"(<mujoco\b[^>]*>)", re.IGNORECASE)
    match = mujoco_open_pattern.search(text)
    if match:
        insert_pos = match.end(1)
        return text[:insert_pos] + '\n  <compiler discardvisual="false"/>' + text[insert_pos:]

    # Case 3: no <mujoco> extension. Insert it as a child of top-level <robot>.
    robot_open_pattern = re.compile(r"(<robot\b[^>]*>)", re.IGNORECASE)
    match = robot_open_pattern.search(text)
    if not match:
        raise ValueError("Could not find top-level <robot> tag in URDF.")
    insert_pos = match.end(1)
    extension = '\n  <mujoco>\n    <compiler discardvisual="false"/>\n  </mujoco>'
    return text[:insert_pos] + extension + text[insert_pos:]


def rewrite_mesh_filenames(
    source_urdf: Path,
    package_map: Dict[str, Path],
    temp_dir: Path,
    fail_on_missing_mesh: bool = True,
    keep_visuals: bool = True,
) -> Path:
    """
    Rewrite mesh filename attributes into absolute file paths.

    This fixes both package:// and relative paths before the URDF is copied into /tmp.
    The original URDF is not modified.
    """
    text = source_urdf.read_text(encoding="utf-8", errors="ignore")

    # URDF mesh file references normally appear as:
    #     <mesh filename="..."/>
    # Some URDFs may also use single quotes, so both are supported.
    mesh_tag_pattern = re.compile(r"(<mesh\b[^>]*?\bfilename\s*=\s*)(['\"])(.*?)(\2)([^>]*>)", re.IGNORECASE | re.DOTALL)

    missing_meshes: List[Tuple[str, List[Path]]] = []
    rewritten_count = 0

    def replace_mesh_filename(match: re.Match[str]) -> str:
        nonlocal rewritten_count
        prefix = match.group(1)
        quote = match.group(2)
        raw_value = match.group(3)
        suffix_quote = match.group(4)
        suffix = match.group(5)

        resolved_value, found, candidates = resolve_mesh_filename(raw_value, source_urdf, package_map)
        if not found:
            missing_meshes.append((raw_value, candidates))

        if resolved_value != raw_value:
            rewritten_count += 1
        return f"{prefix}{quote}{resolved_value}{suffix_quote}{suffix}"

    new_text = mesh_tag_pattern.sub(replace_mesh_filename, text)

    if keep_visuals:
        new_text = preserve_urdf_visuals(new_text)
        print('[INFO] Keeping URDF visual meshes: injected/updated <compiler discardvisual="false"/>.')

    if rewritten_count:
        print(f"[INFO] Rewrote {rewritten_count} mesh filename path(s) to absolute paths.")

    if missing_meshes:
        print("[ERROR] Some mesh files could not be found.", file=sys.stderr)
        for raw_value, candidates in missing_meshes:
            print(f"  - {raw_value}", file=sys.stderr)
            print("    tried:", file=sys.stderr)
            for candidate in candidates[:8]:
                print(f"      {candidate.expanduser().resolve(strict=False)}", file=sys.stderr)
            if len(candidates) > 8:
                print(f"      ... {len(candidates) - 8} more", file=sys.stderr)

        if fail_on_missing_mesh:
            raise FileNotFoundError(
                "Mesh path resolution failed. Put meshes in ./meshes or ../meshes relative to the URDF, "
                "or add --package-map package_name=/path/to/package."
            )

    temp_urdf = temp_dir / source_urdf.name
    temp_urdf.write_text(new_text, encoding="utf-8")
    return temp_urdf




def _version_tuple(version_text: str) -> Tuple[int, int, int]:
    """Parse the numeric prefix of a semantic version like 3.3.1 or 3.4.0.dev0."""
    nums = [int(x) for x in re.findall(r"\d+", version_text)[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])  # type: ignore[return-value]


def prepare_urdf_root_for_floating_base(
    urdf_path: Path,
    base_body_name: str,
    *,
    enable_saveinertial: bool,
) -> str:
    """
    Preserve the URDF root link as a real MJCF body before MuJoCo compilation.

    MuJoCo maps the URDF root link to MJCF worldbody. worldbody cannot have an
    inertial element or a joint. Adding a freejoint only after mj_saveLastXML()
    therefore creates a new body too late: the original root-link inertia has
    already been lost from the saved MJCF structure.

    We solve this before compilation by adding a synthetic massless URDF root
    link and a fixed joint from that synthetic root to the requested base link.
    compiler fusestatic="false" is required so the fixed child is not fused
    back into worldbody. After MJCF export, the temporary wrapper is removed and
    the preserved base body is promoted to a direct child of <worldbody>; only
    then can add_freejoint_to_base_body() legally add the floating-base joint.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if _local_tag(root.tag) != "robot":
        raise ValueError(f"Expected URDF root <robot>, got <{root.tag}>")

    link_names = {
        elem.attrib.get("name")
        for elem in root
        if _local_tag(elem.tag) == "link" and elem.attrib.get("name")
    }
    if base_body_name not in link_names:
        raise RuntimeError(
            f"URDF has no <link name='{base_body_name}'>. Available links include: "
            + ", ".join(sorted(link_names)[:20]) # type: ignore
        )

    child_links = set()
    for joint in root:
        if _local_tag(joint.tag) != "joint":
            continue
        for child in joint:
            if _local_tag(child.tag) == "child" and child.attrib.get("link"):
                child_links.add(child.attrib["link"])

    if base_body_name in child_links:
        raise RuntimeError(
            f"--base='{base_body_name}' is not the URDF root link; it already has a parent joint. "
            "Choose the actual URDF root link as --base."
        )

    mujoco_ext = next((e for e in root if _local_tag(e.tag) == "mujoco"), None)
    if mujoco_ext is None:
        mujoco_ext = ET.Element("mujoco")
        root.insert(0, mujoco_ext)

    compiler = next((e for e in mujoco_ext if _local_tag(e.tag) == "compiler"), None)
    if compiler is None:
        compiler = ET.SubElement(mujoco_ext, "compiler")

    compiler.set("fusestatic", "false")
    if enable_saveinertial:
        compiler.set("saveinertial", "true")

    synthetic_root = "__mujoco_world__"
    suffix = 1
    while synthetic_root in link_names:
        synthetic_root = f"__mujoco_world_{suffix}__"
        suffix += 1

    synthetic_joint = "__mujoco_world_to_base__"
    existing_joint_names = {
        elem.attrib.get("name")
        for elem in root
        if _local_tag(elem.tag) == "joint" and elem.attrib.get("name")
    }
    suffix = 1
    while synthetic_joint in existing_joint_names:
        synthetic_joint = f"__mujoco_world_to_base_{suffix}__"
        suffix += 1

    ET.SubElement(root, "link", {"name": synthetic_root})
    joint = ET.SubElement(root, "joint", {"name": synthetic_joint, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": synthetic_root})
    ET.SubElement(joint, "child", {"link": base_body_name})

    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(
        f"[OK] Preserved URDF root '{base_body_name}' as an MJCF body via synthetic root '{synthetic_root}'."
    )
    print('[OK] Set URDF compiler fusestatic="false" so the base body cannot be fused into worldbody.')
    if enable_saveinertial:
        print('[OK] Enabled compiler saveinertial="true" so saved MJCF bodies get explicit <inertial> clauses.')

    return synthetic_root


def promote_base_body_from_synthetic_root(
    xml_path: Path,
    base_body_name: str,
    synthetic_root_name: str,
) -> bool:
    """Promote the preserved base body to a direct child of ``<worldbody>``.

    During URDF compilation we temporarily add a synthetic URDF root so MuJoCo
    preserves the original URDF root link as a real MJCF ``<body>`` with its
    inertia.  The exported MJCF therefore has the temporary hierarchy::

        worldbody
          body synthetic_root
            body base_body

    A MuJoCo free joint is only legal on a top-level body, i.e. a body that is a
    direct child of ``<worldbody>``.  This function removes the temporary wrapper
    and moves ``base_body`` into the wrapper's former position before the
    freejoint is added.

    The synthetic root is created with an identity fixed transform and no
    physical content, so promotion preserves the base pose and dynamics.  The
    function fails loudly if the exported structure contains unexpected content
    instead of silently discarding it.
    """
    xml_path = xml_path.expanduser().resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if _local_tag(root.tag) != "mujoco":
        raise ValueError(f"Expected MJCF root <mujoco>, got <{root.tag}>")

    worldbody = _find_worldbody(root)

    synthetic_root: Optional[ET.Element] = None
    synthetic_index: Optional[int] = None
    for idx, child in enumerate(list(worldbody)):
        if _local_tag(child.tag) == "body" and child.attrib.get("name") == synthetic_root_name:
            synthetic_root = child
            synthetic_index = idx
            break

    if synthetic_root is None:
        # Idempotent behavior: if base is already top-level, there is nothing to do.
        for child in list(worldbody):
            if _local_tag(child.tag) == "body" and child.attrib.get("name") == base_body_name:
                print(f"[INFO] Base body '{base_body_name}' is already a direct child of <worldbody>.")
                return False
        raise RuntimeError(
            f"Synthetic root body '{synthetic_root_name}' was not found as a direct child of <worldbody>, "
            f"and base body '{base_body_name}' is not already top-level."
        )

    # The temporary synthetic root must have identity pose.  Our generated URDF
    # fixed joint has no <origin>, so any non-identity transform indicates that
    # the file was modified or the structure is not the one we created.
    transform_attrs = {"pos", "quat", "axisangle", "euler", "xyaxes", "zaxis"}
    unexpected_transforms = {k: v for k, v in synthetic_root.attrib.items() if k in transform_attrs}
    if unexpected_transforms:
        raise RuntimeError(
            f"Synthetic root '{synthetic_root_name}' unexpectedly has transform attributes "
            f"{unexpected_transforms}. Refusing to promote the base without composing transforms."
        )

    base_body: Optional[ET.Element] = None
    for child in list(synthetic_root):
        if _local_tag(child.tag) == "body" and child.attrib.get("name") == base_body_name:
            base_body = child
            break

    if base_body is None:
        direct_body_names = [
            child.attrib.get("name", "<unnamed>")
            for child in list(synthetic_root)
            if _local_tag(child.tag) == "body"
        ]
        raise RuntimeError(
            f"Base body '{base_body_name}' is not a direct child of synthetic root "
            f"'{synthetic_root_name}'. Direct child bodies are: {direct_body_names or ['<none>']}"
        )

    # The synthetic root is intentionally empty except for the preserved base.
    # With compiler saveinertial="true", MuJoCo may save an explicit zero-mass
    # <inertial> even for this dummy static wrapper.  That inertial is safe to
    # discard only after verifying that it carries no physical mass/inertia.
    unexpected_children = []
    for child in list(synthetic_root):
        if child is base_body:
            continue

        tag = _local_tag(child.tag)
        if tag == "inertial":
            try:
                mass = float(child.attrib.get("mass", "0"))
                diag = [float(x) for x in child.attrib.get("diaginertia", "0 0 0").split()]
                full = [float(x) for x in child.attrib.get("fullinertia", "").split()]
            except ValueError as exc:
                raise RuntimeError(
                    f"Synthetic root '{synthetic_root_name}' has an invalid inertial: {child.attrib}."
                ) from exc

            inertia_values = full if full else diag
            if abs(mass) <= 1e-12 and all(abs(x) <= 1e-12 for x in inertia_values):
                # Harmless serialization artifact from saveinertial=true.
                continue

        name = child.attrib.get("name")
        unexpected_children.append(f"<{tag}>" + (f" name='{name}'" if name else ""))

    if unexpected_children:
        raise RuntimeError(
            f"Synthetic root '{synthetic_root_name}' contains unexpected child elements: "
            + ", ".join(unexpected_children)
            + ". Refusing to remove the wrapper because doing so could discard model content."
        )

    assert synthetic_index is not None
    synthetic_root.remove(base_body)
    worldbody.remove(synthetic_root)
    worldbody.insert(synthetic_index, base_body)

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    print(
        f"[OK] Promoted body '{base_body_name}' from synthetic root '{synthetic_root_name}' "
        "to a direct child of <worldbody>."
    )
    return True


def assert_base_has_explicit_inertial(xml_path: Path, base_body_name: str) -> None:
    """Fail early if the exported MJCF base body has no explicit inertial."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    base = _find_body_by_name(root, base_body_name)
    if base is None:
        raise RuntimeError(f"Exported MJCF has no <body name='{base_body_name}'>.")
    inertial = next((c for c in list(base) if _local_tag(c.tag) == "inertial"), None)
    if inertial is None:
        raise RuntimeError(
            f"Exported MJCF body '{base_body_name}' has no explicit <inertial>. "
            "Refusing to continue because later collision-geom density=0 would make the base massless."
        )
    try:
        mass = float(inertial.attrib.get("mass", "0"))
    except ValueError as exc:
        raise RuntimeError(f"Invalid base inertial mass: {inertial.attrib.get('mass')}") from exc
    if mass <= 0:
        raise RuntimeError(f"Base body '{base_body_name}' has non-positive mass {mass}.")
    print(f"[OK] Verified explicit base inertial before geom post-processing: mass={mass:.9g}.")


def validate_final_mass_preservation(
    mujoco_module,
    xml_path: Path,
    expected_total_mass: float,
    base_body_name: str,
) -> None:
    """Reload the final MJCF and verify that post-processing did not change robot mass."""
    final_model = mujoco_module.MjModel.from_xml_path(str(xml_path))
    final_total_mass = float(sum(final_model.body_mass))
    mass_diff = abs(final_total_mass - expected_total_mass)
    mass_tol = 1e-5

    if mass_diff > mass_tol:
        raise RuntimeError(
            "Final MJCF total mass changed during post-processing: "
            f"before={expected_total_mass:.9f}, "
            f"after={final_total_mass:.9f}, "
            f"diff={mass_diff:.9e}, "
            f"tolerance={mass_tol:.1e}"
        )

    print(
        "[OK] Final mass validation passed: "
        f"before={expected_total_mass:.9f}, "
        f"after={final_total_mass:.9f}, "
        f"diff={mass_diff:.3e} kg."
    )

def format_mjcf_number(value: float) -> str:
    """Format a number for compact MJCF attributes."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _local_tag(tag: str) -> str:
    """Return local XML tag name, ignoring a namespace if one exists."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _insert_top_level_section(root: ET.Element, section: ET.Element) -> None:
    """Insert a top-level MJCF section in a readable location."""
    # For MuJoCo files, actuator is conventionally after worldbody/contact/equality/tendon.
    preferred_before = {"sensor", "keyframe"}
    for idx, child in enumerate(list(root)):
        if _local_tag(child.tag) in preferred_before:
            root.insert(idx, section)
            return
    root.append(section)


def _find_or_create_top_level_section(root: ET.Element, section_name: str, insert_before: Optional[set[str]] = None) -> ET.Element:
    """Find or create a top-level MJCF section."""
    for child in root:
        if _local_tag(child.tag) == section_name:
            return child

    section = ET.Element(section_name)
    if insert_before:
        for idx, child in enumerate(list(root)):
            if _local_tag(child.tag) in insert_before:
                root.insert(idx, section)
                return section
    root.append(section)
    return section


def _find_direct_child_by_name(parent: ET.Element, tag_name: str, name: str) -> Optional[ET.Element]:
    """Find a direct child by local tag and exact name attribute."""
    for child in list(parent):
        if _local_tag(child.tag) == tag_name and child.attrib.get("name") == name:
            return child
    return None


def ensure_light_and_floor_in_mjcf(xml_path: Path) -> Tuple[bool, bool, bool, bool]:
    """
    Add or update the default checker floor assets, directional light, and floor plane.

    Returns:
        (texture_created, material_created, light_created, floor_created)
    """
    xml_path = xml_path.expanduser().resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if _local_tag(root.tag) != "mujoco":
        raise ValueError(f"Expected MJCF root <mujoco>, got <{root.tag}>")

    # Keep assets before worldbody in a conventional MJCF order.
    asset = _find_or_create_top_level_section(root, "asset", insert_before={"worldbody", "contact", "actuator", "sensor", "keyframe"})

    texture = _find_direct_child_by_name(asset, "texture", "texplane")
    texture_created = texture is None
    if texture is None:
        texture = ET.Element("texture")
        asset.append(texture)
    texture.attrib.clear()
    texture.attrib.update(
        {
            "name": "texplane",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".2 .3 .4",
            "rgb2": ".1 0.15 0.2",
            "width": "512",
            "height": "512",
        }
    )

    material = _find_direct_child_by_name(asset, "material", "MatPlane")
    material_created = material is None
    if material is None:
        material = ET.Element("material")
        asset.append(material)
    material.attrib.clear()
    material.attrib.update(
        {
            "name": "MatPlane",
            "reflectance": "0.3",
            "texture": "texplane",
            "texrepeat": "1 1",
            "texuniform": "true",
        }
    )

    worldbody = _find_worldbody(root)

    light = _find_direct_child_by_name(worldbody, "light", "main_light")
    light_created = light is None
    if light is None:
        light = ET.Element("light")
        # Put light at the beginning of worldbody, before floor and robot body.
        worldbody.insert(0, light)
    light.attrib.clear()
    light.attrib.update(
        {
            "name": "main_light",
            "directional": "true",
            "pos": "-0.5 0.5 3",
            "dir": "0 0 -1",
            "castshadow": "false",
        }
    )

    floor = _find_direct_child_by_name(worldbody, "geom", "floor")
    floor_created = floor is None
    if floor is None:
        floor = ET.Element("geom")
        # Put floor right after the light when possible.
        insert_idx = 1 if len(list(worldbody)) >= 1 else 0
        worldbody.insert(insert_idx, floor)
    floor.attrib.clear()
    floor.attrib.update(
        {
            "name": "floor",
            "pos": "0 0 0",
            "size": "100 100 .125",
            "type": "plane",
            "material": "MatPlane",
            "condim": "3",
            "friction": "1 0.01 0.01",
            "group": "0",
        }
    )

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return texture_created, material_created, light_created, floor_created



def _find_or_create_direct_default(parent: ET.Element, class_name: Optional[str] = None) -> ET.Element:
    """Find or create a direct <default> child, optionally by class name."""
    for child in list(parent):
        if _local_tag(child.tag) != "default":
            continue
        if class_name is None and "class" not in child.attrib:
            return child
        if class_name is not None and child.attrib.get("class") == class_name:
            return child

    attrib = {} if class_name is None else {"class": class_name}
    created = ET.Element("default", attrib)
    parent.append(created)
    return created


def _find_or_create_direct_child(parent: ET.Element, tag_name: str) -> ET.Element:
    """Find or create a direct child with the given local tag name."""
    for child in list(parent):
        if _local_tag(child.tag) == tag_name:
            return child
    child = ET.Element(tag_name)
    parent.append(child)
    return child


def ensure_robot_default_classes_in_mjcf(xml_path: Path) -> None:
    """
    Ensure the generated MJCF has a compact default class hierarchy:

        default / robot / motor / visual / collision

    The values intentionally follow the user-provided template. Visual and collision
    geoms later only need class="visual" or class="collision".
    """
    xml_path = xml_path.expanduser().resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if _local_tag(root.tag) != "mujoco":
        raise ValueError(f"Expected MJCF root <mujoco>, got <{root.tag}>")

    # Put <default> before <asset>/<worldbody> when we create it.
    top_default = None
    for child in list(root):
        if _local_tag(child.tag) == "default" and "class" not in child.attrib:
            top_default = child
            break
    if top_default is None:
        top_default = ET.Element("default")
        insert_idx = 0
        for idx, child in enumerate(list(root)):
            if _local_tag(child.tag) in {"compiler", "option", "size", "statistic"}:
                insert_idx = idx + 1
        root.insert(insert_idx, top_default)

    robot_default = _find_or_create_direct_default(top_default, "robot")
    motor_default = _find_or_create_direct_default(robot_default, "motor")
    visual_default = _find_or_create_direct_default(robot_default, "visual")
    collision_default = _find_or_create_direct_default(robot_default, "collision")

    # Keep the child elements but make their attributes exactly match the desired defaults.
    motor_joint = _find_or_create_direct_child(motor_default, "joint")
    motor_joint.attrib.clear()
    motor_motor = _find_or_create_direct_child(motor_default, "motor")
    motor_motor.attrib.clear()

    visual_geom = _find_or_create_direct_child(visual_default, "geom")
    visual_geom.attrib.clear()
    visual_geom.attrib.update(
        {
            "material": "default_material",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        }
    )

    collision_geom = _find_or_create_direct_child(collision_default, "geom")
    collision_geom.attrib.clear()
    collision_geom.attrib.update(
        {
            "material": "collision_material",
            "condim": "3",
            "contype": "1",
            "conaffinity": "1",
            "solref": "0.005 1",
            "friction": "1 0.01 0.001",
            "density": "0",
            "group": "1",
        }
    )

    # Material definitions used by the default classes. Plane material is handled by
    # ensure_light_and_floor_in_mjcf().
    asset = _find_or_create_top_level_section(root, "asset", insert_before={"worldbody", "contact", "actuator", "sensor", "keyframe"})

    default_material = _find_direct_child_by_name(asset, "material", "default_material")
    if default_material is None:
        default_material = ET.Element("material")
        asset.insert(0, default_material)
    default_material.attrib.clear()
    default_material.attrib.update(
        {
            "name": "default_material",
            "rgba": "0.7 0.7 0.7 1",
        }
    )

    collision_material = _find_direct_child_by_name(asset, "material", "collision_material")
    if collision_material is None:
        collision_material = ET.Element("material")
        asset.insert(1 if len(list(asset)) >= 1 else 0, collision_material)
    collision_material.attrib.clear()
    collision_material.attrib.update(
        {
            "name": "collision_material",
            "rgba": "0.0 0.4 0.8 0.2",
        }
    )

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)



def _find_body_by_name(root: ET.Element, body_name: str) -> Optional[ET.Element]:
    """Find an MJCF <body> by exact name."""
    for body in root.iter():
        if _local_tag(body.tag) == "body" and body.attrib.get("name") == body_name:
            return body
    return None


def _collect_body_names(root: ET.Element, limit: int = 80) -> List[str]:
    """Collect body names for readable error messages."""
    names: List[str] = []
    for body in root.iter():
        if _local_tag(body.tag) != "body":
            continue
        name = body.attrib.get("name")
        if name:
            names.append(name)
    return names[:limit]


def body_has_direct_freejoint_or_free_joint(body: ET.Element) -> bool:
    """Check whether this body already has a direct freejoint/free joint child."""
    for child in list(body):
        tag = _local_tag(child.tag)
        if tag == "freejoint":
            return True
        if tag == "joint" and child.attrib.get("type", "hinge").lower() == "free":
            return True
    return False


def body_has_direct_non_free_joint(body: ET.Element) -> bool:
    """Check whether this body already has a direct non-free joint child."""
    for child in list(body):
        if _local_tag(child.tag) != "joint":
            continue
        if child.attrib.get("type", "hinge").lower() != "free":
            return True
    return False


def _find_worldbody(root: ET.Element) -> ET.Element:
    """Return the top-level <worldbody> section from an MJCF XML tree."""
    for child in root:
        if _local_tag(child.tag) == "worldbody":
            return child
    raise RuntimeError("MJCF XML has no top-level <worldbody> section.")


def _xml_has_any_freejoint(root: ET.Element) -> bool:
    """Check whether the MJCF already contains a free joint anywhere."""
    for elem in root.iter():
        tag = _local_tag(elem.tag)
        if tag == "freejoint":
            return True
        if tag == "joint" and elem.attrib.get("type", "hinge").lower() == "free":
            return True
    return False


def _wrap_worldbody_children_as_base(worldbody: ET.Element, base_body_name: str) -> ET.Element:
    """Wrap all existing direct <worldbody> children into a new base body."""
    old_children = list(worldbody)
    if not old_children:
        raise RuntimeError("Cannot create floating base: <worldbody> is empty.")

    base_body = ET.Element("body", {"name": base_body_name})
    for child in old_children:
        worldbody.remove(child)
        base_body.append(child)
    worldbody.append(base_body)
    return base_body


def add_freejoint_to_base_body(xml_path: Path, base_body_name: str) -> bool:
    """
    Add <freejoint/> using the user-specified --base name.

    Policy:
      - <body name=--base> must already exist in the exported MJCF.
      - Insert <freejoint/> into that body.
      - If the body is absent, fail instead of wrapping <worldbody> after compilation.

    The base body is now preserved before URDF compilation by
    prepare_urdf_root_for_floating_base(), so a missing base body indicates a real
    conversion problem. Wrapping worldbody children after compilation is unsafe
    because the URDF root-link inertial has already been discarded at that point.
    """
    xml_path = xml_path.expanduser().resolve()
    if not base_body_name or not base_body_name.strip():
        raise ValueError("--base must be a non-empty base body name, e.g. --base=base_link")
    base_body_name = base_body_name.strip()

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if _local_tag(root.tag) != "mujoco":
        raise ValueError(f"Expected MJCF root <mujoco>, got <{root.tag}>")

    if _xml_has_any_freejoint(root):
        print("[INFO] XML already contains a freejoint/free joint. Skip adding another one.")
        return False

    worldbody = _find_worldbody(root)
    target_body = _find_body_by_name(root, base_body_name)

    if target_body is None:
        available_names = _collect_body_names(root, limit=20)
        details = ", ".join(available_names) if available_names else "<none>"
        raise RuntimeError(
            f"MJCF body name='{base_body_name}' was not found after URDF compilation. "
            "Do not wrap worldbody children after compilation: that loses the URDF root-link inertia. "
            f"Existing body names include: {details}"
        )

    if body_has_direct_freejoint_or_free_joint(target_body):
        print(f"[INFO] Body '{base_body_name}' already has a freejoint/free joint. Skip adding another one.")
        return False

    if body_has_direct_non_free_joint(target_body):
        raise RuntimeError(
            f"Body '{base_body_name}' already has a direct non-free <joint>. "
            "A floating base body should not also have a hinge/slide joint as its direct joint. "
            "Please choose a root/base body instead."
        )

    freejoint_name = f"{base_body_name}_freejoint"
    target_body.insert(0, ET.Element("freejoint", {"name": freejoint_name}))

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    print(f"[OK] Added <freejoint name='{freejoint_name}'/> to body '{base_body_name}'.")
    print("[WARN] Floating-base qpos layout: qpos[0:3]=base_pos, qpos[3:7]=base_quat, qpos[7:]=joint_pos.")
    print("[WARN] Floating-base qvel layout: qvel[0:3]=base_lin_vel, qvel[3:6]=base_ang_vel, qvel[6:]=joint_vel.")
    return True


def _int_attr_is_zero(elem: ET.Element, attr_name: str) -> bool:
    """Return True if an integer-like MJCF attribute exists and is zero."""
    raw = elem.attrib.get(attr_name)
    if raw is None:
        return False
    try:
        return int(float(raw.strip())) == 0
    except ValueError:
        return False


def _geom_has_mesh(elem: ET.Element) -> bool:
    """Return True for MJCF geoms that use a mesh asset."""
    geom_type = elem.attrib.get("type", "").lower()
    return geom_type == "mesh" or "mesh" in elem.attrib


def _geom_looks_visual(elem: ET.Element) -> bool:
    """
    Detect a visual-only geom after MuJoCo URDF compilation.

    With <compiler discardvisual="false"/>, MuJoCo normally keeps URDF visual meshes
    as geoms with contact disabled, i.e. contype="0" and conaffinity="0".
    Some files also preserve names/classes containing "visual". Collision geoms may
    also use mesh assets, so mesh type alone is not enough to identify visuals.
    """
    if _int_attr_is_zero(elem, "contype") and _int_attr_is_zero(elem, "conaffinity"):
        return True

    text_fields = [
        elem.attrib.get("name", ""),
        elem.attrib.get("class", ""),
        elem.attrib.get("childclass", ""),
    ]
    return any("visual" in item.lower() for item in text_fields)


def assign_geom_classes_to_mjcf(xml_path: Path) -> Tuple[int, int]:
    """
    Assign visual/collision parameters to generated MJCF geoms.

    Important compatibility note:
        Some MuJoCo loading/merging paths reject per-element class="visual" on
        <geom>, even though default classes are useful as a readable parameter
        template. To keep the generated XML robust in this project, this function
        writes the effective visual/collision attributes directly onto each geom
        and removes any per-geom class attribute.

    Policy:
      - Visual-only mesh geoms -> material="default_material", contype=0,
        conaffinity=0, group=2.
      - Collision/contact geoms -> material="collision_material", condim=3,
        contype=1, conaffinity=1, solref="0.005 1",
        friction="1 0.01 0.001", density=0, group=1.
    """
    xml_path = xml_path.expanduser().resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if _local_tag(root.tag) != "mujoco":
        raise ValueError(f"Expected MJCF root <mujoco>, got <{root.tag}>")

    visual_count = 0
    collision_count = 0

    common_clear_attrs = (
        "class",
        "group",
        "contype",
        "conaffinity",
        "rgba",
        "material",
        "condim",
        "friction",
        "solref",
        "density",
    )

    for elem in root.iter():
        if _local_tag(elem.tag) != "geom":
            continue

        # Preserve environment geoms if the script is re-run on an XML that already has them.
        if elem.attrib.get("name") == "floor":
            continue

        is_visual = _geom_has_mesh(elem) and _geom_looks_visual(elem)

        for attr in common_clear_attrs:
            elem.attrib.pop(attr, None)

        if is_visual:
            elem.attrib.update(
                {
                    "material": "default_material",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "2",
                }
            )
            visual_count += 1
        else:
            elem.attrib.update(
                {
                    "material": "collision_material",
                    "condim": "3",
                    "contype": "1",
                    "conaffinity": "1",
                    "solref": "0.005 1",
                    "friction": "1 0.01 0.001",
                    "density": "0",
                    "group": "1",
                }
            )
            collision_count += 1

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return visual_count, collision_count

def add_motor_actuators_to_mjcf(
    xml_path: Path,
    force_limit: float = 99.0,
    skip_existing_actuated_joints: bool = True,
) -> int:
    """
    Add <motor> actuators for all scalar movable joints in a MuJoCo MJCF XML.

    Notes:
        - MJCF fixed joints do not appear as <joint> elements, so every hinge/slide joint
          found here is movable.
        - free and ball joints are skipped because a scalar <motor joint="..."/> is not
          appropriate for them.
        - ctrlrange and forcerange are both set to [-force_limit, force_limit].
    """
    xml_path = xml_path.expanduser().resolve()
    if force_limit <= 0:
        raise ValueError("motor force limit must be positive")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if _local_tag(root.tag) != "mujoco":
        raise ValueError(f"Expected MJCF root <mujoco>, got <{root.tag}>")

    actuator = None
    for child in root:
        if _local_tag(child.tag) == "actuator":
            actuator = child
            break
    if actuator is None:
        actuator = ET.Element("actuator")
        _insert_top_level_section(root, actuator)

    already_actuated_joints = set()
    for act in actuator:
        joint_name = act.attrib.get("joint")
        if joint_name:
            already_actuated_joints.add(joint_name)

    scalar_joint_names: List[str] = []
    seen = set()
    for joint in root.iter():
        if _local_tag(joint.tag) != "joint":
            continue
        name = joint.attrib.get("name")
        if not name or name in seen:
            continue
        seen.add(name)

        joint_type = joint.attrib.get("type", "hinge").lower()
        if joint_type in {"free", "ball"}:
            continue
        scalar_joint_names.append(name)

    added = 0
    limit_text = format_mjcf_number(abs(force_limit))
    range_text = f"-{limit_text} {limit_text}"

    for joint_name in scalar_joint_names:
        if skip_existing_actuated_joints and joint_name in already_actuated_joints:
            continue
        motor_name = f"{joint_name}_motor"
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": motor_name,
                "joint": joint_name,
                "ctrlrange": range_text,
                "ctrllimited": "true",
                "forcerange": range_text,
                "forcelimited": "true",
            },
        )
        added += 1

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return added

def convert_urdf_to_mjcf(
    urdf_path: Path,
    output_path: Path,
    fix_mesh_paths: bool,
    package_roots: Optional[Iterable[str]],
    package_map_items: Optional[Iterable[str]],
    keep_temp: bool,
    overwrite: bool,
    allow_missing_meshes: bool,
    keep_visuals: bool,
    add_motor_actuators: bool,
    motor_force_limit: float,
    base_body_name: str,
) -> None:
    """Load URDF with MuJoCo and save compiled MJCF XML."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'mujoco' is not installed. Install it first:\n"
            "    pip install mujoco"
        ) from exc

    urdf_path = urdf_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if urdf_path.suffix.lower() != ".urdf":
        print(f"[WARN] Input file does not end with .urdf: {urdf_path}", file=sys.stderr)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            f"Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="urdf2xml_")
    temp_dir = Path(temp_dir_obj.name)
    compile_path = urdf_path

    try:
        if fix_mesh_paths:
            package_map = build_auto_package_map(urdf_path, package_roots)
            package_map.update(parse_package_map(package_map_items))
            compile_path = rewrite_mesh_filenames(
                source_urdf=urdf_path,
                package_map=package_map,
                temp_dir=temp_dir,
                fail_on_missing_mesh=not allow_missing_meshes,
                keep_visuals=keep_visuals,
            )
        elif keep_visuals:
            temp_urdf = temp_dir / urdf_path.name
            temp_urdf.write_text(preserve_urdf_visuals(urdf_path.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
            compile_path = temp_urdf
            print('[INFO] Keeping URDF visual meshes: injected/updated <compiler discardvisual="false"/>.')

        # Always prepare a temporary URDF copy before compilation. The original URDF is never modified.
        if compile_path == urdf_path:
            temp_urdf = temp_dir / urdf_path.name
            shutil.copy2(urdf_path, temp_urdf)
            compile_path = temp_urdf

        saveinertial_supported = _version_tuple(getattr(mujoco, "__version__", "0.0.0")) >= (3, 3, 1)
        synthetic_root_name = prepare_urdf_root_for_floating_base(
            compile_path,
            base_body_name=base_body_name,
            enable_saveinertial=saveinertial_supported,
        )
        if not saveinertial_supported:
            print(
                f"[WARN] MuJoCo {getattr(mujoco, '__version__', 'unknown')} is older than 3.3.1; "
                "compiler saveinertial is unavailable. Explicit URDF inertials will still be preserved."
            )

        print(f"[INFO] Loading URDF: {urdf_path}")
        print(f"[INFO] Using prepared temporary URDF: {compile_path}")

        model = mujoco.MjModel.from_xml_path(str(compile_path))
        expected_total_mass = float(sum(model.body_mass))
        print(f"[INFO] Compiled robot total mass before MJCF post-processing: {expected_total_mass:.9g}")

        if output_path.exists() and overwrite:
            output_path.unlink()

        mujoco.mj_saveLastXML(str(output_path), model)
        print(f"[OK] Saved MJCF XML: {output_path}")

        # The synthetic URDF root was only needed during compilation to preserve
        # base_link as a real body with inertia.  Remove that temporary MJCF wrapper
        # now, so the base becomes top-level and can legally receive a freejoint.
        promote_base_body_from_synthetic_root(
            output_path,
            base_body_name=base_body_name,
            synthetic_root_name=synthetic_root_name,
        )

        # This must pass before assign_geom_classes_to_mjcf() sets collision density=0.
        assert_base_has_explicit_inertial(output_path, base_body_name)

        ensure_robot_default_classes_in_mjcf(output_path)
        print("[OK] Ensured robot default classes: robot / motor / visual / collision.")

        visual_count, collision_count = assign_geom_classes_to_mjcf(output_path)
        print(f"[OK] Assigned visual parameters to {visual_count} visual mesh geom(s).")
        print(f"[OK] Assigned collision parameters to {collision_count} collision/contact geom(s).")

        add_freejoint_to_base_body(output_path, base_body_name=base_body_name)

        texture_created, material_created, light_created, floor_created = ensure_light_and_floor_in_mjcf(output_path)
        print(
            "[OK] Ensured checker floor asset: "
            f"texture texplane ({'created' if texture_created else 'updated'}), "
            f"material MatPlane ({'created' if material_created else 'updated'})."
        )
        print(
            "[OK] Ensured world light/floor: "
            f"main_light ({'created' if light_created else 'updated'}), "
            f"floor ({'created' if floor_created else 'updated'})."
        )

        if add_motor_actuators:
            added = add_motor_actuators_to_mjcf(output_path, force_limit=motor_force_limit)
            limit_text = format_mjcf_number(abs(motor_force_limit))
            print(f"[OK] Added {added} motor actuator(s) for scalar movable joint(s).")
            print(
                f"[WARN] Motor ctrlrange/forcerange is set to -{limit_text} {limit_text}. "
                "Please edit the generated XML manually if your robot needs different torque limits."
            )

        validate_final_mass_preservation(
            mujoco,
            output_path,
            expected_total_mass=expected_total_mass,
            base_body_name=base_body_name,
        )

        if keep_temp and compile_path != urdf_path:
            temp_copy = output_path.with_name(output_path.stem + "_resolved_mesh_paths.urdf")
            shutil.copy2(compile_path, temp_copy)
            print(f"[OK] Saved temporary resolved URDF: {temp_copy}")

    finally:
        temp_dir_obj.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert URDF to MuJoCo MJCF XML using MuJoCo's official Python API."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the input .urdf file.",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Required MJCF body name to receive <freejoint/>, e.g. --base=base_link.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .xml path. Default: same folder and same basename as the URDF.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output XML if it already exists.",
    )
    parser.add_argument(
        "--no-mesh-fix",
        action="store_true",
        help="Do not rewrite mesh filename paths before compiling.",
    )
    parser.add_argument(
        "--no-package-fix",
        action="store_true",
        help="Deprecated alias of --no-mesh-fix. Kept for compatibility with the old script.",
    )
    parser.add_argument(
        "--package-root",
        action="append",
        default=None,
        help="Directory containing ROS package folders. Can be used multiple times.",
    )
    parser.add_argument(
        "--package-map",
        action="append",
        default=None,
        help="Explicit package path mapping, e.g. --package-map ZB26RD02B-URDF1.1=/home/user/ZB26RD02B-URDF1.1 . Can be used multiple times.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Also save the temporary URDF with resolved mesh paths next to the output XML.",
    )
    parser.add_argument(
        "--allow-missing-meshes",
        action="store_true",
        help="Continue to MuJoCo compile even if some mesh files cannot be resolved.",
    )
    parser.add_argument(
        "--discard-visual",
        action="store_true",
        help="Use MuJoCo's URDF default and discard pure visual meshes. By default this script keeps visuals.",
    )
    parser.add_argument(
        "--no-actuator",
        action="store_true",
        help="Do not add default motor actuators after URDF conversion.",
    )
    parser.add_argument(
        "--motor-force-limit",
        type=float,
        default=99.0,
        help="Absolute ctrlrange/forcerange limit for generated motor actuators. Default: 99.",
    )

    args = parser.parse_args()

    urdf_path = Path(args.path)
    output_path = Path(args.output) if args.output else urdf_path.expanduser().resolve().with_suffix(".xml")

    try:
        convert_urdf_to_mjcf(
            urdf_path=urdf_path,
            output_path=output_path,
            fix_mesh_paths=not (args.no_mesh_fix or args.no_package_fix),
            package_roots=args.package_root,
            package_map_items=args.package_map,
            keep_temp=args.keep_temp,
            overwrite=args.overwrite,
            allow_missing_meshes=args.allow_missing_meshes,
            keep_visuals=not args.discard_visual,
            add_motor_actuators=not args.no_actuator,
            motor_force_limit=args.motor_force_limit,
            base_body_name=args.base,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())