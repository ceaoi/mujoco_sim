#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a URDF file to MuJoCo MJCF XML.

Version: mesh-path-fix-keep-visual-2026-06-17

Usage:
    python urdf2xml.py --path=/path/to/robot.urdf

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

        print(f"[INFO] Loading URDF: {urdf_path}")
        if compile_path != urdf_path:
            print(f"[INFO] Using temporary URDF: {compile_path}")

        model = mujoco.MjModel.from_xml_path(str(compile_path))

        if output_path.exists() and overwrite:
            output_path.unlink()

        mujoco.mj_saveLastXML(str(output_path), model)
        print(f"[OK] Saved MJCF XML: {output_path}")

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
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())