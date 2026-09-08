import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mujoco_sim.utils.urdf2xml import relativize_mjcf_mesh_paths


@pytest.mark.parametrize(
    ("xml_relative_path", "mesh_relative_path", "expected_file"),
    (
        ("robot.xml", "base.stl", "base.stl"),
        ("mjcf/robot.xml", "meshes/base.stl", "../meshes/base.stl"),
        (
            "exports/models/robot.xml",
            "assets/meshes/base.stl",
            "../../assets/meshes/base.stl",
        ),
    ),
)
def test_relativize_mjcf_mesh_paths_uses_output_directory(
    tmp_path: Path,
    xml_relative_path: str,
    mesh_relative_path: str,
    expected_file: str,
):
    xml_path = tmp_path / xml_relative_path
    mesh_path = tmp_path / mesh_relative_path
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.touch()
    xml_path.write_text(
        f'<mujoco><asset><mesh name="base" file="{mesh_path}"/></asset></mujoco>',
        encoding="utf-8",
    )

    rewritten_count = relativize_mjcf_mesh_paths(xml_path)

    mesh_file = ET.parse(xml_path).find("./asset/mesh").get("file")
    assert rewritten_count == 1
    assert mesh_file == expected_file
    assert not Path(mesh_file).is_absolute()
    assert (xml_path.parent / mesh_file).resolve() == mesh_path.resolve()


def test_relativize_mjcf_mesh_paths_leaves_existing_relative_paths_unchanged(
    tmp_path: Path,
):
    xml_path = tmp_path / "robot.xml"
    original_xml = '<mujoco><asset><mesh name="base" file="meshes/base.stl"/></asset></mujoco>'
    xml_path.write_text(original_xml, encoding="utf-8")

    rewritten_count = relativize_mjcf_mesh_paths(xml_path)

    assert rewritten_count == 0
    assert xml_path.read_text(encoding="utf-8") == original_xml
