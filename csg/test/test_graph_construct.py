from pathlib import Path

import numpy as np
import pytest
import trimesh

from csg import graph_construct


def _two_patch_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 4, 5],
            [0, 5, 1],
        ]
    )
    face_colors = np.array(
        [
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 255, 0, 255],
        ],
        dtype=np.uint8,
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_colors=face_colors,
        process=False,
    )


def test_segment_patches_preserve_face_ids_and_build_adjacency():
    mesh = _two_patch_mesh()

    patches = graph_construct.load_segment_patches(mesh)
    records, graph = graph_construct.build_patch_adjacency(mesh, patches)

    assert len(patches) == 2
    assert {tuple(patch.metadata["face_id"]) for patch in patches} == {
        (0, 1),
        (2, 3),
    }
    assert [record["patch_index"] for record in records] == [0, 1]
    assert records[0]["neighboring_patches"] == [1]
    assert records[1]["neighboring_patches"] == [0]
    assert sum(record["area"] for record in records) == pytest.approx(2.0)

    assert len(graph["edges"]) == 1
    edge = graph["edges"][0]
    assert {edge["patch0"], edge["patch1"]} == {0, 1}
    assert edge["length"] == pytest.approx(1.0)
    assert edge["dihedral_angle"] == pytest.approx(np.pi / 2.0)
    assert edge["boundary_v"].shape == (2, 3)


def test_prepare_patch_graph_merges_fit_and_graph_records(monkeypatch):
    input_mesh = _two_patch_mesh()

    def fake_fit(_patch, **_options):
        parameters = np.array([0.0, 0.0, 1.0, 0.0])
        return {
            "type": "plane",
            "fit_rate": 1.0,
            "params": parameters,
            "fits": {
                "plane": {"fit_rate": 1.0, "params": parameters},
            },
        }

    monkeypatch.setattr(graph_construct, "fit_patch_primitives", fake_fit)
    mesh, patches, results, graph = graph_construct.prepare_patch_graph(input_mesh)

    assert len(mesh.faces) == 4
    assert len(patches) == len(results) == len(graph["nodes"]) == 2
    for patch_index, result in enumerate(results):
        assert result["patch_index"] == patch_index
        assert result["type"] == "plane"
        assert np.array_equal(result["parameter"], result["params"])
        assert "area" in result
        assert "boundary_edges" in result
        assert "neighboring_patches" in result


def test_model_scale_rejects_implausibly_large_patch_sphere():
    mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    results = [
        {
            "type": "sphere",
            "fit_rate": 1.0,
            "params": np.array([100.0, 0.0, 0.0, 100.0]),
            "fits": {
                "sphere": {
                    "fit_rate": 1.0,
                    "params": np.array([100.0, 0.0, 0.0, 100.0]),
                },
                "plane": {
                    "fit_rate": 0.9,
                    "params": np.array([1.0, 0.0, 0.0, -1.0]),
                },
            },
        }
    ]

    graph_construct._enforce_model_fit_bounds(mesh, results)

    assert results[0]["fits"]["sphere"]["fit_rate"] == 0.0
    assert results[0]["fits"]["sphere"]["rejected_reason"] == "radius_exceeds_model_scale"
    assert results[0]["type"] == "plane"


def test_prepare_patch_graph_handles_single_degenerate_face():
    ply_path = Path(
        "example_data/final/ply/"
        "00140553_e2c0841b6c86e3bfdcc8c477_trimesh_000/gt.ply"
    )

    if not ply_path.is_file():
        pytest.skip(f"example data not available: {ply_path}")

    _, patches, results, _ = graph_construct.prepare_patch_graph(ply_path)

    assert len(patches) == len(results)
    degenerate = [
        index
        for index, patch in enumerate(patches)
        if not np.any(np.asarray(patch.area_faces) > 1e-12)
    ]
    assert degenerate
    for index in degenerate:
        assert results[index]["type"] == "invalid"
        assert results[index]["fit_rate"] == 0.0
