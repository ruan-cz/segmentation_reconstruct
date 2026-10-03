import numpy as np
import trimesh

from csg.csg_core.validation import _mesh_contains_grid
from csg.graph_construct import fit_patch_primitives


def test_grid_occupancy_matches_exact_contains_for_disconnected_solids():
    first = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    second = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    second.apply_translation([0.0, 0.0, 2.0])
    mesh = trimesh.util.concatenate((first, second))
    axes = [
        np.linspace(-0.8, 0.8, 7),
        np.linspace(-0.8, 0.8, 7),
        np.linspace(-0.8, 2.8, 15),
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    actual = _mesh_contains_grid(mesh, axes, pitch=0.2)

    np.testing.assert_array_equal(actual, mesh.contains(grid))


def test_single_triangle_patch_can_be_fitted():
    patch = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )

    result = fit_patch_primitives(patch)

    assert result["type"] == "plane"
