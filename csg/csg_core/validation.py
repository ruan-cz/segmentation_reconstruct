"""CSG occupancy evaluation and geometry validation."""

import numpy as np
import trimesh

from . import candidates as candidate_module


def _node_bounds(node):
    operation = node["op"]
    if operation == "UNION":
        bounds = [_node_bounds(child) for child in node["children"]]
        return np.stack(
            [
                np.min([item[0] for item in bounds], axis=0),
                np.max([item[1] for item in bounds], axis=0),
            ]
        )
    if operation == "INTERSECTION":
        bounds = [_node_bounds(child) for child in node["children"]]
        lower = np.max([item[0] for item in bounds], axis=0)
        upper = np.min([item[1] for item in bounds], axis=0)
        return np.stack([lower, np.maximum(lower, upper)])
    if operation == "DIFFERENCE":
        return _node_bounds(node["children"][0])
    if operation == "EMPTY":
        return np.zeros((2, 3))
    candidate = candidate_module.PrimitiveCandidate(
        operation,
        node["parameters"],
        node.get("patch_ids", []),
        0.0,
        0.0,
    )
    return candidate_module.primitive_bounds(candidate)


def csg_contains(node, points):
    """Evaluate whether each point lies inside a CSG IR node."""
    operation = node["op"]
    if operation == "EMPTY":
        return np.zeros(len(points), dtype=bool)
    if operation == "UNION":
        return np.any([csg_contains(child, points) for child in node["children"]], axis=0)
    if operation == "INTERSECTION":
        return np.all([csg_contains(child, points) for child in node["children"]], axis=0)
    if operation == "DIFFERENCE":
        inside = csg_contains(node["children"][0], points)
        for child in node["children"][1:]:
            inside &= ~csg_contains(child, points)
        return inside
    return candidate_module.primitive_contains(operation, node["parameters"], points)


def validate_csg(
    csg_ir,
    selected_candidates,
    results,
    target_mesh=None,
    voxel_resolution=48,
    maximum_voxels=180000,
):
    """Measure patch coverage and, optionally, approximate voxel agreement."""
    covered = {patch for candidate in selected_candidates for patch in candidate.patch_ids}
    patch_ids = {int(result.get("patch_index", index)) for index, result in enumerate(results)}
    patch_area = {
        int(result.get("patch_index", index)): float(result.get("area", result["mesh"].area))
        for index, result in enumerate(results)
    }
    total_area = max(sum(patch_area.values()), candidate_module.EPSILON)
    report = {
        "surface_coverage": float(
            sum(patch_area.get(index, 0.0) for index in covered) / total_area
        ),
        "missing_patch_ids": sorted(patch_ids - covered),
        "mean_fitting_error": (
            float(
                np.mean(
                    [candidate.fitting_error for candidate in selected_candidates]
                )
            )
            if selected_candidates
            else np.inf
        ),
        "voxel_iou": None,
        "missing_volume_ratio": None,
        "extra_volume_ratio": None,
        "volume_consistency": None,
    }
    if target_mesh is None or csg_ir["op"] == "EMPTY":
        return report
    target_mesh = (
        target_mesh.dump(concatenate=True)
        if isinstance(target_mesh, trimesh.Scene)
        else target_mesh
    )
    bounds = np.stack(
        [
            np.minimum(target_mesh.bounds[0], _node_bounds(csg_ir)[0]),
            np.maximum(target_mesh.bounds[1], _node_bounds(csg_ir)[1]),
        ]
    )
    extent = bounds[1] - bounds[0]
    pitch = max(np.max(extent) / voxel_resolution, candidate_module.EPSILON)
    counts = np.maximum(1, np.ceil(extent / pitch).astype(int))
    while int(np.prod(counts)) > maximum_voxels:
        pitch *= 1.1
        counts = np.maximum(1, np.ceil(extent / pitch).astype(int))
    axes = [bounds[0, axis] + (np.arange(counts[axis]) + 0.5) * pitch for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    predicted = csg_contains(csg_ir, grid)
    try:
        target = target_mesh.contains(grid)
    except Exception:
        voxel_grid = target_mesh.voxelized(pitch).fill()
        target = voxel_grid.is_filled(grid)
    intersection = np.count_nonzero(predicted & target)
    union = np.count_nonzero(predicted | target)
    target_count = max(np.count_nonzero(target), 1)
    predicted_count = max(np.count_nonzero(predicted), 1)
    report.update(
        {
            "voxel_iou": float(intersection / max(union, 1)),
            "missing_volume_ratio": float(
                np.count_nonzero(target & ~predicted) / target_count
            ),
            "extra_volume_ratio": float(
                np.count_nonzero(predicted & ~target) / predicted_count
            ),
            "volume_consistency": float(
                min(target_count, predicted_count)
                / max(target_count, predicted_count)
            ),
            "voxel_pitch": float(pitch),
        }
    )
    return report


class CSGValidator:
    """Evaluate CSG occupancy and reconstruction quality."""

    contains = staticmethod(csg_contains)

    def __init__(self, voxel_resolution=48, maximum_voxels=180000):
        self.voxel_resolution = int(voxel_resolution)
        self.maximum_voxels = int(maximum_voxels)

    def validate(self, csg_ir, selected_candidates, results, target_mesh=None):
        return validate_csg(
            csg_ir,
            selected_candidates,
            results,
            target_mesh=target_mesh,
            voxel_resolution=self.voxel_resolution,
            maximum_voxels=self.maximum_voxels,
        )


__all__ = ["CSGValidator", "csg_contains", "validate_csg"]
