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


def _mesh_contains_grid(target_mesh, axes, pitch):
    """Classify grid centers with one ray per XY column instead of per voxel."""
    shape = tuple(len(axis) for axis in axes)
    xy = np.stack(np.meshgrid(axes[0], axes[1], indexing="ij"), axis=-1).reshape(-1, 2)
    occupancy = np.zeros((len(xy), shape[2]), dtype=bool)
    bounds = target_mesh.bounds
    in_bounds = np.all((xy >= bounds[0, :2]) & (xy <= bounds[1, :2]), axis=1)
    ray_columns = np.flatnonzero(in_bounds)
    if not len(ray_columns):
        return occupancy.reshape(-1)

    origins = np.column_stack(
        (xy[ray_columns], np.full(len(ray_columns), bounds[0, 2] - pitch))
    )
    directions = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
    locations, ray_indices, _ = target_mesh.ray.intersects_location(
        origins, directions, multiple_hits=True
    )
    if not len(ray_indices):
        return occupancy.reshape(-1)
    order = np.lexsort((locations[:, 2], ray_indices))
    heights = locations[order, 2]
    ray_indices = ray_indices[order]
    indices, starts, counts = np.unique(
        ray_indices, return_index=True, return_counts=True
    )
    for ray_index, start, count in zip(indices, starts, counts):
        crossings = heights[start : start + count]
        crossings = crossings[np.r_[True, np.diff(crossings) > pitch * 1e-6]]
        # A closed mesh has an even number of crossings. A grazing ray may
        # touch an edge or vertex; check only those ambiguous columns exactly.
        column = ray_columns[ray_index]
        if len(crossings) % 2:
            column_points = np.column_stack(
                (np.repeat(xy[column][None, :], shape[2], axis=0), axes[2])
            )
            occupancy[column] = target_mesh.contains(column_points)
        else:
            occupancy[column] = np.searchsorted(
                crossings, axes[2], side="right"
            ) % 2 == 1
    return occupancy.reshape(-1)


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
        if not target_mesh.is_watertight:
            raise ValueError("open mesh requires voxel occupancy")
        target = _mesh_contains_grid(target_mesh, axes, pitch)
    except Exception:
        # Preserve the previous validation fallback for ray backends that
        # cannot process a particular mesh or degenerate column.
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
