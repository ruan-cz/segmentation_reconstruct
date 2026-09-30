"""Primitive data, fitting, and candidate generation."""

import dataclasses
import itertools

import numpy as np
import scipy.optimize
import scipy.spatial
import trimesh


EPSILON = 1e-9


# Candidate data model and shared helpers


@dataclasses.dataclass
class PrimitiveCandidate:
    """One geometric primitive hypothesis associated with surface patches."""

    primitive_type: str
    parameters: dict
    patch_ids: list
    fitting_error: float
    confidence: float
    operation: str = "UNKNOWN"
    metadata: dict = dataclasses.field(default_factory=dict)

    def as_dict(self):
        return {
            "primitive_type": self.primitive_type,
            "parameters": self.parameters,
            "patch_ids": list(self.patch_ids),
            "fitting_error": float(self.fitting_error),
            "confidence": float(self.confidence),
            "operation": self.operation,
            "metadata": self.metadata,
        }


def _normalize(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length <= EPSILON:
        return None
    return vector / length


def _rotation_equivalent(first, second, tolerance_degrees=3.0):
    cosine = np.cos(np.deg2rad(tolerance_degrees))
    alignment = np.abs(np.asarray(first).T @ np.asarray(second))
    return bool(np.all(alignment.max(axis=0) >= cosine))


def _seed_fit(result, primitive_type):
    key = primitive_type.lower()
    fit = result.get("fits", {}).get(key, {})
    rate = float(fit.get("fit_rate", 0.0))
    parameters = np.asarray(fit.get("params", []), dtype=float).reshape(-1)
    if str(result.get("type", "")).upper() == primitive_type.upper():
        rate = max(rate, float(result.get("fit_rate", 0.0)))
        if parameters.size == 0:
            parameters = np.asarray(
                result.get("parameter", result.get("params", [])), dtype=float
            ).reshape(-1)
    return rate, parameters


# Plane-based cube reconstruction


def _get_plane_patch(results, min_plane_fit_rate=0.8):
    planes = {}
    for fallback_index, result in enumerate(results):
        fit_rate, parameters = _seed_fit(result, "PLANE")
        if fit_rate < min_plane_fit_rate:
            continue
        if parameters.size < 3:
            continue
        normal_length = np.linalg.norm(parameters[:3])
        if normal_length <= EPSILON:
            continue
        normal = parameters[:3] / normal_length
        mesh = result["mesh"]
        offset = (
            float(parameters[3] / normal_length)
            if parameters.size >= 4
            else -float(np.dot(normal, np.asarray(mesh.vertices).mean(axis=0)))
        )
        patch_id = int(result.get("patch_index", fallback_index))
        planes[patch_id] = {
            "patch_id": patch_id,
            "normal": normal,
            "offset": offset,
            "mesh": mesh,
            "fit_rate": fit_rate,
            "area": float(result.get("area", mesh.area)),
        }
    return planes

"""
Build local frames directly from fitted plane normals.
"""
def _infer_cube_coordinate_systems(
    planes,
    normal_tolerance_degrees=12.0,
    orthogonal_tolerance_degrees=12.0,
    max_frames=8,
):
    
    normals = []
    weights = []
    for plane in planes.values():
        normals.append(plane["normal"])
        weights.append(plane["area"])
    if not normals:
        return []
    normals = np.asarray(normals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    clusters = []
    cosine = np.cos(np.deg2rad(normal_tolerance_degrees))
    for normal, weight in zip(normals, weights):
        selected = None
        for cluster in clusters:
            if abs(np.dot(cluster["direction"], normal)) >= cosine:
                selected = cluster
                break
        if selected is None:
            clusters.append({"direction": normal.copy(), "weight": weight})
        else:
            aligned = normal if np.dot(selected["direction"], normal) >= 0 else -normal
            direction = selected["direction"] * selected["weight"] + aligned * weight
            selected["direction"] = _normalize(direction)
            selected["weight"] += weight

    def frame_score(rotation):
        alignment = np.abs(normals @ rotation)
        residual = 1.0 - alignment.max(axis=1)
        return float(1.0 - np.average(residual, weights=weights))

    frames = []
    identity = np.eye(3)
    identity_score = frame_score(identity)
    if identity_score >= cosine:
        frames.append({"rotation": identity, "score": identity_score, "axis_aligned": True})
    orthogonal_limit = np.sin(np.deg2rad(orthogonal_tolerance_degrees))
    for first, second in itertools.combinations(clusters, 2):
        direction0 = first["direction"]
        direction1 = second["direction"]
        if abs(np.dot(direction0, direction1)) > orthogonal_limit:
            continue
        direction1 = _normalize(direction1 - np.dot(direction0, direction1) * direction0)
        if direction1 is None:
            continue
        direction2 = _normalize(np.cross(direction0, direction1))
        rotation = np.column_stack([direction0, direction1, direction2])
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
        score = frame_score(rotation)
        if score < cosine:
            continue
        if any(_rotation_equivalent(rotation, frame["rotation"]) for frame in frames):
            continue
        frames.append({"rotation": rotation, "score": score, "axis_aligned": False})

    # Assemblies of rotated blocks can contain several unrelated orthogonal
    # frames.  The old global score rejected all of them because every plane
    # had to align to one frame.  Add local frames supported by a substantial
    # inlier area; unrelated faces are ignored for this frame and can be
    # handled by another candidate on the next pass.
    total_weight = max(float(np.sum(weights)), EPSILON)
    for first, second in itertools.combinations(normals, 2):
        if abs(float(np.dot(first, second))) > orthogonal_limit:
            continue
        direction0 = _normalize(first)
        direction1 = _normalize(second - np.dot(direction0, second) * direction0)
        if direction0 is None or direction1 is None:
            continue
        direction2 = _normalize(np.cross(direction0, direction1))
        if direction2 is None:
            continue
        rotation = np.column_stack([direction0, direction1, direction2])
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
        alignment = np.abs(normals @ rotation)
        inliers = alignment.max(axis=1) >= cosine
        support = float(np.sum(weights[inliers]) / total_weight)
        if support < 0.12:
            continue
        local_score = float(np.average(alignment[inliers].max(axis=1), weights=weights[inliers]))
        if any(_rotation_equivalent(rotation, frame["rotation"]) for frame in frames):
            continue
        frames.append(
            {
                "rotation": rotation,
                "score": local_score * support,
                "axis_aligned": False,
            }
        )
    frames.sort(key=lambda frame: (not frame["axis_aligned"], -frame["score"]))
    return frames[:max_frames]


def _local_plane_surfaces(planes, rotation, normal_tolerance_degrees=12.0):
    cosine = np.cos(np.deg2rad(normal_tolerance_degrees))
    surfaces = {}
    for patch_id, plane in planes.items():
        mesh = plane["mesh"]
        # compute nearest axis family: 0, 1, 2 for x, y, z
        normal = plane["normal"]
        local_normal = rotation.T @ normal
        family = int(np.argmax(np.abs(local_normal)))
        if abs(local_normal[family]) < cosine:
            continue
        local_vertices = np.asarray(mesh.vertices, dtype=float) @ rotation
        bounds = np.stack([local_vertices.min(axis=0), local_vertices.max(axis=0)])
        surfaces[patch_id] = {
            "patch_id": patch_id,
            "family": family,
            "position": float(-plane["offset"] / local_normal[family]),
            "bounds": bounds,
            "normal_error": float(1.0 - abs(local_normal[family])),
            "area": plane["area"],
        }
    return surfaces


def _aggregated_graph(graph):
    neighbors = {int(node["patch_index"]): {} for node in graph["nodes"]}
    for edge in graph["edges"]:
        first = int(edge["patch0"])
        second = int(edge["patch1"])
        length = float(edge.get("length", 0.0))
        angle = float(edge.get("dihedral_angle", 0.0))
        for source, target in ((first, second), (second, first)):
            item = neighbors.setdefault(source, {}).setdefault(
                target, {"length": 0.0, "weighted_angle": 0.0}
            )
            item["length"] += length
            item["weighted_angle"] += length * angle
    for adjacent in neighbors.values():
        for item in adjacent.values():
            item["dihedral_angle"] = item["weighted_angle"] / max(
                item["length"], EPSILON
            )
    return neighbors


def _cube_patch_compatibility(first, second, edge, tolerance):
    if edge["length"] <= tolerance:
        return -np.inf
    if first["family"] == second["family"]:
        if abs(first["position"] - second["position"]) > tolerance:
            return -np.inf
        return 1.0 + np.log1p(edge["length"])
    overlap = np.maximum(
        0.0,
        np.minimum(first["bounds"][1], second["bounds"][1]) - np.maximum(first["bounds"][0], second["bounds"][0]),
    )
    shared_axis = ({0, 1, 2} - {first["family"], second["family"]}).pop()
    if overlap[shared_axis] <= tolerance:
        return -np.inf
    angle_error = abs(edge["dihedral_angle"] - np.pi / 2.0) / (np.pi / 2.0)
    return float(2.0 + np.log1p(edge["length"]) - 0.5 * angle_error)


def _region_bounds(region, surfaces):
    bounds = np.stack([surfaces[index]["bounds"] for index in region])
    return np.stack([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])


def _clustered_support_count(values, tolerance):
    clusters = []
    for value in sorted(values):
        if not clusters or abs(value - clusters[-1]) > tolerance:
            clusters.append(value)
    return len(clusters)


def _valid_cube_region(region, surfaces, tolerance):
    if not region:
        return False
    bounds = _region_bounds(region, surfaces)
    if np.any(bounds[1] - bounds[0] <= tolerance):
        return False
    for axis in range(3):
        positions = [
            surfaces[index]["position"]
            for index in region
            if surfaces[index]["family"] == axis
        ]
        if _clustered_support_count(positions, tolerance) > 2:
            return False
    for index in region:
        surface = surfaces[index]
        axis = surface["family"]
        distance = min(
            abs(surface["position"] - bounds[0, axis]),
            abs(surface["position"] - bounds[1, axis]),
        )
        if distance > tolerance:
            return False
    return True


def _cube_region_score(region, surfaces, neighbors):
    internal_length = 0.0
    region_set = set(region)
    for source in region:
        for target, edge in neighbors.get(source, {}).items():
            if target in region_set and source < target:
                internal_length += edge["length"]
    area = sum(surfaces[index]["area"] for index in region)
    normal_error = np.mean([surfaces[index]["normal_error"] for index in region])
    return float(len(region) + np.log1p(area + internal_length) - normal_error)


def _partition_cube_surfaces(surfaces, graph, tolerance):
    """
    Partition compatible planar surfaces into disjoint cube regions.

    Returns ``(partition, alternatives)``: the disjoint winning regions and
    the losing region-growing trials.  A losing trial still describes a
    plausible solid when coplanar patches it needs were absorbed by a larger
    neighboring region (for example a beam whose flush side faces also
    belong to an attached rotated bar); emitting it as an extra candidate
    lets the global selection decide instead of the greedy size ordering.
    """
    neighbors = _aggregated_graph(graph)

    def grow_region(seed, available):
        selected = [seed]
        selected_set = {seed}
        while True:
            options = {}
            for current in selected:
                for candidate, edge in neighbors.get(current, {}).items():
                    if candidate not in available or candidate in selected_set:
                        continue
                    compatibility = _cube_patch_compatibility(
                        surfaces[current], surfaces[candidate], edge, tolerance
                    )
                    if np.isfinite(compatibility):
                        orthogonal = (
                            surfaces[current]["family"]
                            != surfaces[candidate]["family"]
                        )
                        previous = options.get(candidate)
                        proposal = (orthogonal, compatibility)
                        if previous is None or proposal > previous:
                            options[candidate] = proposal

            accepted = None
            for candidate, (orthogonal, _) in sorted(
                options.items(), key=lambda item: item[1], reverse=True
            ):
                selected_families = {
                    surfaces[index]["family"] for index in selected
                }
                if not orthogonal and len(selected_families) >= 2:
                    current_bounds = _region_bounds(selected, surfaces)
                    family = surfaces[candidate]["family"]
                    tangential_axes = [axis for axis in range(3) if axis != family]
                    candidate_bounds = surfaces[candidate]["bounds"]
                    expands_tangential_extent = any(
                        candidate_bounds[0, axis]
                        < current_bounds[0, axis] - tolerance
                        or candidate_bounds[1, axis]
                        > current_bounds[1, axis] + tolerance
                        for axis in tangential_axes
                    )
                    if expands_tangential_extent:
                        continue
                if _valid_cube_region(
                    selected + [candidate], surfaces, tolerance
                ):
                    accepted = candidate
                    break
            if accepted is None:
                return selected
            selected.append(accepted)
            selected_set.add(accepted)

    eligible = set(surfaces)
    subgraphs = []
    alternatives = []
    covered = set()
    while eligible:
        trials = [
            grow_region(seed, eligible)
            for seed in eligible
        ]
        region = max(
            trials,
            key=lambda item: (len(item), _cube_region_score(item, surfaces, neighbors)),
        )
        region_set = set(region)
        for trial in trials:
            trial_set = set(trial)
            if trial_set and not trial_set <= (region_set | covered):
                alternatives.append(sorted(trial_set))
        subgraphs.append(sorted(region_set))
        covered |= region_set
        eligible.difference_update(region_set)
    return subgraphs, alternatives


def _matrix_to_euler_xyz(rotation):
    rotation = np.asarray(rotation, dtype=float)
    value = np.clip(-rotation[2, 0], -1.0, 1.0)
    y_angle = np.arcsin(value)
    if abs(np.cos(y_angle)) > 1e-7:
        x_angle = np.arctan2(rotation[2, 1], rotation[2, 2])
        z_angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x_angle = np.arctan2(-rotation[1, 2], rotation[1, 1])
        z_angle = 0.0
    return np.rad2deg([x_angle, y_angle, z_angle])


def _cube_candidate(region, surfaces, rotation, frame_score, axis_aligned):
    local_bounds = _region_bounds(region, surfaces)
    local_center = local_bounds.mean(axis=0)
    size = local_bounds[1] - local_bounds[0]
    center = rotation @ local_center
    scale = max(np.linalg.norm(size), EPSILON)
    distance_error = np.mean(
        [
            min(
                abs(surfaces[index]["position"] - local_bounds[0, surfaces[index]["family"]]),
                abs(surfaces[index]["position"] - local_bounds[1, surfaces[index]["family"]]),
            )
            / scale
            for index in region
        ]
    )
    normal_error = np.mean([surfaces[index]["normal_error"] for index in region])
    fitting_error = float(distance_error + normal_error)
    constraint_factor = min(1.0, len(region) / 3.0)
    confidence = float(np.exp(-8.0 * fitting_error) * frame_score * constraint_factor)
    return PrimitiveCandidate(
        primitive_type="CUBE",
        parameters={
            "center": center,
            "size": size,
            "rotation": rotation,
            "euler_xyz_degrees": _matrix_to_euler_xyz(rotation),
            "local_bounds": local_bounds,
        },
        patch_ids=sorted(region),
        fitting_error=fitting_error,
        confidence=confidence,
        operation="ADD",
        metadata={"axis_aligned": axis_aligned, "frame_score": frame_score},
    )


def generate_cube_candidates(
    results,
    graph,
    tolerance=None,
    min_patch_count=3,
    min_plane_fit_rate=0.8,
):
    """Generate axis-aligned and rotated cubes from compatible plane patches."""
    all_vertices = np.vstack([np.asarray(result["mesh"].vertices) for result in results])
    model_scale = max(np.linalg.norm(np.ptp(all_vertices, axis=0)), 1.0)
    tolerance = tolerance or model_scale * 1e-5
    candidates = []
    planes = _get_plane_patch(results, min_plane_fit_rate)

    for frame in _infer_cube_coordinate_systems(planes):
        rotation = frame["rotation"]
        surfaces = _local_plane_surfaces(planes, rotation)
        partition, alternatives = _partition_cube_surfaces(surfaces, graph, tolerance)
        for region in partition + alternatives:
            axis_count = len({surfaces[index]["family"] for index in region})
            if len(region) < min_patch_count or axis_count < 2:
                continue
            candidates.append(
                _cube_candidate(
                    region,
                    surfaces,
                    rotation,
                    frame["score"],
                    frame["axis_aligned"],
                )
            )
    unique = {}
    for candidate in candidates:
        key = tuple(candidate.patch_ids)
        previous = unique.get(key)
        if previous is None or candidate.fitting_error < previous.fitting_error:
            unique[key] = candidate
    return list(unique.values())


# Curved primitive fitting and candidate generation


def _result_lookup(results):
    """Map stable patch identifiers to their fitted result records."""
    lookup = {}
    for index, result in enumerate(results):
        patch_id = int(result.get("patch_index", index))
        lookup[patch_id] = result
    return lookup


def _sample_patch_points(results, patch_ids, max_points=6000, return_patch_ids=False):
    result_lookup = _result_lookup(results)
    points = []
    normals = []
    source_patch_ids = []
    for patch_id in patch_ids:
        if int(patch_id) not in result_lookup:
            raise KeyError(f"unknown patch id: {patch_id}")
        mesh = result_lookup[int(patch_id)]["mesh"]
        patch_points = np.asarray(mesh.vertices, dtype=float)
        patch_normals = np.asarray(mesh.vertex_normals, dtype=float)
        points.append(patch_points)
        normals.append(patch_normals)
        source_patch_ids.append(np.full(len(patch_points), int(patch_id), dtype=int))
    points = np.vstack(points)
    normals = np.vstack(normals)
    source_patch_ids = np.concatenate(source_patch_ids)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        normals = normals[indices]
        source_patch_ids = source_patch_ids[indices]
    if return_patch_ids:
        return points, normals, source_patch_ids
    return points, normals


def _point_in_polygon(points, polygon):
    """Return a mask for points inside a 2-D polygon using ray casting."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    inside = np.zeros(len(points), dtype=bool)
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (first[1] > points[:, 1]) != (second[1] > points[:, 1])
        denominator = second[1] - first[1]
        denominator = denominator if abs(denominator) > EPSILON else EPSILON
        intersections = (second[0] - first[0]) * (points[:, 1] - first[1]) / denominator + first[0]
        inside ^= crosses & (points[:, 0] < intersections)
    return inside


def _polygon_boundary_distance(points, polygon):
    """Return the distance of each 2-D point to the polygon boundary."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(points) == 0 or len(polygon) < 2:
        return np.zeros(len(points), dtype=float)
    starts = polygon
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths_squared = np.maximum(np.sum(edges**2, axis=1), EPSILON)
    relative = points[:, None, :] - starts[None, :, :]
    projection = np.clip(
        np.sum(relative * edges[None, :, :], axis=2) / lengths_squared[None, :],
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + projection[:, :, None] * edges[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def _simplify_polygon(polygon, tolerance):
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) <= 3:
        return polygon
    polygon = polygon.copy()
    changed = True
    while changed and len(polygon) > 3:
        changed = False
        distances = []
        for index in range(len(polygon)):
            previous = polygon[index - 1]
            current = polygon[index]
            following = polygon[(index + 1) % len(polygon)]
            edge = following - previous
            length = max(float(np.linalg.norm(edge)), EPSILON)
            cross_value = edge[0] * (current[1] - previous[1]) - edge[1] * (
                current[0] - previous[0]
            )
            distance = abs(float(cross_value)) / length
            distances.append(distance)
        index = int(np.argmin(distances))
        if distances[index] <= tolerance:
            polygon = np.delete(polygon, index, axis=0)
            changed = True
    return polygon


def _extrusion_axis_hypotheses(normals, tolerance_degrees=12.0, side_tolerance_degrees=15.0):
    """Return clustered extrusion-axis hypotheses, best first.

    One connected set of planar patches can host several prisms with
    different axes (for example a top boss and a side slot whose faces are
    connected through the cube faces of the base body), so every
    sufficiently supported axis is returned instead of only the global
    optimum.
    """
    normals = np.asarray(normals, dtype=float)
    normals = normals[np.linalg.norm(normals, axis=1) > EPSILON]
    if len(normals) < 3:
        return []
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    proposals = []
    covariance = normals.T @ normals / len(normals)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    proposals.append(eigenvectors[:, np.argmin(eigenvalues)])
    for first, second in itertools.combinations(normals, 2):
        cross = _normalize(np.cross(first, second))
        if cross is not None:
            proposals.append(cross)
    side_limit = np.sin(np.deg2rad(side_tolerance_degrees))
    cosine = np.cos(np.deg2rad(tolerance_degrees))
    scored = []
    for proposal in proposals:
        axis = _normalize(proposal)
        if axis is None:
            continue
        side_mask = np.abs(normals @ axis) <= side_limit
        if np.count_nonzero(side_mask) < 3:
            continue
        residual = float(np.mean(np.abs(normals[side_mask] @ axis)))
        side_normals = normals[side_mask]
        clusters = []
        for normal in side_normals:
            for cluster in clusters:
                if abs(float(np.dot(cluster[0], normal))) >= cosine:
                    cluster[1] += 1
                    break
            else:
                clusters.append([normal, 1])
        largest_cluster = max(cluster[1] for cluster in clusters)
        diversity = 1.0 - largest_cluster / len(side_normals)
        score = (diversity, int(np.count_nonzero(side_mask)), -residual)
        scored.append((score, axis, residual))
    scored.sort(key=lambda item: item[0], reverse=True)
    hypotheses = []
    for score, axis, residual in scored:
        if any(abs(float(np.dot(axis, kept[1]))) >= cosine for kept in hypotheses):
            continue
        hypotheses.append((score, axis, residual))
    return hypotheses


def _extrusion_axis_from_normals(normals):
    hypotheses = _extrusion_axis_hypotheses(normals)
    if not hypotheses:
        return None, np.inf
    _, axis, residual = hypotheses[0]
    return axis, residual


def _fit_extrusion_group(results, graph, side_ids, plane_ids):
    """Fit a polygonal prism from planar side patches and optional caps."""
    result_lookup = _result_lookup(results)
    side_ids = sorted(set(side_ids))
    if len(side_ids) < 3:
        raise ValueError("an extrusion needs at least three side patches")
    normal_by_patch = {}
    points_by_patch = {}
    for patch_id in side_ids:
        result = result_lookup[patch_id]
        _, parameters = _seed_fit(result, "PLANE")
        if parameters.size < 3:
            raise ValueError("missing plane parameters")
        normal = _normalize(parameters[:3])
        if normal is None:
            raise ValueError("invalid side normal")
        normal_by_patch[patch_id] = normal
        points_by_patch[patch_id] = np.asarray(result["mesh"].vertices, dtype=float)
    normals = [normal_by_patch[patch_id] for patch_id in side_ids]
    axis, normal_error = _extrusion_axis_from_normals(normals)
    if axis is None or normal_error > np.sin(np.deg2rad(12.0)):
        raise ValueError("side normals do not define one extrusion axis")
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    origin = np.mean(np.vstack(list(points_by_patch.values())), axis=0)
    origin = origin - axis * np.dot(origin, axis)
    first, second = _orthogonal_basis(axis)

    # Drop side patches whose vertices do not lie on the profile boundary.
    # A planar face inside the profile hull (for example the interior face a
    # slot cuts into) is not a wall of this prism and would otherwise
    # stretch the hull over unrelated geometry.
    original_side_count = len(side_ids)
    while True:
        all_points = np.vstack([points_by_patch[patch_id] for patch_id in side_ids])
        local = np.column_stack(((all_points - origin) @ first, (all_points - origin) @ second))
        if len(local) < 3:
            raise ValueError("insufficient extrusion points")
        hull = scipy.spatial.ConvexHull(local)
        polygon = local[hull.vertices]
        raw_polygon_vertex_count = len(polygon)
        profile_scale = max(float(np.linalg.norm(np.ptp(polygon, axis=0))), EPSILON)
        polygon = _simplify_polygon(polygon, 1e-4 * profile_scale)
        boundary_tolerance = max(1e-5, 0.02 * profile_scale)
        kept = []
        for patch_id in side_ids:
            patch_points = points_by_patch[patch_id]
            patch_local = np.column_stack(
                ((patch_points - origin) @ first, (patch_points - origin) @ second)
            )
            distance = _polygon_boundary_distance(patch_local, polygon)
            if np.mean(distance <= boundary_tolerance) >= 0.5:
                kept.append(patch_id)
        if len(kept) == len(side_ids):
            break
        if len(kept) < 3:
            raise ValueError("side patches do not form the profile boundary")
        side_ids = kept
    if len(side_ids) * 2 < original_side_count:
        # Only the patches touching the silhouette of a much larger point set
        # survived (for example the top/bottom rings of a screw projected
        # into one profile plane).  That silhouette is not a polygonal prism.
        raise ValueError("side patches only form the silhouette of a larger shape")
    normals = [normal_by_patch[patch_id] for patch_id in side_ids]
    normal_error = float(np.mean(np.abs(np.asarray(normals) @ axis)))
    first_turn = np.cross(
        np.r_[polygon[1] - polygon[0], 0.0],
        np.r_[polygon[2] - polygon[1], 0.0],
    )[2]
    if abs(first_turn) < EPSILON:
        raise ValueError("degenerate extrusion polygon")
    axial = (all_points - origin) @ axis
    extent = np.array([axial.min(), axial.max()], dtype=float)
    if extent[1] - extent[0] <= EPSILON:
        raise ValueError("degenerate extrusion height")

    associated = list(side_ids)
    cap_cosine = np.cos(np.deg2rad(12.0))
    for patch_id in plane_ids:
        if patch_id in associated:
            continue
        result = result_lookup[patch_id]
        _, parameters = _seed_fit(result, "PLANE")
        if parameters.size < 3:
            continue
        normal = _normalize(parameters[:3])
        if normal is None or abs(np.dot(normal, axis)) < cap_cosine:
            continue
        vertices = np.asarray(result["mesh"].vertices, dtype=float)
        projected_axial = (vertices - origin) @ axis
        near_end = np.min(np.abs(projected_axial[:, None] - extent[None, :]), axis=1)
        local_vertices = np.column_stack(
            ((vertices - origin) @ first, (vertices - origin) @ second)
        )
        inside = _point_in_polygon(local_vertices, polygon)
        for vertex_index, vertex in enumerate(local_vertices):
            if inside[vertex_index]:
                continue
            for first_vertex, second_vertex in zip(
                polygon, np.roll(polygon, -1, axis=0)
            ):
                edge = second_vertex - first_vertex
                edge_length = max(float(np.dot(edge, edge)), EPSILON)
                projection = np.clip(
                    np.dot(vertex - first_vertex, edge) / edge_length, 0.0, 1.0
                )
                closest = first_vertex + projection * edge
                if np.linalg.norm(vertex - closest) <= 1e-5:
                    inside[vertex_index] = True
                    break
        if np.max(near_end) <= max(1e-5, 0.03 * (extent[1] - extent[0])) and np.all(inside):
            associated.append(patch_id)

    return {
        "axis_point": origin,
        "axis": axis,
        "extent": extent,
        "basis": np.column_stack([first, second]),
        "polygon": polygon,
        "raw_polygon_vertex_count": raw_polygon_vertex_count,
        "side_patch_ids": side_ids,
        "cap_patch_ids": [patch_id for patch_id in associated if patch_id not in side_ids],
        "normal_error": normal_error,
    }


def _cap_pair_extrusion_candidates(results, graph, plane_ids, maximum_error=0.05):
    """Recover prisms seen mostly from their two congruent end caps.

    Grooves and slots often expose only the two end faces and a single side
    face, so the side-driven path (which needs at least three side patches)
    can never fire.  Two parallel, congruent, non-adjacent plane patches
    connected by at least one plane side patch define the same prism from
    the caps: the cap outline is the profile, the side patches are walls.
    """
    result_lookup = _result_lookup(results)
    neighbors = _aggregated_graph(graph)
    all_vertices = np.vstack(
        [np.asarray(result["mesh"].vertices, dtype=float) for result in results]
    )
    model_scale = max(float(np.linalg.norm(np.ptp(all_vertices, axis=0))), EPSILON)
    plane_set = set(plane_ids)
    normals = {}
    for patch_id in plane_ids:
        _, parameters = _seed_fit(result_lookup[patch_id], "PLANE")
        normal = _normalize(parameters[:3])
        if normal is not None:
            normals[patch_id] = normal
    cosine = np.cos(np.deg2rad(20.0))
    side_limit = np.sin(np.deg2rad(15.0))
    candidates = []
    for first, second in itertools.combinations(sorted(plane_set), 2):
        if second in neighbors.get(first, {}):
            continue
        normal_first = normals.get(first)
        normal_second = normals.get(second)
        if normal_first is None or normal_second is None:
            continue
        alignment = float(np.dot(normal_first, normal_second))
        if abs(alignment) < cosine:
            continue
        area_first = float(result_lookup[first].get("area", 0.0))
        area_second = float(result_lookup[second].get("area", 0.0))
        if min(area_first, area_second) <= 0.5 * max(area_first, area_second):
            continue
        # Average the two (sign-aligned) normals so opposite draft angles
        # cancel instead of tilting the prism axis.
        axis = _normalize(normal_first + np.sign(alignment) * normal_second)
        if axis is None:
            continue
        vertices_first = np.asarray(result_lookup[first]["mesh"].vertices, dtype=float)
        vertices_second = np.asarray(result_lookup[second]["mesh"].vertices, dtype=float)
        if (
            abs(float((vertices_first @ axis).mean() - (vertices_second @ axis).mean()))
            < 1e-3 * model_scale
        ):
            continue
        shared = (
            set(neighbors.get(first, {}))
            & set(neighbors.get(second, {}))
            & plane_set
        )
        side_ids = sorted(
            patch_id
            for patch_id in shared
            if abs(float(np.dot(normals[patch_id], axis))) < side_limit
        )
        if not side_ids:
            continue
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axis = -axis
        first_basis, second_basis = _orthogonal_basis(axis)
        basis = np.column_stack([first_basis, second_basis])
        origin = np.mean(np.vstack([vertices_first, vertices_second]), axis=0)
        origin = origin - axis * np.dot(origin, axis)
        try:
            local_first = (vertices_first - origin) @ basis
            hull = scipy.spatial.ConvexHull(local_first)
        except scipy.spatial.QhullError:
            continue
        polygon = local_first[hull.vertices]
        raw_polygon_vertex_count = len(polygon)
        profile_scale = max(float(np.linalg.norm(np.ptp(polygon, axis=0))), EPSILON)
        polygon = _simplify_polygon(polygon, 1e-4 * profile_scale)
        if len(polygon) < 3:
            continue
        # Side patches must lie on the profile boundary, just like in the
        # side-driven path; interior patches (for example slot walls inside
        # a cube face outline) only stretch the prism over unrelated faces.
        boundary_tolerance = max(1e-5, 0.02 * profile_scale)
        side_ids = [
            patch_id
            for patch_id in side_ids
            if np.mean(
                _polygon_boundary_distance(
                    (
                        np.asarray(
                            result_lookup[patch_id]["mesh"].vertices, dtype=float
                        )
                        - origin
                    )
                    @ basis,
                    polygon,
                )
                <= boundary_tolerance
            )
            >= 0.5
        ]
        if not side_ids:
            continue
        local_second = (vertices_second - origin) @ basis
        near = _polygon_boundary_distance(local_second, polygon) <= max(
            1e-5, 0.02 * profile_scale
        )
        overlap = _point_in_polygon(local_second, polygon) | near
        if np.mean(overlap) < 0.5:
            continue
        axial = np.concatenate(
            [(vertices_first - origin) @ axis, (vertices_second - origin) @ axis]
        )
        extent = np.array([axial.min(), axial.max()], dtype=float)
        if extent[1] - extent[0] <= EPSILON:
            continue
        normal_error = float(
            np.mean(np.abs(np.asarray([normals[patch_id] for patch_id in side_ids]) @ axis))
        )
        if normal_error > maximum_error:
            continue
        parameters = {
            "axis_point": origin,
            "axis": axis,
            "extent": extent,
            "basis": basis,
            "polygon": polygon,
            "raw_polygon_vertex_count": raw_polygon_vertex_count,
            "side_patch_ids": side_ids,
            "cap_patch_ids": [first, second],
            "normal_error": normal_error,
        }
        patch_ids = sorted(set(side_ids) | {first, second})
        rectangular = raw_polygon_vertex_count == 4
        candidates.append(
            PrimitiveCandidate(
                primitive_type="EXTRUSION",
                parameters=parameters,
                patch_ids=patch_ids,
                fitting_error=normal_error,
                confidence=float(np.exp(-8.0 * normal_error) * (0.85 if rectangular else 1.0)),
                metadata={
                    "side_patch_ids": side_ids,
                    "rectangular": rectangular,
                    "raw_polygon_vertex_count": raw_polygon_vertex_count,
                    "cap_driven": True,
                },
            )
        )
    return candidates


def generate_extrusion_candidates(results, graph, min_plane_fit_rate=0.8, maximum_error=0.05):
    """Generate polygonal-prism candidates from adjacent planar patches."""
    result_lookup = _result_lookup(results)
    plane_ids = []
    for patch_id, result in enumerate(results):
        rate, parameters = _seed_fit(result, "PLANE")
        if rate >= min_plane_fit_rate and parameters.size >= 4:
            plane_ids.append(int(result.get("patch_index", patch_id)))
    if len(plane_ids) < 3:
        return []
    total_plane_area = sum(
        float(result_lookup[patch_id].get("area", 0.0)) for patch_id in plane_ids
    )
    neighbors = _aggregated_graph(graph)

    def connected_groups(patch_ids):
        groups = []
        remaining = set(patch_ids)
        while remaining:
            seed = min(remaining)
            group = set()
            stack = [seed]
            while stack:
                current = stack.pop()
                if current in group or current not in remaining:
                    continue
                group.add(current)
                stack.extend(
                    adjacent
                    for adjacent in neighbors.get(current, {})
                    if adjacent in remaining
                )
            remaining.difference_update(group)
            groups.append(sorted(group))
        return groups

    side_limit = np.sin(np.deg2rad(15.0))
    candidates = []
    for component in connected_groups(plane_ids):
        if len(component) < 3:
            continue
        normals = {}
        for patch_id in component:
            _, parameters = _seed_fit(result_lookup[patch_id], "PLANE")
            normal = _normalize(parameters[:3])
            if normal is not None:
                normals[patch_id] = normal
        # A plane component can host several prisms with different axes
        # (their faces stay connected through the surrounding body), so every
        # supported axis hypothesis gets its own set of side bands.
        for _, axis, _ in _extrusion_axis_hypotheses(list(normals.values())):
            side_ids = [
                patch_id
                for patch_id, normal in normals.items()
                if abs(np.dot(normal, axis)) < side_limit
            ]
            for side_group in connected_groups(side_ids):
                if len(side_group) < 3:
                    continue
                side_areas = np.asarray(
                    [result_lookup[patch_id].get("area", 0.0) for patch_id in side_group]
                )
                if np.sum(side_areas) <= EPSILON:
                    continue
                if np.max(side_areas) / np.sum(side_areas) > 0.95:
                    continue
                try:
                    parameters = _fit_extrusion_group(results, graph, side_group, plane_ids)
                except (ValueError, np.linalg.LinAlgError, scipy.spatial.QhullError):
                    continue
                patch_ids = sorted(
                    set(parameters["side_patch_ids"]) | set(parameters["cap_patch_ids"])
                )
                polygon = parameters["polygon"]
                rectangular = parameters["raw_polygon_vertex_count"] == 4
                # A high-complexity non-rectangular hull is usually an assembly of
                # rotated/cut blocks, not one polygonal prism.  Keep ordinary
                # extrusions and rectangular profiles, but avoid swallowing an entire
                # multi-feature model as a single candidate.
                if (
                    len(parameters["side_patch_ids"]) >= 10
                    and not rectangular
                    and parameters["raw_polygon_vertex_count"] > 8
                ):
                    continue
                associated_area = sum(
                    float(result_lookup[patch_id].get("area", 0.0)) for patch_id in patch_ids
                )
                if (
                    rectangular
                    and total_plane_area > EPSILON
                    and associated_area / total_plane_area > 0.8
                ):
                    continue
                fitting_error = float(parameters["normal_error"])
                # A large axis residual means the "sides" are not a coherent
                # band of prism walls (for example a flat top face mixed with
                # chamfer facets around it).  Curved candidates already apply
                # the same kind of gate.
                if fitting_error > maximum_error:
                    continue
                confidence = float(np.exp(-8.0 * fitting_error) * (0.85 if rectangular else 1.0))
                candidates.append(
                    PrimitiveCandidate(
                        primitive_type="EXTRUSION",
                        parameters=parameters,
                        patch_ids=patch_ids,
                        fitting_error=fitting_error,
                        confidence=confidence,
                        metadata={
                            "side_patch_ids": parameters["side_patch_ids"],
                            "rectangular": rectangular,
                            "raw_polygon_vertex_count": parameters["raw_polygon_vertex_count"],
                        },
                    )
                )
    unique = {}
    for candidate in candidates + _cap_pair_extrusion_candidates(
        results, graph, plane_ids, maximum_error=maximum_error
    ):
        key = tuple(candidate.patch_ids)
        previous = unique.get(key)
        if previous is None or candidate.fitting_error < previous.fitting_error:
            unique[key] = candidate
    return list(unique.values())


def _orthogonal_basis(axis):
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, reference)) > 0.85:
        reference = np.array([0.0, 1.0, 0.0])
    first = _normalize(reference - np.dot(reference, axis) * axis)
    second = _normalize(np.cross(axis, first))
    return first, second


def _fit_circle_2d(points):
    matrix = np.column_stack(
        [2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))]
    )
    target = np.sum(points**2, axis=1)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    center = solution[:2]
    radius_squared = solution[2] + np.dot(center, center)
    if radius_squared <= EPSILON:
        raise ValueError("degenerate circle")
    return center, float(np.sqrt(radius_squared))


def fit_cylinder(points, normals=None):
    """Fit an open or partial cylinder and return parameters plus RMS error."""
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    if normals is not None and len(normals) == len(points):
        normal_matrix = np.asarray(normals).T @ np.asarray(normals)
        _, vectors = np.linalg.eigh(normal_matrix)
        axis = vectors[:, 0]
    else:
        _, vectors = np.linalg.eigh(np.cov(points - center, rowvar=False))
        axis = vectors[:, -1]
    axis = _normalize(axis)
    first, second = _orthogonal_basis(axis)
    basis = np.column_stack([first, second])
    projected = (points - center) @ basis
    circle_center, radius = _fit_circle_2d(projected)
    axis_point = center + basis @ circle_center

    def residual(parameters):
        point = parameters[:3]
        direction = _normalize(parameters[3:6])
        if direction is None:
            return np.full(len(points), 1e3)
        radial = np.linalg.norm(np.cross(points - point, direction), axis=1)
        return radial - parameters[6]

    initial = np.concatenate([axis_point, axis, [radius]])
    lower = np.concatenate([np.full(6, -np.inf), [EPSILON]])
    upper = np.full(7, np.inf)
    optimized = scipy.optimize.least_squares(
        residual, initial, bounds=(lower, upper), max_nfev=400
    )
    axis_point = optimized.x[:3]
    axis = _normalize(optimized.x[3:6])
    radius = float(optimized.x[6])
    axis_point = axis_point + axis * np.dot(center - axis_point, axis)
    axial = (points - axis_point) @ axis
    scale = max(np.linalg.norm(np.ptp(points, axis=0)), EPSILON)
    error = float(np.sqrt(np.mean(residual(optimized.x) ** 2)) / scale)
    return {
        "axis_point": axis_point,
        "axis": axis,
        "radius": radius,
        "extent": np.array([axial.min(), axial.max()]),
    }, error


def fit_sphere(points, normals=None):
    """Fit a sphere and return its center, radius, and normalized RMS error."""
    points = np.asarray(points, dtype=float)
    matrix = np.column_stack([2.0 * points, np.ones(len(points))])
    target = np.sum(points**2, axis=1)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    center = solution[:3]
    radius_squared = solution[3] + np.dot(center, center)
    if radius_squared <= EPSILON:
        raise ValueError("degenerate sphere")
    radius = np.sqrt(radius_squared)

    def residual(parameters):
        return np.linalg.norm(points - parameters[:3], axis=1) - parameters[3]

    optimized = scipy.optimize.least_squares(
        residual,
        np.concatenate([center, [radius]]),
        bounds=(np.array([-np.inf, -np.inf, -np.inf, EPSILON]), np.full(4, np.inf)),
        max_nfev=300,
    )
    center = optimized.x[:3]
    radius = float(optimized.x[3])
    scale = max(np.linalg.norm(np.ptp(points, axis=0)), EPSILON)
    error = float(np.sqrt(np.mean(residual(optimized.x) ** 2)) / scale)
    return {"center": center, "radius": radius}, error


def fit_cone(points, normals=None):
    """Fit a finite cone and return apex, axis, angle, extent, and error."""
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    if normals is not None and len(normals) == len(points):
        normals = np.asarray(normals, dtype=float)
        target = np.sum(normals * points, axis=1)
        apex, _, _, _ = np.linalg.lstsq(normals, target, rcond=None)
    else:
        apex = center.copy()
    covariance = np.cov(points - center, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    candidate_axes = [vectors[:, index] for index in range(3)]
    best = None
    scale = max(np.linalg.norm(np.ptp(points, axis=0)), EPSILON)
    for initial_axis in candidate_axes:
        for direction_sign in (-1.0, 1.0):
            axis = initial_axis * direction_sign
            axial = (points - apex) @ axis
            if np.median(axial) < 0:
                axis = -axis
                axial = -axial
            radial = np.linalg.norm((points - apex) - axial[:, None] * axis, axis=1)
            positive = axial > scale * 1e-6
            if np.count_nonzero(positive) < 3:
                continue
            angle = np.arctan(np.median(radial[positive] / axial[positive]))
            angle = float(np.clip(angle, np.deg2rad(1.0), np.deg2rad(80.0)))

            def residual(parameters):
                current_apex = parameters[:3]
                current_axis = _normalize(parameters[3:6])
                if current_axis is None:
                    return np.full(len(points), 1e3)
                vectors_to_points = points - current_apex
                current_axial = vectors_to_points @ current_axis
                current_radial = np.linalg.norm(
                    vectors_to_points - current_axial[:, None] * current_axis, axis=1
                )
                geometric = current_radial - current_axial * np.tan(parameters[6])
                negative_penalty = np.minimum(current_axial, 0.0)
                return geometric + negative_penalty

            initial = np.concatenate([apex, axis, [angle]])
            lower = np.concatenate([np.full(6, -np.inf), [np.deg2rad(0.5)]])
            upper = np.concatenate([np.full(6, np.inf), [np.deg2rad(85.0)]])
            optimized = scipy.optimize.least_squares(
                residual, initial, bounds=(lower, upper), max_nfev=500
            )
            error = float(np.sqrt(np.mean(residual(optimized.x) ** 2)) / scale)
            if best is None or error < best[0]:
                best = (error, optimized.x)
    if best is None:
        raise ValueError("cone fitting failed")
    error, parameters = best
    apex = parameters[:3]
    axis = _normalize(parameters[3:6])
    angle = float(parameters[6])
    axial = (points - apex) @ axis
    return {
        "apex": apex,
        "axis": axis,
        "angle": angle,
        "extent": np.array([max(0.0, axial.min()), axial.max()]),
    }, error


def fit_torus(points, normals=None):
    """Fit a torus and return its frame, two radii, and normalized error."""
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    candidate_axes = [vectors[:, index] for index in range(3)]
    if normals is not None and len(normals) == len(points):
        normal_covariance = np.asarray(normals).T @ np.asarray(normals)
        _, normal_vectors = np.linalg.eigh(normal_covariance)
        candidate_axes.extend(normal_vectors[:, index] for index in range(3))
    scale = max(np.linalg.norm(np.ptp(points, axis=0)), EPSILON)
    best = None
    for initial_axis in candidate_axes:
        axis = _normalize(initial_axis)
        relative = points - center
        axial = relative @ axis
        radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
        major_radius = max(float(np.median(radial)), scale * 1e-3)
        minor_radius = max(
            float(np.median(np.sqrt((radial - major_radius) ** 2 + axial**2))),
            scale * 1e-3,
        )

        def residual(parameters):
            current_center = parameters[:3]
            current_axis = _normalize(parameters[3:6])
            if current_axis is None:
                return np.full(len(points), 1e3)
            relative_points = points - current_center
            current_axial = relative_points @ current_axis
            current_radial = np.linalg.norm(
                relative_points - current_axial[:, None] * current_axis, axis=1
            )
            tube_distance = np.sqrt(
                (current_radial - parameters[6]) ** 2 + current_axial**2
            )
            return tube_distance - parameters[7]

        initial = np.concatenate([center, axis, [major_radius, minor_radius]])
        lower = np.concatenate([np.full(6, -np.inf), [EPSILON, EPSILON]])
        upper = np.full(8, np.inf)
        optimized = scipy.optimize.least_squares(
            residual, initial, bounds=(lower, upper), max_nfev=600
        )
        error = float(np.sqrt(np.mean(residual(optimized.x) ** 2)) / scale)
        if best is None or error < best[0]:
            best = (error, optimized.x)
    if best is None:
        raise ValueError("torus fitting failed")
    error, parameters = best
    return {
        "center": parameters[:3],
        "axis": _normalize(parameters[3:6]),
        "major_radius": float(parameters[6]),
        "minor_radius": float(parameters[7]),
    }, error


def _curved_pair_compatible(first, second, primitive_type, model_scale):
    if primitive_type == "TORUS":
        return True
    first_rate, first_parameters = _seed_fit(first, primitive_type)
    second_rate, second_parameters = _seed_fit(second, primitive_type)
    if first_rate <= 0.0 or second_rate <= 0.0:
        return False
    if (
        primitive_type == "SPHERE"
        and first_parameters.size >= 4
        and second_parameters.size >= 4
    ):
        radius = max(abs(first_parameters[3]), abs(second_parameters[3]), EPSILON)
        return bool(
            np.linalg.norm(first_parameters[:3] - second_parameters[:3]) <= 0.15 * radius
            and abs(first_parameters[3] - second_parameters[3]) <= 0.15 * radius
        )
    if (
        primitive_type == "CYLINDER"
        and first_parameters.size >= 7
        and second_parameters.size >= 7
    ):
        first_axis = _normalize(first_parameters[3:6])
        second_axis = _normalize(second_parameters[3:6])
        first_radius = abs(float(first_parameters[6]))
        second_radius = abs(float(second_parameters[6]))
        radius = max(first_radius, second_radius, EPSILON)
        if first_axis is None or second_axis is None:
            return False
        # A short, smaller-diameter segment must not bridge two equal outer
        # segments through the transitive grouping step below.
        radius_difference = abs(first_radius - second_radius) / radius
        return bool(
            abs(np.dot(first_axis, second_axis)) >= np.cos(np.deg2rad(12.0))
            and radius_difference <= 0.04
            and np.linalg.norm(
                np.cross(second_parameters[:3] - first_parameters[:3], first_axis)
            )
            <= max(0.20 * radius, model_scale * 1e-4)
        )
    if (
        primitive_type == "CONE"
        and first_parameters.size >= 7
        and second_parameters.size >= 7
    ):
        first_axis = _normalize(first_parameters[3:6])
        second_axis = _normalize(second_parameters[3:6])
        if first_axis is None or second_axis is None:
            return False
        return bool(
            abs(np.dot(first_axis, second_axis)) >= np.cos(np.deg2rad(15.0))
            and abs(first_parameters[6] - second_parameters[6]) <= np.deg2rad(8.0)
            and np.linalg.norm(first_parameters[:3] - second_parameters[:3]) <= model_scale * 0.1
        )
    return True


def _curved_patch_groups(results, graph, primitive_type, min_seed_rate, model_scale):
    result_lookup = _result_lookup(results)
    eligible = set()
    for patch_id, result in enumerate(results):
        rate, _ = _seed_fit(result, primitive_type)
        if rate >= min_seed_rate or str(result.get("type", "")).upper() == primitive_type:
            eligible.add(int(result.get("patch_index", patch_id)))
    if primitive_type in {"CYLINDER", "SPHERE"}:
        eligible = {
            patch_id
            for patch_id in eligible
            if str(result_lookup[patch_id].get("type", "")).upper() == primitive_type
        }
    if primitive_type == "CONE":
        eligible = {
            patch_id
            for patch_id in eligible
            if str(result_lookup[patch_id].get("type", "")).upper() == "CONE"
        }
    if primitive_type == "TORUS":
        eligible = {
            patch_id
            for patch_id, result in result_lookup.items()
            if str(result.get("type", "")).upper() == "SPHERE"
            and _seed_fit(result, "PLANE")[0] < 0.95
            and len(result["mesh"].vertices) >= 16
        }

    if primitive_type in {"CYLINDER", "SPHERE"}:
        parent = {patch_id: patch_id for patch_id in eligible}

        def find(patch_id):
            while parent[patch_id] != patch_id:
                parent[patch_id] = parent[parent[patch_id]]
                patch_id = parent[patch_id]
            return patch_id

        def union(first, second):
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for first, second in itertools.combinations(sorted(eligible), 2):
            compatible = _curved_pair_compatible(
                result_lookup[first], result_lookup[second], primitive_type, model_scale
            )
            if compatible and primitive_type == "CYLINDER":
                _, first_parameters = _seed_fit(result_lookup[first], "CYLINDER")
                _, second_parameters = _seed_fit(result_lookup[second], "CYLINDER")
                first_axis = _normalize(first_parameters[3:6])
                second_axis = _normalize(second_parameters[3:6])
                radius = max(abs(first_parameters[6]), abs(second_parameters[6]), EPSILON)
                if first_axis is None or second_axis is None:
                    compatible = False
                else:
                    # The seed axes are compared sign-independently elsewhere,
                    # but the axial projections below are signed.  Align the
                    # directions first, otherwise two distant segments with
                    # opposite axis signs falsely report a zero interval gap.
                    if float(np.dot(first_axis, second_axis)) < 0.0:
                        second_axis = -second_axis
                    radius_match = (
                        abs(first_parameters[6] - second_parameters[6])
                        <= 0.04 * radius
                    )
                    projections = []
                    for patch_id, axis_point, axis in (
                        (first, first_parameters[:3], first_axis),
                        (second, second_parameters[:3], second_axis),
                    ):
                        vertices = np.asarray(
                            result_lookup[patch_id]["mesh"].vertices, dtype=float
                        )
                        projections.append((vertices - axis_point) @ axis)
                    interval_gap = max(
                        projections[0].min() - projections[1].max(),
                        projections[1].min() - projections[0].max(),
                        0.0,
                    )
                    compatible = bool(
                        radius_match
                        and interval_gap
                        <= max(0.005 * model_scale, 0.10 * radius)
                    )
            if compatible:
                union(first, second)

        components = {}
        for patch_id in sorted(eligible):
            components.setdefault(find(patch_id), []).append(patch_id)
        return list(components.values())

    neighbors = _aggregated_graph(graph)
    if primitive_type == "TORUS":
        edge_groups = []
        for first in sorted(eligible):
            for second in neighbors.get(first, {}):
                if second in eligible and first < second:
                    edge_groups.append([first, second])
        components = []
        remaining = set(eligible)
        while remaining:
            seed = min(remaining)
            component = set()
            stack = [seed]
            while stack:
                current = stack.pop()
                if current in component or current not in remaining:
                    continue
                component.add(current)
                stack.extend(
                    adjacent
                    for adjacent in neighbors.get(current, {})
                    if adjacent in remaining
                )
            remaining.difference_update(component)
            components.append(sorted(component))
        return edge_groups + components

    groups = []
    remaining = set(eligible)
    while remaining:
        seed = min(remaining)
        group = set()
        stack = [seed]
        while stack:
            current = stack.pop()
            if current in group or current not in remaining:
                continue
            group.add(current)
            for adjacent in neighbors.get(current, {}):
                if adjacent in remaining and _curved_pair_compatible(
                    result_lookup[current],
                    result_lookup[adjacent],
                    primitive_type,
                    model_scale,
                ):
                    stack.append(adjacent)
        remaining.difference_update(group)
        groups.append(sorted(group))
    return groups


def _valid_sphere_fit(parameters, model_bounds, model_scale):
    """Reject the large-radius sphere produced by an almost planar patch."""
    center = np.asarray(parameters.get("center", []), dtype=float).reshape(-1)
    radius = float(parameters.get("radius", np.nan))
    if center.size != 3 or not np.all(np.isfinite(center)):
        return False
    if not np.isfinite(radius) or radius <= EPSILON:
        return False

    # A sphere used to explain this model must have a scale comparable to the
    # model.  Least-squares fitting of a plane otherwise converges to a sphere
    # with a huge radius and a nearly zero normalized residual.
    if radius > model_scale:
        return False

    # A center many model diagonals away is another signature of an almost
    # planar patch being approximated by a numerically huge sphere.
    model_center = model_bounds.mean(axis=0)
    if np.linalg.norm(center - model_center) > 2.0 * model_scale:
        return False

    sphere_bounds = np.stack([center - radius, center + radius])
    tolerance = model_scale * 1e-6
    overlap = np.minimum(sphere_bounds[1], model_bounds[1]) - np.maximum(
        sphere_bounds[0], model_bounds[0]
    )
    return bool(np.all(overlap >= -tolerance))


def _sphere_surface_quality(points, parameters):
    """Measure whether fitted radial directions span a spherical surface.

    Chamfers and fillets often obtain an excellent sphere residual despite
    being only a cylindrical strip.  Their normalized radial directions have
    one nearly-zero covariance eigenvalue, unlike a genuine spherical patch.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(parameters.get("center", []), dtype=float).reshape(-1)
    radius = float(parameters.get("radius", np.nan))
    if center.size != 3 or not np.isfinite(radius) or radius <= EPSILON:
        return 0.0
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > EPSILON
    if np.count_nonzero(valid) < 6:
        return 0.0
    vectors = vectors[valid] / lengths[valid, None]
    covariance = np.cov(vectors, rowvar=False)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return 0.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(max(0.0, eigenvalues[0]) / max(eigenvalues[-1], EPSILON))


def _attach_planar_caps(candidate, results, plane_ids, neighbors, min_plane_fit_rate=0.8):
    """Collect flat end-cap patch requests for a cylinder/cone candidate.

    Curved candidates otherwise ignore the flat annulus or disk closing one
    end of the primitive (for example the bottom face of a nut), leaving
    those patches uncovered although the primitive's extent ends exactly
    there.  Only patches whose own best type is ``plane`` and which share a
    graph edge with the candidate's patches are considered, so thin thread
    crest rings (typed cylinder) and unrelated coplanar faces are never
    stolen as caps.  Conflicting requests for the same patch are dropped by
    the caller.
    """
    if candidate.primitive_type not in {"CYLINDER", "CONE"}:
        return []
    parameters = candidate.parameters
    axis = _normalize(parameters.get("axis"))
    if axis is None:
        return []
    origin = np.asarray(
        parameters.get("axis_point", parameters.get("apex")), dtype=float
    )
    extent = np.asarray(parameters["extent"], dtype=float)
    height = max(float(extent[1] - extent[0]), EPSILON)
    result_lookup = _result_lookup(results)
    cap_cosine = np.cos(np.deg2rad(12.0))
    claimed = set(candidate.patch_ids)
    adjacent = {
        neighbor
        for patch_id in claimed
        for neighbor in neighbors.get(patch_id, {})
    } - claimed
    caps = []
    for patch_id in plane_ids:
        if patch_id in claimed or patch_id not in adjacent:
            continue
        result = result_lookup[patch_id]
        if str(result.get("type", "")).lower() != "plane":
            continue
        _, plane_parameters = _seed_fit(result, "PLANE")
        if plane_parameters.size < 3:
            continue
        normal = _normalize(plane_parameters[:3])
        if normal is None or abs(np.dot(normal, axis)) < cap_cosine:
            continue
        vertices = np.asarray(result["mesh"].vertices, dtype=float)
        relative = vertices - origin
        axial = relative @ axis
        near_end = np.min(np.abs(axial[:, None] - extent[None, :]), axis=1)
        if np.max(near_end) > max(1e-5, 0.03 * height):
            continue
        radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
        if candidate.primitive_type == "CYLINDER":
            expected = np.full(len(vertices), abs(float(parameters["radius"])))
        else:
            expected = np.maximum(0.0, axial * np.tan(parameters["angle"]))
        margin = max(1e-5, 0.02 * max(float(np.max(expected)), EPSILON))
        if np.max(radial - expected) > margin:
            continue
        caps.append(patch_id)
    return caps


def generate_curved_candidates(
    results,
    graph,
    primitive_types=("CYLINDER", "CONE", "SPHERE", "TORUS"),
    min_seed_rate=0.55,
    maximum_error=0.04,
):
    """Fit curved primitive candidates to compatible connected patch groups."""
    result_lookup = _result_lookup(results)
    all_vertices = np.vstack([np.asarray(result["mesh"].vertices) for result in results])
    model_bounds = np.stack([all_vertices.min(axis=0), all_vertices.max(axis=0)])
    model_scale = max(np.linalg.norm(np.ptp(all_vertices, axis=0)), EPSILON)
    plane_ids = [
        patch_id
        for patch_id, result in result_lookup.items()
        if _seed_fit(result, "PLANE")[0] >= 0.8
    ]
    fitters = {
        "CYLINDER": fit_cylinder,
        "CONE": fit_cone,
        "SPHERE": fit_sphere,
        "TORUS": fit_torus,
    }
    candidates = []
    for primitive_type in primitive_types:
        groups = _curved_patch_groups(
            results, graph, primitive_type, min_seed_rate, model_scale
        )
        for patch_ids in groups:
            points, normals = _sample_patch_points(results, patch_ids)
            try:
                parameters, error = fitters[primitive_type](points, normals)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
            if not np.isfinite(error) or error > maximum_error:
                continue
            if primitive_type == "SPHERE" and not _valid_sphere_fit(
                parameters, model_bounds, model_scale
            ):
                continue
            if primitive_type == "SPHERE":
                # Do not promote a narrow fillet/chamfer strip to a standalone
                # sphere.  Keep complete and sufficiently broad partial
                # spheres, including disconnected patches of one sphere.
                if _sphere_surface_quality(points, parameters) < 0.03:
                    # A shallow but genuine spherical cap (for example a domed
                    # nut top) has the same low directional spread as a fillet
                    # strip.  The strip is explained better by a valid torus;
                    # the cap is not (its torus fit degenerates to a
                    # near-zero major radius), so it stays a sphere.
                    try:
                        torus_parameters, torus_error = fit_torus(points, normals)
                    except (ValueError, np.linalg.LinAlgError, RuntimeError):
                        torus_parameters, torus_error = None, np.inf
                    valid_torus = (
                        torus_parameters is not None
                        and np.isfinite(torus_error)
                        and float(torus_parameters.get("major_radius", 0.0))
                        > 1.05 * float(torus_parameters.get("minor_radius", 0.0))
                    )
                    if valid_torus and torus_error < error:
                        continue
                    # A rescued cap is only a small part of its sphere, so the
                    # primitive must be clipped to the body region it belongs
                    # to; otherwise the complete ball adds a large false
                    # volume.
                    parameters = dict(parameters)
                    parameters["clip_bounds"] = _sphere_cap_clip_bounds(
                        results, graph, patch_ids, parameters, model_scale
                    )
            if primitive_type == "TORUS":
                major_radius = float(parameters.get("major_radius", 0.0))
                minor_radius = float(parameters.get("minor_radius", 0.0))
                if (
                    not np.isfinite(major_radius)
                    or not np.isfinite(minor_radius)
                    or major_radius <= 1.05 * minor_radius
                ):
                    continue
            transition_patch_ids = [
                index
                for index in patch_ids
                if result_lookup[index].get("feature_type")
                in {"FILLET_OR_CHAMFER", "CHAMFER"}
            ]
            metadata = {}
            if transition_patch_ids and primitive_type in {"TORUS", "CONE"}:
                parameters = dict(parameters)
                parameters["clip_bounds"] = _patch_clip_bounds(
                    results, transition_patch_ids, model_scale
                )
                neighbors = _aggregated_graph(graph)
                support_patch_ids = sorted(
                    {
                        neighbor
                        for transition_patch_id in transition_patch_ids
                        for neighbor in neighbors.get(transition_patch_id, {})
                        if neighbor not in transition_patch_ids
                    }
                )
                metadata.update(
                    {
                        "transition": True,
                        "transition_type": "FILLET_OR_CHAMFER",
                        "support_patch_ids": support_patch_ids,
                        "transition_patch_ids": transition_patch_ids,
                        "clip_bounds": parameters["clip_bounds"].copy(),
                    }
                )
            seed_confidence = np.mean(
                [
                    max(_seed_fit(result_lookup[index], primitive_type)[0], 0.25)
                    for index in patch_ids
                ]
            )
            confidence = float(np.exp(-12.0 * error) * seed_confidence)
            candidates.append(
                PrimitiveCandidate(
                    primitive_type=primitive_type,
                    parameters=parameters,
                    patch_ids=patch_ids,
                    fitting_error=float(error),
                    confidence=confidence,
                    metadata=metadata,
                )
            )

    torus_candidates = [
        candidate
        for candidate in candidates
        if candidate.primitive_type == "TORUS" and len(candidate.patch_ids) >= 2
    ]
    if torus_candidates:
        filtered = []
        for candidate in candidates:
            if candidate.primitive_type != "SPHERE" or len(candidate.patch_ids) != 1:
                filtered.append(candidate)
                continue
            covered_by_torus = [
                torus
                for torus in torus_candidates
                if set(candidate.patch_ids).intersection(torus.patch_ids)
                and torus.fitting_error < 0.5 * candidate.fitting_error
            ]
            if not covered_by_torus:
                filtered.append(candidate)
        candidates = filtered

    neighbors = _aggregated_graph(graph)
    cap_requests = [
        (candidate, _attach_planar_caps(candidate, results, plane_ids, neighbors))
        for candidate in candidates
    ]
    request_count = {}
    for _, caps in cap_requests:
        for patch_id in caps:
            request_count[patch_id] = request_count.get(patch_id, 0) + 1
    for candidate, caps in cap_requests:
        # A flat face often closes several primitives at once (for example
        # the face a post stands on and a hole opens through); contested
        # patches stay unassigned instead of corrupting every candidate.
        accepted = [patch_id for patch_id in caps if request_count[patch_id] == 1]
        if accepted:
            candidate.patch_ids = sorted(set(candidate.patch_ids) | set(accepted))
            candidate.metadata["cap_patch_ids"] = accepted
    return candidates


def classify_transition_patches(results, graph):
    """Annotate patches that are likely chamfers or fillet transitions.

    Transition patches remain available as surface evidence, but are not
    emitted as independent primitives.  This keeps region growing from
    spreading through a feature merely because a partial sphere fit has a
    small residual.
    """
    result_lookup = _result_lookup(results)
    neighbors = _aggregated_graph(graph)
    transitions = {}
    for patch_id, result in result_lookup.items():
        best_type = str(result.get("type", "")).upper()
        adjacent_types = {
            str(result_lookup[neighbor].get("type", "")).upper()
            for neighbor in neighbors.get(patch_id, {})
            if neighbor in result_lookup
        }
        feature = "SURFACE"
        if best_type == "SPHERE":
            _, parameters = _seed_fit(result, "SPHERE")
            if parameters.size >= 4:
                points = np.asarray(result["mesh"].vertices, dtype=float)
                quality = _sphere_surface_quality(
                    points,
                    {"center": parameters[:3], "radius": parameters[3]},
                )
                if quality < 0.03:
                    feature = "FILLET_OR_CHAMFER"
        if feature == "SURFACE" and best_type == "PLANE":
            curved_neighbors = adjacent_types.intersection(
                {"CYLINDER", "CONE", "SPHERE", "TORUS"}
            )
            area = float(result.get("area", result["mesh"].area))
            neighbor_areas = [
                float(result_lookup[neighbor].get("area", result_lookup[neighbor]["mesh"].area))
                for neighbor in neighbors.get(patch_id, {})
                if neighbor in result_lookup
            ]
            reference_area = max(float(np.median(neighbor_areas)), EPSILON)
            if curved_neighbors and area <= 0.35 * reference_area:
                feature = "CHAMFER"
        result["feature_type"] = feature
        if feature != "SURFACE":
            transitions[patch_id] = {
                "feature_type": feature,
                "neighbor_patch_ids": sorted(neighbors.get(patch_id, {})),
                "neighbor_types": sorted(adjacent_types),
            }
    return transitions


def _patch_clip_bounds(results, patch_ids, model_scale):
    """Return a padded world-space box enclosing a partial surface group."""
    result_lookup = _result_lookup(results)
    points = np.vstack(
        [np.asarray(result_lookup[index]["mesh"].vertices, dtype=float) for index in patch_ids]
    )
    padding = max(float(model_scale) * 1e-4, 1e-8)
    return np.stack([points.min(axis=0) - padding, points.max(axis=0) + padding])


def _sphere_cap_clip_bounds(results, graph, patch_ids, sphere_parameters, model_scale):
    """Return a clip box for a rescued spherical cap.

    The clip only bounds the cap along its axis: laterally the ball already
    follows the dome silhouette, and an AABB's square cross-section would
    add false corner volume wherever the ball is wider than the patch box
    (for example where the dome merges into a round nut body).  Along the
    axis the cap solid continues into the body, so the bound is extended
    over the body-side neighbor patches (rims/walls below the cap) instead
    of cutting at the cap patch's edge.
    """
    result_lookup = _result_lookup(results)
    neighbors = _aggregated_graph(graph)
    center = np.asarray(sphere_parameters["center"], dtype=float)
    points = np.vstack(
        [np.asarray(result_lookup[patch_id]["mesh"].vertices, dtype=float) for patch_id in patch_ids]
    )
    padding = max(float(model_scale) * 1e-4, 1e-8)
    lower = points.min(axis=0) - padding
    upper = points.max(axis=0) + padding
    axis = _normalize(points.mean(axis=0) - center)
    if axis is None:
        return np.stack([lower, upper])
    dominant = int(np.argmax(np.abs(axis)))
    if abs(axis[dominant]) < 0.99:
        return np.stack([lower, upper])
    cap_sign = 1.0 if float((points.mean(axis=0) - center)[dominant]) >= 0.0 else -1.0
    cap_edge = (lower if cap_sign > 0.0 else upper)[dominant]
    body_station = None
    for patch_id in patch_ids:
        for adjacent in neighbors.get(patch_id, {}):
            if adjacent not in result_lookup:
                continue
            axial = (
                np.asarray(result_lookup[adjacent]["mesh"].vertices, dtype=float)[:, dominant]
                * cap_sign
            )
            # Body-side neighbors lie fully below the cap's lower edge;
            # recess walls piercing the cap span across it and are ignored.
            if axial.max() <= cap_edge * cap_sign + padding:
                station = float(axial.min()) * cap_sign
                body_station = (
                    station if body_station is None else min(body_station, station)
                )
    if body_station is None:
        return np.stack([lower, upper])
    # Laterally the ball is left unclipped (see docstring); along the axis it
    # now reaches the body's near side.
    model_min = points.min(axis=0) - float(model_scale)
    model_max = points.max(axis=0) + float(model_scale)
    lower = np.where(np.arange(3) == dominant, lower, model_min)
    upper = np.where(np.arange(3) == dominant, upper, model_max)
    if cap_sign > 0.0:
        lower[dominant] = min(lower[dominant], body_station - padding)
    else:
        upper[dominant] = max(upper[dominant], body_station + padding)
    return np.stack([lower, upper])


def generate_transition_curved_candidates(results, graph, maximum_error=0.04):
    """Fit local torus/cone mother surfaces for fillet transition patches.

    The fitted primitive is clipped to the observed patch bounds.  This is
    important: a complete torus or cone would add a large unrelated volume,
    while a clipped mother surface can participate in a CSG intersection with
    the neighboring base primitive.
    """
    transitions = classify_transition_patches(results, graph)
    if not transitions:
        return []
    all_vertices = np.vstack(
        [np.asarray(result["mesh"].vertices, dtype=float) for result in results]
    )
    model_scale = max(float(np.linalg.norm(np.ptp(all_vertices, axis=0))), EPSILON)
    result_lookup = _result_lookup(results)
    torus_groups = _curved_patch_groups(
        results, graph, "TORUS", 0.0, model_scale
    )
    protected_torus_patches = {
        patch_id
        for group in torus_groups
        if len(group) >= 2
        for patch_id in group
    }
    candidates = []
    for patch_id, transition in transitions.items():
        if transition["feature_type"] not in {"FILLET_OR_CHAMFER", "CHAMFER"}:
            continue
        if patch_id in protected_torus_patches:
            continue
        points, normals = _sample_patch_points(results, [patch_id])
        if transition["feature_type"] == "CHAMFER":
            patch_area = float(
                result_lookup[patch_id].get("area", result_lookup[patch_id]["mesh"].area)
            )
            if patch_area <= 1e-4 * model_scale**2:
                continue
        support_patch_ids = transition["neighbor_patch_ids"]
        clip_bounds = _patch_clip_bounds(results, [patch_id], model_scale)

        try:
            cone_parameters, cone_error = fit_cone(points, normals)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            cone_parameters, cone_error = None, np.inf
        if cone_parameters is not None:
            angle_degrees = float(np.rad2deg(cone_parameters["angle"]))
            axial_length = float(np.ptp(cone_parameters["extent"]))
            apex_distance = float(
                np.linalg.norm(np.mean(points, axis=0) - cone_parameters["apex"])
            )
            valid_cone = (
                np.isfinite(cone_error)
                and cone_error <= maximum_error
                and 3.0 <= angle_degrees <= 78.0
                and axial_length > model_scale * 1e-5
                and apex_distance <= 20.0 * model_scale
            )
            if valid_cone:
                cone_parameters = dict(cone_parameters)
                cone_parameters["clip_bounds"] = clip_bounds.copy()
                candidates.append(
                    PrimitiveCandidate(
                        primitive_type="CONE",
                        parameters=cone_parameters,
                        patch_ids=[patch_id],
                        fitting_error=float(cone_error),
                        confidence=float(0.25 * np.exp(-10.0 * cone_error)),
                        operation="ADD",
                        metadata={
                            "transition": True,
                            "transition_type": transition["feature_type"],
                            "support_patch_ids": list(support_patch_ids),
                            "clip_bounds": clip_bounds.copy(),
                        },
                    )
                )

        if transition["feature_type"] == "FILLET_OR_CHAMFER":
            try:
                torus_parameters, torus_error = fit_torus(points, normals)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                torus_parameters, torus_error = None, np.inf
        else:
            torus_parameters, torus_error = None, np.inf
        if torus_parameters is None:
            continue
        major_radius = float(torus_parameters.get("major_radius", 0.0))
        minor_radius = float(torus_parameters.get("minor_radius", 0.0))
        valid_torus = (
            np.isfinite(torus_error)
            and torus_error <= maximum_error
            and major_radius > 1.1 * minor_radius
            and minor_radius > EPSILON
            and minor_radius < 2.0 * model_scale
        )
        if not valid_torus:
            continue
        torus_parameters = dict(torus_parameters)
        torus_parameters["clip_bounds"] = clip_bounds.copy()
        candidates.append(
            PrimitiveCandidate(
                primitive_type="TORUS",
                parameters=torus_parameters,
                patch_ids=[patch_id],
                fitting_error=float(torus_error),
                confidence=float(0.25 * np.exp(-10.0 * torus_error)),
                operation="ADD",
                metadata={
                    "transition": True,
                    "transition_type": transition["feature_type"],
                    "support_patch_ids": list(support_patch_ids),
                    "clip_bounds": clip_bounds.copy(),
                },
            )
        )
    return candidates


def _mesh_boundary_loops(mesh):
    """Return ordered vertex loops from a patch mesh boundary."""
    sorted_edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    if len(sorted_edges) == 0:
        return []
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    edges = unique_edges[counts == 1]
    if len(edges) == 0:
        return []
    adjacency = {}
    unused = set()
    for first, second in edges:
        edge = tuple(sorted((int(first), int(second))))
        unused.add(edge)
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    loops = []
    while unused:
        first_edge = next(iter(unused))
        start, current = first_edge
        loop = [start, current]
        unused.remove(first_edge)
        previous = start
        while current != start:
            choices = [
                vertex
                for vertex in adjacency.get(current, [])
                if tuple(sorted((current, vertex))) in unused
                and vertex != previous
            ]
            if not choices:
                closing = tuple(sorted((current, start)))
                if closing in unused:
                    unused.remove(closing)
                break
            following = choices[0]
            unused.remove(tuple(sorted((current, following))))
            previous, current = current, following
            if current != start:
                loop.append(current)
        if len(loop) >= 4:
            loops.append(np.asarray(mesh.vertices[loop], dtype=float))
    return loops


def _fit_periodic_bspline(profile, sample_count=160):
    """Fit a compact periodic cubic B-spline to one ordered 2-D profile."""
    profile = np.asarray(profile, dtype=float).reshape(-1, 2)
    if len(profile) < 4:
        raise ValueError("a periodic spline profile needs at least four points")
    keep = np.ones(len(profile), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(profile, axis=0), axis=1) > 1e-10
    profile = profile[keep]
    if len(profile) >= 2 and np.linalg.norm(profile[0] - profile[-1]) <= 1e-10:
        profile = profile[:-1]
    if len(profile) < 4:
        raise ValueError("degenerate spline profile")
    scale = max(float(np.linalg.norm(np.ptp(profile, axis=0))), EPSILON)
    degree = min(3, len(profile) - 1)
    smoothing = len(profile) * (2e-3 * scale) ** 2
    knots_and_controls, parameters = scipy.interpolate.splprep(
        [profile[:, 0], profile[:, 1]],
        k=degree,
        s=smoothing,
        per=True,
    )
    knots, controls, degree = knots_and_controls
    controls = np.column_stack(controls)
    sample_parameters = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    sampled = np.column_stack(
        scipy.interpolate.splev(sample_parameters, knots_and_controls)
    )
    fitted = np.column_stack(scipy.interpolate.splev(parameters, knots_and_controls))
    error = float(np.sqrt(np.mean(np.sum((fitted - profile) ** 2, axis=1))) / scale)
    signed_area = 0.5 * np.sum(
        sampled[:, 0] * np.roll(sampled[:, 1], -1)
        - sampled[:, 1] * np.roll(sampled[:, 0], -1)
    )
    if signed_area < 0.0:
        sampled = sampled[::-1]
    return {
        "spline_degree": int(degree),
        "spline_knots": np.asarray(knots, dtype=float),
        "spline_control_points": controls,
        "spline_periodic": True,
        "sampled_profile": sampled,
        "spline_fit_error": error,
    }


def _spline_profile_loop(result_lookup, side_ids, cap_ids, origin, axis, basis, extent):
    """Extract the best ordered end profile from caps or side boundaries."""
    height = max(float(np.ptp(extent)), EPSILON)
    axial_tolerance = max(0.03 * height, 1e-8)
    side_points = np.vstack(
        [
            np.asarray(result_lookup[patch_id]["mesh"].vertices, dtype=float)
            for patch_id in side_ids
        ]
    )
    side_span = np.ptp((side_points - origin) @ basis, axis=0)
    options = []
    for patch_id in list(cap_ids) + list(side_ids):
        mesh = result_lookup[patch_id]["mesh"]
        for loop in _mesh_boundary_loops(mesh):
            axial = (loop - origin) @ axis
            nearest_end = min(
                abs(float(np.mean(axial)) - extent[0]),
                abs(float(np.mean(axial)) - extent[1]),
            )
            if np.ptp(axial) > axial_tolerance or nearest_end > axial_tolerance:
                continue
            local = (loop - origin) @ basis
            perimeter = float(
                np.sum(np.linalg.norm(local - np.roll(local, 1, axis=0), axis=1))
            )
            if perimeter <= EPSILON:
                continue
            # The profile loop belongs to the extruded feature, so it cannot
            # be much larger than the side surface itself.  This rejects the
            # boundary loop of a large supporting face (for example the cube
            # face a free-form boss sits on).
            loop_span = np.ptp(local, axis=0)
            consistent = bool(
                np.all(loop_span <= 1.3 * side_span + 0.02 * max(side_span))
            )
            options.append((perimeter, local, consistent))
    if not options:
        return None
    consistent_options = [option for option in options if option[2]]
    pool = consistent_options or options
    return max(pool, key=lambda item: item[0])[1]


def _grow_planar_spline_sides(
    result_lookup, neighbors, side_ids, side_axes, axis, min_surface_fit_rate
):
    """Extend a spline side group with adjacent flat profile segments.

    Straight stretches of an otherwise free-form profile are segmented as
    planar patches and excluded from the seed set by the analytic-fit
    filter, but they still belong to the extruded side surface.
    """
    side_set = set(side_ids)
    changed = True
    while changed:
        changed = False
        side_points = np.vstack(
            [
                np.asarray(result_lookup[patch_id]["mesh"].vertices, dtype=float)
                for patch_id in sorted(side_set)
            ]
        )
        axial = (side_points - side_points.mean(axis=0)) @ axis
        lower, upper = float(axial.min()), float(axial.max())
        margin = 0.1 * max(upper - lower, EPSILON)
        for side_id in sorted(side_set):
            for adjacent in neighbors.get(side_id, {}):
                if adjacent in side_set or adjacent in side_axes:
                    continue
                result = result_lookup.get(adjacent)
                if result is None:
                    continue
                plane_rate, plane_parameters = _seed_fit(result, "PLANE")
                if plane_rate < 0.8 or plane_parameters.size < 3:
                    continue
                spline_rate, _ = _seed_fit(result, "SPLINE_EXTRUSION")
                if spline_rate < min_surface_fit_rate:
                    continue
                normal = _normalize(plane_parameters[:3])
                if normal is None or abs(np.dot(normal, axis)) > np.sin(np.deg2rad(15.0)):
                    continue
                vertices = np.asarray(result["mesh"].vertices, dtype=float)
                axial_range = (vertices - side_points.mean(axis=0)) @ axis
                if (
                    axial_range.min() < lower - margin
                    or axial_range.max() > upper + margin
                ):
                    continue
                side_set.add(adjacent)
                changed = True
    return sorted(side_set)


def generate_spline_extrusion_candidates(
    results,
    graph,
    min_surface_fit_rate=0.82,
    maximum_analytic_fit_rate=0.82,
):
    """Recover solids made by extruding a periodic free-form spline curve.

    Side patches are recognized by their Gauss-sphere normals lying in a
    plane, even when plane/cylinder/cone/sphere fits are all poor.  Ordered
    boundary loops from the planar caps or side surface are then fitted with a
    periodic B-spline.  The resulting parameters describe both the profile
    curve and the tensor-product extrusion surface.
    """
    result_lookup = _result_lookup(results)
    neighbors = _aggregated_graph(graph)
    side_axes = {}
    for patch_id, result in result_lookup.items():
        rate, parameters = _seed_fit(result, "SPLINE_EXTRUSION")
        if rate < min_surface_fit_rate or parameters.size < 3:
            continue
        analytic_rate = max(
            _seed_fit(result, primitive_type)[0]
            for primitive_type in ("PLANE", "CYLINDER", "CONE", "SPHERE")
        )
        if analytic_rate >= maximum_analytic_fit_rate:
            continue
        axis = _normalize(parameters[:3])
        if axis is not None:
            side_axes[patch_id] = axis
    if not side_axes:
        return []

    groups = []
    remaining = set(side_axes)
    cosine = np.cos(np.deg2rad(8.0))
    while remaining:
        seed = min(remaining)
        group = set()
        stack = [seed]
        while stack:
            current = stack.pop()
            if current in group or current not in remaining:
                continue
            group.add(current)
            stack.extend(
                adjacent
                for adjacent in neighbors.get(current, {})
                if adjacent in remaining
                and abs(np.dot(side_axes[current], side_axes[adjacent])) >= cosine
            )
        remaining.difference_update(group)
        groups.append(sorted(group))

    candidates = []
    for side_ids in groups:
        reference_axis = side_axes[side_ids[0]]
        aligned_axes = [
            axis if np.dot(axis, reference_axis) >= 0.0 else -axis
            for axis in (side_axes[index] for index in side_ids)
        ]
        axis = _normalize(np.mean(aligned_axes, axis=0))
        if axis is None:
            continue
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axis = -axis
        side_ids = _grow_planar_spline_sides(
            result_lookup, neighbors, side_ids, side_axes, axis, min_surface_fit_rate
        )
        side_points = np.vstack(
            [np.asarray(result_lookup[index]["mesh"].vertices, dtype=float) for index in side_ids]
        )
        origin = np.mean(side_points, axis=0)
        first, second = _orthogonal_basis(axis)
        basis = np.column_stack([first, second])
        axial = (side_points - origin) @ axis
        extent = np.array([axial.min(), axial.max()], dtype=float)
        if np.ptp(extent) <= EPSILON:
            continue

        cap_ids = []
        for side_id in side_ids:
            for adjacent in neighbors.get(side_id, {}):
                if adjacent in side_ids or adjacent in cap_ids:
                    continue
                plane_rate, plane_parameters = _seed_fit(result_lookup[adjacent], "PLANE")
                if plane_rate < 0.8 or plane_parameters.size < 3:
                    continue
                normal = _normalize(plane_parameters[:3])
                if normal is None or abs(np.dot(normal, axis)) < np.cos(np.deg2rad(12.0)):
                    continue
                cap_vertices = np.asarray(result_lookup[adjacent]["mesh"].vertices, dtype=float)
                cap_axial = (cap_vertices - origin) @ axis
                if np.min(np.abs(cap_axial[:, None] - extent[None, :])) <= 0.04 * np.ptp(extent):
                    cap_ids.append(adjacent)

        profile = _spline_profile_loop(
            result_lookup, side_ids, cap_ids, origin, axis, basis, extent
        )
        if profile is None:
            continue
        try:
            spline = _fit_periodic_bspline(profile)
        except (ValueError, TypeError, scipy.linalg.LinAlgError):
            continue
        sampled_profile = spline["sampled_profile"]
        if len(sampled_profile) < 8:
            continue
        # Caps must be covered by the recovered profile.  A large neighboring
        # plane (for example the face a boss stands on) passes the axial cap
        # test but lies mostly outside the actual profile.
        profile_scale = max(
            float(np.linalg.norm(np.ptp(sampled_profile, axis=0))), EPSILON
        )
        boundary_tolerance = max(1e-5, 0.02 * profile_scale)
        kept_caps = []
        for cap_id in cap_ids:
            cap_vertices = np.asarray(
                result_lookup[cap_id]["mesh"].vertices, dtype=float
            )
            cap_local = (cap_vertices - origin) @ basis
            inside = _point_in_polygon(cap_local, sampled_profile)
            near = (
                _polygon_boundary_distance(cap_local, sampled_profile)
                <= boundary_tolerance
            )
            if np.count_nonzero(inside | near) >= 0.9 * len(cap_local):
                kept_caps.append(cap_id)
        cap_ids = kept_caps
        control_points = spline["spline_control_points"]
        control_profile_world = origin + control_points @ basis.T
        surface_control_net = np.stack(
            [
                control_profile_world + extent[0] * axis,
                control_profile_world + extent[1] * axis,
            ],
            axis=1,
        )
        side_error = float(
            np.mean(
                [1.0 - _seed_fit(result_lookup[index], "SPLINE_EXTRUSION")[0] for index in side_ids]
            )
        )
        fitting_error = float(side_error + spline["spline_fit_error"])
        parameters = {
            "axis_point": origin,
            "axis": axis,
            "extent": extent,
            "basis": basis,
            "polygon": sampled_profile,
            "profile_kind": "PERIODIC_BSPLINE",
            **spline,
            "surface_degrees": np.array([spline["spline_degree"], 1], dtype=int),
            "surface_knots_u": spline["spline_knots"].copy(),
            "surface_knots_v": np.array([0.0, 0.0, 1.0, 1.0]),
            "surface_control_net": surface_control_net,
            "surface_parameterization": "S(u,v)=C(u)+v*axis",
            "side_patch_ids": side_ids,
            "cap_patch_ids": sorted(set(cap_ids)),
        }
        patch_ids = sorted(set(side_ids) | set(cap_ids))
        candidates.append(
            PrimitiveCandidate(
                primitive_type="SPLINE_EXTRUSION",
                parameters=parameters,
                patch_ids=patch_ids,
                fitting_error=fitting_error,
                confidence=float(0.8 * np.exp(-8.0 * fitting_error)),
                metadata={
                    "profile_kind": "PERIODIC_BSPLINE",
                    "side_patch_ids": side_ids,
                    "cap_patch_ids": sorted(set(cap_ids)),
                },
            )
        )
    return candidates


def _drop_cube_covered_rectangles(candidates, results, min_area_overlap=0.9, min_cap_driven_overlap=0.45):
    """Drop extrusions whose surface a cube candidate already explains.

    A rectangular prism and a cube describe the same solid.  When a cube
    candidate covers nearly the same patches, the extrusion duplicate would
    otherwise compete with the more structured cube explanation during
    beam-search selection (same reasoning as the large-rectangle rejection
    in ``generate_extrusion_candidates``).  Cap-driven prisms are
    speculative by construction, so they are already dropped when cube
    faces make up roughly half of their surface.
    """
    cubes = [candidate for candidate in candidates if candidate.primitive_type == "CUBE"]
    if not cubes:
        return candidates
    result_lookup = _result_lookup(results)

    def area_of(patch_ids):
        return sum(
            float(result_lookup[patch_id].get("area", 0.0)) for patch_id in patch_ids
        )

    kept = []
    for candidate in candidates:
        if candidate.primitive_type != "EXTRUSION":
            kept.append(candidate)
            continue
        own_area = area_of(candidate.patch_ids)
        if own_area <= EPSILON:
            kept.append(candidate)
            continue
        overlap = max(
            (
                area_of(set(candidate.patch_ids).intersection(cube.patch_ids))
                for cube in cubes
            ),
            default=0.0,
        )
        if candidate.metadata.get("rectangular") and overlap / own_area >= min_area_overlap:
            continue
        if (
            candidate.metadata.get("cap_driven")
            and overlap / own_area >= min_cap_driven_overlap
        ):
            continue
        kept.append(candidate)
    return kept


def _truncate_cubes_at_boundary_planes(candidates, results, graph, min_plane_fit_rate=0.8):
    """Clip cube candidates at neighboring boundary planes their box overhangs.

    A rotated bar whose visible faces stop at a neighboring body's face
    plane (for example an arm plate flush with the bottom of the beam it
    passes through) is recovered from its patches as a box whose tilted
    corner leaks past that plane into open space.  When every patch of the
    cube lies on one side of a world-axis-aligned plane evidenced by an
    adjacent patch, clip the cube at that plane.
    """
    neighbors = _aggregated_graph(graph)
    result_lookup = _result_lookup(results)
    all_vertices = np.vstack(
        [np.asarray(result["mesh"].vertices, dtype=float) for result in results]
    )
    model_scale = max(float(np.linalg.norm(np.ptp(all_vertices, axis=0))), EPSILON)
    tolerance = 1e-4 * model_scale
    axis_cosine = np.cos(np.deg2rad(2.0))
    for candidate in candidates:
        if candidate.primitive_type != "CUBE":
            continue
        member = set(candidate.patch_ids)
        if not member:
            continue
        member_vertices = np.vstack(
            [
                np.asarray(result_lookup[patch_id]["mesh"].vertices, dtype=float)
                for patch_id in sorted(member)
            ]
        )
        adjacent = sorted(
            {
                neighbor
                for patch_id in member
                for neighbor in neighbors.get(patch_id, {})
                if neighbor not in member
            }
        )
        for patch_id in adjacent:
            result = result_lookup.get(patch_id)
            if result is None:
                continue
            rate, plane_parameters = _seed_fit(result, "PLANE")
            if rate < min_plane_fit_rate or plane_parameters.size < 4:
                continue
            normal = _normalize(plane_parameters[:3])
            if normal is None:
                continue
            dominant = int(np.argmax(np.abs(normal)))
            if abs(normal[dominant]) < axis_cosine:
                continue
            position = -float(plane_parameters[3]) / float(plane_parameters[dominant])
            box = primitive_bounds(candidate)
            lower = float(box[0, dominant])
            upper = float(box[1, dominant])
            if not (lower + tolerance < position < upper - tolerance):
                continue
            # The cube's own surface evidence must lie (essentially) on one
            # side of the plane; patches straddling the plane mean the box
            # legitimately continues past it.
            coord = member_vertices[:, dominant]
            below = float(np.mean(coord < position - tolerance))
            above = float(np.mean(coord > position + tolerance))
            if below > 0.005 and above > 0.005:
                continue
            clip = np.asarray(
                candidate.parameters.get("clip_bounds", box), dtype=float
            ).copy()
            if below <= 0.005:
                clip[0, dominant] = max(clip[0, dominant], position)
            else:
                clip[1, dominant] = min(clip[1, dominant], position)
            if np.any(clip[1] - clip[0] <= tolerance):
                continue
            candidate.parameters["clip_bounds"] = clip
            candidate.metadata["boundary_clip"] = True
    return candidates


def generate_pocket_floor_cutters(results, graph, extrusion_candidates, min_plane_fit_rate=0.8):
    """Recover the central void enclosed by groove prisms around a floor.

    A cross recess cut by groove prisms (one per arm) leaves its deepest
    center column outside every groove profile: the floor patch sits at the
    bottom, bounded laterally by the groove walls.  The center cutter is a
    box from the floor plane up past the groove tops, bounded laterally by
    the inner envelope of the adjacent prisms' cap walls.
    """
    result_lookup = _result_lookup(results)
    neighbors = _aggregated_graph(graph)
    all_vertices = np.vstack(
        [np.asarray(result["mesh"].vertices, dtype=float) for result in results]
    )
    model_scale = max(float(np.linalg.norm(np.ptp(all_vertices, axis=0))), EPSILON)
    prisms = [
        candidate
        for candidate in extrusion_candidates
        if candidate.primitive_type in {"EXTRUSION", "SPLINE_EXTRUSION"}
    ]
    cutters = []
    for patch_id, result in result_lookup.items():
        if str(result.get("type", "")).lower() != "plane":
            continue
        rate, plane_parameters = _seed_fit(result, "PLANE")
        if rate < min_plane_fit_rate or plane_parameters.size < 4:
            continue
        normal = _normalize(plane_parameters[:3])
        if normal is None:
            continue
        dominant = int(np.argmax(np.abs(normal)))
        if abs(normal[dominant]) < np.cos(np.deg2rad(2.0)):
            continue
        adjacent_prisms = []
        for candidate in prisms:
            sides = candidate.metadata.get("side_patch_ids") or candidate.parameters.get(
                "side_patch_ids", []
            )
            if not set(sides).intersection(neighbors.get(patch_id, {})):
                continue
            axis = _normalize(candidate.parameters.get("axis", []))
            if axis is None or abs(float(np.dot(axis, normal))) >= np.sin(np.deg2rad(15.0)):
                continue
            adjacent_prisms.append(candidate)
        if len(adjacent_prisms) < 2:
            continue
        # The groove walls must open to the same side of the floor plane
        # (the void sits on one side of a pocket bottom).
        floor_s = float(np.dot(normal, np.asarray(result["mesh"].vertices, dtype=float).mean(axis=0)))
        side_vertices = []
        for candidate in adjacent_prisms:
            for side_id in candidate.metadata.get("side_patch_ids") or candidate.parameters.get("side_patch_ids", []):
                side_vertices.append(
                    np.asarray(result_lookup[side_id]["mesh"].vertices, dtype=float)
                )
        side_vertices = np.vstack(side_vertices)
        side_sign = np.sign((side_vertices - np.asarray(result["mesh"].vertices).mean(axis=0)) @ normal)
        if np.mean(side_sign >= 0) < 0.9 and np.mean(side_sign <= 0) < 0.9:
            continue
        if np.mean(side_sign <= 0) > np.mean(side_sign >= 0):
            normal = -normal
            floor_s = -floor_s
        center = np.asarray(result["mesh"].vertices, dtype=float).mean(axis=0)
        lateral_axes = [axis_i for axis_i in range(3) if axis_i != dominant]
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        top_s = -np.inf
        for candidate in adjacent_prisms:
            for cap_id in candidate.parameters.get("cap_patch_ids", []):
                cap_vertices = np.asarray(
                    result_lookup[cap_id]["mesh"].vertices, dtype=float
                )
                for axis_i in lateral_axes:
                    coords = cap_vertices[:, axis_i]
                    below = coords[coords < center[axis_i]]
                    above = coords[coords > center[axis_i]]
                    if len(below):
                        lower[axis_i] = min(lower[axis_i], float(np.max(below)))
                    if len(above):
                        upper[axis_i] = max(upper[axis_i], float(np.min(above)))
            bounds = primitive_bounds(candidate)
            corners = np.array(
                [[bounds[i][0], bounds[j][1], bounds[k][2]]
                 for i in range(2) for j in range(2) for k in range(2)]
            )
            top_s = max(top_s, float(np.max(corners @ normal)))
        lateral = np.array([lower[a] for a in lateral_axes] + [upper[a] for a in lateral_axes])
        if not np.isfinite(lateral).all() or not np.isfinite(top_s):
            continue
        lower = lower[lateral_axes]
        upper = upper[lateral_axes]
        if np.any(upper - lower <= EPSILON):
            continue
        # The wall envelope must tightly frame the floor patch; a much
        # larger or smaller envelope means the patch is an ordinary body
        # face, not a pocket bottom enclosed by groove walls.
        floor_vertices = np.asarray(result["mesh"].vertices, dtype=float)
        floor_lo = floor_vertices.min(axis=0)[lateral_axes]
        floor_hi = floor_vertices.max(axis=0)[lateral_axes]
        lat_tol = 0.3 * np.maximum(floor_hi - floor_lo, EPSILON) + 1e-4 * model_scale
        if np.any(lower < floor_lo - lat_tol) or np.any(upper > floor_hi + lat_tol):
            continue
        if np.any(lower > floor_lo + lat_tol) or np.any(upper < floor_hi - lat_tol):
            continue
        margin = 0.02 * model_scale
        lateral_basis = np.column_stack([np.eye(3)[:, a] for a in lateral_axes])
        rotation = np.column_stack([lateral_basis, normal])
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] *= -1.0
        size = np.concatenate(
            [upper - lower, [top_s + margin - floor_s]]
        )
        if np.any(size <= EPSILON):
            continue
        center_local = np.concatenate(
            [(upper + lower) / 2.0, [(floor_s + top_s + margin) / 2.0]]
        )
        cube_center = rotation @ center_local
        cutters.append(
            PrimitiveCandidate(
                primitive_type="CUBE",
                parameters={
                    "center": cube_center,
                    "size": size,
                    "rotation": rotation,
                    "euler_xyz_degrees": _matrix_to_euler_xyz(rotation),
                    "local_bounds": np.stack([-size / 2.0, size / 2.0]),
                },
                patch_ids=[patch_id],
                fitting_error=0.0,
                confidence=0.6,
                metadata={"pocket_floor": True},
            )
        )
    return cutters


def generate_planar_notch_cutters(results, graph, cylinder_candidates, min_plane_fit_rate=0.8):
    """Generate box cutters for planar chord notches on cylinder candidates.

    A flat notch milled into a cylinder (the cutting plane is parallel to
    the cylinder axis and spans only part of its height) exposes only a
    small rectangular face, which fits no generator: cube recovery needs at
    least three faces, extrusion recovery needs three side walls, and a
    cap pair needs two congruent end faces.  The cutting box is recovered
    analytically from the chord plane and the cylinder parameters instead.
    """
    result_lookup = _result_lookup(results)
    neighbors = _aggregated_graph(graph)
    all_vertices = np.vstack(
        [np.asarray(result["mesh"].vertices, dtype=float) for result in results]
    )
    model_scale = max(float(np.linalg.norm(np.ptp(all_vertices, axis=0))), EPSILON)
    cutters = []
    for cylinder in cylinder_candidates:
        if cylinder.primitive_type != "CYLINDER":
            continue
        parameters = cylinder.parameters
        axis = _normalize(parameters.get("axis"))
        if axis is None:
            continue
        origin = np.asarray(parameters["axis_point"], dtype=float)
        radius = abs(float(parameters["radius"]))
        extent = np.asarray(parameters["extent"], dtype=float)
        height = max(float(extent[1] - extent[0]), EPSILON)
        cap_patch_ids = set(cylinder.metadata.get("cap_patch_ids", []))
        side_patch_ids = [
            patch_id for patch_id in cylinder.patch_ids if patch_id not in cap_patch_ids
        ]
        side_area = sum(
            float(result_lookup[patch_id].get("area", 0.0)) for patch_id in side_patch_ids
        )
        adjacent = sorted(
            {
                neighbor
                for patch_id in side_patch_ids
                for neighbor in neighbors.get(patch_id, {})
                if neighbor not in cylinder.patch_ids
            }
        )
        for patch_id in adjacent:
            result = result_lookup.get(patch_id)
            if result is None or str(result.get("type", "")).lower() != "plane":
                continue
            rate, plane_parameters = _seed_fit(result, "PLANE")
            if rate < min_plane_fit_rate or plane_parameters.size < 4:
                continue
            normal = _normalize(plane_parameters[:3])
            if normal is None:
                continue
            # The notch plane is parallel to the cylinder axis.
            if abs(float(np.dot(normal, axis))) > np.sin(np.deg2rad(10.0)):
                continue
            vertices = np.asarray(result["mesh"].vertices, dtype=float)
            mean = vertices.mean(axis=0)
            closest = origin + axis * float((mean - origin) @ axis)
            radial_vector = mean - closest
            distance = float(np.linalg.norm(radial_vector))
            # A chord plane cuts through the cross-section; a tangent or an
            # outside face does not.
            if distance >= radius - 0.02 * radius:
                continue
            # The notch face lies on the chord, so all of its vertices are
            # inside the cylinder; a large body face the cylinder merely
            # touches is rejected here.
            relative = vertices - origin
            radial = np.linalg.norm(relative - (relative @ axis)[:, None] * axis, axis=1)
            if np.max(radial) > radius + max(0.02 * radius, 1e-4 * model_scale):
                continue
            axial = relative @ axis
            axial_tolerance = 0.05 * height
            if (
                axial.min() < extent[0] - axial_tolerance
                or axial.max() > extent[1] + axial_tolerance
            ):
                continue
            # A notch face is much smaller than the wall it cuts.
            if float(result.get("area", result["mesh"].area)) > 0.5 * max(side_area, EPSILON):
                continue
            y_dir = _normalize(radial_vector)
            x_dir = _normalize(np.cross(axis, y_dir))
            if y_dir is None or x_dir is None:
                continue
            rotation = np.column_stack([x_dir, y_dir, axis])
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            margin = 0.05 * radius
            z_min = float(axial.min())
            z_max = float(axial.max())
            # Extend through any cylinder end the notch reaches.
            if z_max >= extent[1] - axial_tolerance:
                z_max = max(z_max, float(extent[1])) + margin
            if z_min <= extent[0] + axial_tolerance:
                z_min = min(z_min, float(extent[0])) - margin
            chord_half = float(np.sqrt(max(radius**2 - distance**2, 0.0)))
            center_local = np.array(
                [0.0, (distance + radius + margin) / 2.0, (z_min + z_max) / 2.0]
            )
            size = np.array(
                [2.0 * chord_half, radius + margin - distance, z_max - z_min]
            )
            if np.any(size <= EPSILON):
                continue
            center = closest + rotation @ center_local
            local_bounds = np.stack(
                [center_local - size / 2.0, center_local + size / 2.0]
            )
            cutter = PrimitiveCandidate(
                primitive_type="CUBE",
                parameters={
                    "center": center,
                    "size": size,
                    "rotation": rotation,
                    "euler_xyz_degrees": _matrix_to_euler_xyz(rotation),
                    "local_bounds": local_bounds,
                },
                patch_ids=[patch_id],
                fitting_error=0.0,
                confidence=0.7,
                metadata={"notch_of_cylinder": list(side_patch_ids)},
            )
            # Claim small planar patches that close the notch (for example
            # the floor at the blind end), testing box containment loosely
            # because intersection curves are tessellated approximately.
            claimed = set(cutter.patch_ids)
            for neighbor in neighbors.get(patch_id, {}):
                if neighbor in claimed or neighbor in cylinder.patch_ids:
                    continue
                neighbor_result = result_lookup.get(neighbor)
                if (
                    neighbor_result is None
                    or str(neighbor_result.get("type", "")).lower() != "plane"
                ):
                    continue
                neighbor_vertices = np.asarray(
                    neighbor_result["mesh"].vertices, dtype=float
                )
                if len(neighbor_vertices) == 0:
                    continue
                inside = primitive_contains(
                    "CUBE",
                    cutter.parameters,
                    neighbor_vertices,
                    tolerance=max(0.1 * radius, 1e-4 * model_scale),
                )
                if np.all(inside):
                    claimed.add(neighbor)
            cutter.patch_ids = sorted(claimed)
            cutters.append(cutter)
    return cutters


def generate_primitive_candidates(
    results,
    graph,
    include_torus=True,
    min_seed_rate=0.55,
    include_spline_extrusion=True,
):
    """Generate and rank cube, extrusion, and curved primitive hypotheses."""
    classify_transition_patches(results, graph)
    candidates = generate_cube_candidates(results, graph)
    candidates.extend(generate_extrusion_candidates(results, graph))
    if include_spline_extrusion:
        candidates.extend(generate_spline_extrusion_candidates(results, graph))
    curved_types = ["CYLINDER", "CONE", "SPHERE"]
    if include_torus:
        curved_types.append("TORUS")
    candidates.extend(
        generate_curved_candidates(
            results, graph, primitive_types=tuple(curved_types), min_seed_rate=min_seed_rate
        )
    )
    candidates.extend(generate_transition_curved_candidates(results, graph))
    candidates.extend(generate_planar_notch_cutters(results, graph, candidates))
    candidates = _drop_cube_covered_rectangles(candidates, results)
    candidates.extend(generate_pocket_floor_cutters(results, graph, candidates))
    _truncate_cubes_at_boundary_planes(candidates, results, graph)
    unique_candidates = {}
    for candidate in candidates:
        key = (candidate.primitive_type, tuple(candidate.patch_ids), bool(candidate.metadata.get("transition")))
        previous = unique_candidates.get(key)
        if previous is None or candidate.fitting_error < previous.fitting_error:
            unique_candidates[key] = candidate
    candidates = list(unique_candidates.values())
    candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.fitting_error))
    return candidates


def primitive_bounds(candidate):
    """Return an axis-aligned world-space bounding box for a candidate."""
    parameters = candidate.parameters
    primitive_type = candidate.primitive_type

    def clipped(bounds):
        clip_bounds = parameters.get("clip_bounds")
        if clip_bounds is None:
            return bounds
        clip_bounds = np.asarray(clip_bounds, dtype=float)
        if clip_bounds.shape != (2, 3) or not np.all(np.isfinite(clip_bounds)):
            return bounds
        lower = np.maximum(bounds[0], clip_bounds[0])
        upper = np.minimum(bounds[1], clip_bounds[1])
        return np.stack([lower, np.maximum(lower, upper)])

    if primitive_type == "CUBE":
        center = np.asarray(parameters["center"])
        rotation = np.asarray(parameters["rotation"])
        half_size = np.asarray(parameters["size"]) / 2.0
        extent = np.abs(rotation) @ half_size
        return clipped(np.stack([center - extent, center + extent]))
    if primitive_type == "SPHERE":
        center = np.asarray(parameters["center"])
        radius = float(parameters["radius"])
        return clipped(np.stack([center - radius, center + radius]))
    if primitive_type in {"CYLINDER", "CONE"}:
        axis = np.asarray(parameters["axis"])
        origin = np.asarray(
            parameters.get("axis_point", parameters.get("apex")), dtype=float
        )
        extent = np.asarray(parameters["extent"])
        endpoints = origin + extent[:, None] * axis
        if primitive_type == "CYLINDER":
            radius = float(parameters["radius"])
        else:
            radius = float(max(abs(extent)) * np.tan(parameters["angle"]))
        radial_extent = radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
        return clipped(np.stack(
            [
                endpoints.min(axis=0) - radial_extent,
                endpoints.max(axis=0) + radial_extent,
            ]
        ))
    if primitive_type == "TORUS":
        center = np.asarray(parameters["center"])
        radius = float(parameters["major_radius"] + parameters["minor_radius"])
        return clipped(np.stack([center - radius, center + radius]))
    if primitive_type in {"EXTRUSION", "SPLINE_EXTRUSION"}:
        origin = np.asarray(parameters["axis_point"], dtype=float)
        axis = np.asarray(parameters["axis"], dtype=float)
        basis = np.asarray(parameters["basis"], dtype=float)
        polygon = np.asarray(parameters["polygon"], dtype=float)
        extent = np.asarray(parameters["extent"], dtype=float)
        profile = origin + polygon @ basis.T
        vertices = np.vstack(
            [profile + extent[0] * axis, profile + extent[1] * axis]
        )
        return np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    raise ValueError(f"unsupported primitive: {primitive_type}")


# Boolean classification and CSG search


def _primitive_normals(candidate, points, point_patch_ids=None):
    points = np.asarray(points, dtype=float)
    parameters = candidate.parameters
    primitive_type = candidate.primitive_type
    if primitive_type == "CUBE":
        center = np.asarray(parameters["center"])
        rotation = np.asarray(parameters["rotation"])
        half_size = np.asarray(parameters["size"]) / 2.0
        local = (points - center) @ rotation
        distance = np.abs(np.abs(local) - half_size)
        axes = np.argmin(distance, axis=1)
        local_normals = np.zeros_like(local)
        local_normals[np.arange(len(points)), axes] = np.sign(local[np.arange(len(points)), axes])
        return local_normals @ rotation.T
    if primitive_type == "SPHERE":
        vectors = points - np.asarray(parameters["center"])
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), EPSILON)
    if primitive_type == "CYLINDER":
        axis = np.asarray(parameters["axis"])
        relative = points - np.asarray(parameters["axis_point"])
        radial = relative - (relative @ axis)[:, None] * axis
        return radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), EPSILON)
    if primitive_type == "CONE":
        axis = np.asarray(parameters["axis"])
        relative = points - np.asarray(parameters["apex"])
        axial = relative @ axis
        radial = relative - axial[:, None] * axis
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), EPSILON)
        normals = radial - np.tan(parameters["angle"]) * axis
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), EPSILON)
    if primitive_type == "TORUS":
        axis = np.asarray(parameters["axis"])
        relative = points - np.asarray(parameters["center"])
        axial = relative @ axis
        radial = relative - axial[:, None] * axis
        radial_unit = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), EPSILON)
        tube_center = radial_unit * parameters["major_radius"]
        normals = relative - tube_center
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), EPSILON)
    if primitive_type in {"EXTRUSION", "SPLINE_EXTRUSION"}:
        origin = np.asarray(parameters["axis_point"], dtype=float)
        axis = np.asarray(parameters["axis"], dtype=float)
        basis = np.asarray(parameters["basis"], dtype=float)
        polygon = np.asarray(parameters["polygon"], dtype=float)
        extent = np.asarray(parameters["extent"], dtype=float)
        relative = points - origin
        axial = relative @ axis
        local = relative @ basis
        edge_vectors = np.roll(polygon, -1, axis=0) - polygon
        edge_lengths = np.maximum(np.linalg.norm(edge_vectors, axis=1), EPSILON)
        edge_distances = np.abs(
            edge_vectors[None, :, 0] * (local[:, None, 1] - polygon[None, :, 1])
            - edge_vectors[None, :, 1] * (local[:, None, 0] - polygon[None, :, 0])
        ) / edge_lengths[None, :]
        nearest_edges = np.argmin(edge_distances, axis=1)
        local_normals = np.column_stack(
            [edge_vectors[:, 1], -edge_vectors[:, 0]]
        ) / edge_lengths[:, None]
        side_normals = local_normals[nearest_edges] @ basis.T
        lower_distance = np.abs(axial - extent[0])
        upper_distance = np.abs(axial - extent[1])
        cap_distance = np.minimum(lower_distance, upper_distance)
        side_distance = edge_distances[np.arange(len(points)), nearest_edges]
        use_cap = cap_distance < side_distance
        if point_patch_ids is not None:
            point_patch_ids = np.asarray(point_patch_ids, dtype=int).reshape(-1)
            side_patch_ids = set(parameters.get("side_patch_ids", []))
            cap_patch_ids = set(parameters.get("cap_patch_ids", []))
            if side_patch_ids or cap_patch_ids:
                use_cap = np.isin(point_patch_ids, list(cap_patch_ids))
        predicted = side_normals
        predicted[use_cap] = np.where(
            (lower_distance[use_cap] <= upper_distance[use_cap])[:, None],
            -axis,
            axis,
        )
        return predicted
    return np.zeros_like(points)

def primitive_contains(operation, parameters, points, tolerance=EPSILON):
    points = np.asarray(points, dtype=float)

    def clipped(mask):
        clip_bounds = parameters.get("clip_bounds")
        if clip_bounds is None:
            return mask
        clip_bounds = np.asarray(clip_bounds, dtype=float)
        if clip_bounds.shape != (2, 3) or not np.all(np.isfinite(clip_bounds)):
            return mask
        return mask & np.all(
            np.logical_and(
                points >= clip_bounds[0] - tolerance,
                points <= clip_bounds[1] + tolerance,
            ),
            axis=1,
        )

    if operation == "CUBE":
        center = np.asarray(parameters["center"])
        rotation = np.asarray(parameters["rotation"])
        half_size = np.asarray(parameters["size"]) / 2.0
        local = (points - center) @ rotation
        return clipped(np.all(np.abs(local) <= half_size + tolerance, axis=1))
    if operation == "SPHERE":
        return clipped(
            np.linalg.norm(points - np.asarray(parameters["center"]), axis=1)
            <= parameters["radius"] + tolerance
        )
    if operation in {"CYLINDER", "CONE"}:
        axis = np.asarray(parameters["axis"])
        origin = np.asarray(parameters.get("axis_point", parameters.get("apex")))
        relative = points - origin
        axial = relative @ axis
        radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
        extent = np.asarray(parameters["extent"])
        extent_mask = np.logical_and(
            axial >= extent[0] - tolerance,
            axial <= extent[1] + tolerance,
        )
        if operation == "CYLINDER":
            return clipped(
                np.logical_and(extent_mask, radial <= parameters["radius"] + tolerance)
            )
        expected_radius = np.maximum(0.0, axial * np.tan(parameters["angle"]))
        return clipped(
            np.logical_and(extent_mask, radial <= expected_radius + tolerance)
        )
    if operation == "TORUS":
        axis = np.asarray(parameters["axis"])
        relative = points - np.asarray(parameters["center"])
        axial = relative @ axis
        radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
        tube = np.sqrt((radial - parameters["major_radius"]) ** 2 + axial**2)
        return clipped(tube <= parameters["minor_radius"] + tolerance)
    if operation in {"EXTRUSION", "SPLINE_EXTRUSION"}:
        origin = np.asarray(parameters["axis_point"], dtype=float)
        axis = np.asarray(parameters["axis"], dtype=float)
        basis = np.asarray(parameters["basis"], dtype=float)
        extent = np.asarray(parameters["extent"], dtype=float)
        relative = points - origin
        axial = relative @ axis
        extent_mask = np.logical_and(
            axial >= extent[0] - tolerance,
            axial <= extent[1] + tolerance,
        )
        local = relative @ basis
        polygon = parameters.get("polygon", parameters.get("control_points_2d"))
        polygon_mask = _point_in_polygon(local, polygon)
        return extent_mask & polygon_mask
    return np.zeros(len(points), dtype=bool)


class PrimitiveFitter:
    """Namespace for the supported analytic surface fitters."""

    cylinder = staticmethod(fit_cylinder)
    cone = staticmethod(fit_cone)
    sphere = staticmethod(fit_sphere)
    torus = staticmethod(fit_torus)


class PrimitiveCandidateGenerator:
    """Generate primitive hypotheses from fitted patches and their graph."""

    def __init__(
        self,
        include_torus=True,
        include_extrusion=True,
        min_seed_rate=0.55,
        include_spline_extrusion=True,
    ):
        self.include_torus = bool(include_torus)
        self.include_extrusion = bool(include_extrusion)
        self.include_spline_extrusion = bool(include_spline_extrusion)
        self.min_seed_rate = float(min_seed_rate)

    def generate_cube(
        self,
        results,
        graph,
        tolerance=None,
        min_patch_count=3,
        min_plane_fit_rate=0.8,
    ):
        return generate_cube_candidates(
            results,
            graph,
            tolerance=tolerance,
            min_patch_count=min_patch_count,
            min_plane_fit_rate=min_plane_fit_rate,
        )

    def generate_curved(
        self,
        results,
        graph,
        primitive_types=("CYLINDER", "CONE", "SPHERE", "TORUS"),
        min_seed_rate=None,
        maximum_error=0.04,
    ):
        seed_rate = self.min_seed_rate if min_seed_rate is None else min_seed_rate
        return generate_curved_candidates(
            results,
            graph,
            primitive_types=primitive_types,
            min_seed_rate=seed_rate,
            maximum_error=maximum_error,
        )

    @staticmethod
    def generate_extrusion(results, graph, min_plane_fit_rate=0.8):
        return generate_extrusion_candidates(
            results,
            graph,
            min_plane_fit_rate=min_plane_fit_rate,
        )

    def generate(self, results, graph):
        classify_transition_patches(results, graph)
        candidates = generate_cube_candidates(results, graph)
        if self.include_extrusion:
            candidates.extend(generate_extrusion_candidates(results, graph))
            if self.include_spline_extrusion:
                candidates.extend(generate_spline_extrusion_candidates(results, graph))
        curved_types = ["CYLINDER", "CONE", "SPHERE"]
        if self.include_torus:
            curved_types.append("TORUS")
        candidates.extend(
            generate_curved_candidates(
                results,
                graph,
                primitive_types=tuple(curved_types),
                min_seed_rate=self.min_seed_rate,
            )
        )
        candidates.extend(generate_transition_curved_candidates(results, graph))
        candidates.extend(generate_planar_notch_cutters(results, graph, candidates))
        candidates = _drop_cube_covered_rectangles(candidates, results)
        candidates.extend(generate_pocket_floor_cutters(results, graph, candidates))
        _truncate_cubes_at_boundary_planes(candidates, results, graph)
        candidates.sort(
            key=lambda candidate: (-candidate.confidence, candidate.fitting_error)
        )
        return candidates


__all__ = [
    "PrimitiveCandidate",
    "PrimitiveCandidateGenerator",
    "PrimitiveFitter",
    "fit_cone",
    "fit_cylinder",
    "fit_sphere",
    "fit_torus",
    "classify_transition_patches",
    "generate_transition_curved_candidates",
    "generate_planar_notch_cutters",
    "generate_pocket_floor_cutters",
    "generate_cube_candidates",
    "generate_curved_candidates",
    "generate_extrusion_candidates",
    "generate_spline_extrusion_candidates",
    "generate_primitive_candidates",
    "primitive_bounds",
    "primitive_contains",
]
