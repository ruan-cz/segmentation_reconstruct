"""Boolean classification, candidate selection, and CSG IR construction."""

import numpy as np
import scipy.spatial

from . import candidates as candidate_module


def classify_boolean_operations(candidates, results, containment_tolerance=1e-4):
    """Classify candidates as ADD or SUBTRACT using normals and containment."""
    orientation_scores = {}
    candidate_points = {}
    for candidate in candidates:
        points, observed_normals, point_patch_ids = candidate_module._sample_patch_points(
            results, candidate.patch_ids, 2500, return_patch_ids=True
        )
        predicted_normals = candidate_module._primitive_normals(
            candidate, points, point_patch_ids=point_patch_ids
        )
        orientation_scores[id(candidate)] = float(
            np.mean(np.sum(observed_normals * predicted_normals, axis=1))
        )
        candidate_points[id(candidate)] = points

    all_points = np.vstack(
        [np.asarray(result["mesh"].vertices, dtype=float) for result in results]
    )
    model_scale = max(np.linalg.norm(np.ptp(all_points, axis=0)), 1.0)
    containment_tolerance = containment_tolerance * model_scale

    for candidate in candidates:
        points = candidate_points[id(candidate)]
        orientation_score = orientation_scores[id(candidate)]
        candidate_bounds = candidate_module.primitive_bounds(candidate)
        possible_containers = []
        candidate_patch_ids = set(candidate.patch_ids)
        for other in candidates:
            if other is candidate:
                continue
            if candidate_patch_ids.intersection(other.patch_ids):
                continue
            if orientation_scores[id(other)] < -0.2:
                continue
            bounds = candidate_module.primitive_bounds(other)
            if np.any(bounds[1] < candidate_bounds[0] - containment_tolerance) or np.any(
                bounds[0] > candidate_bounds[1] + containment_tolerance
            ):
                continue
            possible_containers.append(other)

        container_mask = np.zeros(len(points), dtype=bool)
        used_containers = []
        for other in possible_containers:
            covered = candidate_module.primitive_contains(
                other.primitive_type,
                other.parameters,
                points,
                tolerance=containment_tolerance,
            )
            if np.any(covered):
                used_containers.append(other)
                container_mask |= covered
        containment_coverage = float(np.mean(container_mask)) if len(points) else 0.0
        containers = [
            (other, candidate_module.primitive_bounds(other), containment_tolerance)
            for other in used_containers
        ]

        subtract_score = max(0.0, -orientation_score)
        add_score = max(0.0, orientation_score)
        through_hole_score = 0.0
        for _, bounds, tolerance in containers:
            inside_axes = np.logical_and(
                candidate_bounds[0] > bounds[0] + tolerance,
                candidate_bounds[1] < bounds[1] - tolerance,
            )
            touches_lower = np.abs(candidate_bounds[0] - bounds[0]) <= tolerance
            touches_upper = np.abs(candidate_bounds[1] - bounds[1]) <= tolerance
            if np.count_nonzero(inside_axes) >= 2 and np.any(
                touches_lower & touches_upper
            ):
                through_hole_score = 1.0
                break
        subtraction_supported = containment_coverage >= 0.9 and (
            orientation_score < -0.2 or through_hole_score > 0.0
        )
        if subtraction_supported:
            through_hole_score = 1.0
            subtract_score += 0.5 + through_hole_score
        else:
            # A reversed normal alone does not prove subtraction.  Open
            # surfaces, flipped input winding, and fitting noise can all
            # produce a negative orientation score without a cutter volume.
            subtract_score = 0.0
            add_score += 0.25
        candidate.operation = "SUBTRACT" if subtract_score > add_score else "ADD"
        candidate.metadata.update(
            {
                "orientation_score": orientation_score,
                "containment_count": len(containers),
                "containment_coverage": containment_coverage,
                "container_patch_ids": [
                    list(other.patch_ids) for other, _, _ in containers
                ],
                "through_hole_score": through_hole_score,
                "operation_confidence": float(abs(subtract_score - add_score)),
                "operation_scores": {
                    "ADD": float(add_score),
                    "SUBTRACT": float(subtract_score),
                },
            }
        )
    return candidates


def generate_boolean_hypotheses(candidates, relative_score_threshold=0.35):
    """Expand uncertain candidate operations into ADD/SUBTRACT hypotheses."""
    hypotheses = []
    for candidate in candidates:
        scores = candidate.metadata.get("operation_scores")
        if not scores:
            hypotheses.append(candidate)
            continue
        maximum = max(scores.values())
        if maximum <= candidate_module.EPSILON:
            hypotheses.append(candidate)
            continue
        for operation, score in scores.items():
            if score < maximum * relative_score_threshold:
                continue
            metadata = dict(candidate.metadata)
            metadata["operation_hypothesis_score"] = float(score / maximum)
            hypotheses.append(
                candidate_module.PrimitiveCandidate(
                    primitive_type=candidate.primitive_type,
                    parameters=candidate.parameters,
                    patch_ids=list(candidate.patch_ids),
                    fitting_error=candidate.fitting_error,
                    confidence=float(candidate.confidence * (0.5 + 0.5 * score / maximum)),
                    operation=operation,
                    metadata=metadata,
                )
            )
    return hypotheses


def _can_append_candidate(selected, candidate, results=None):
    selected_patches = {patch for item in selected for patch in item.patch_ids}
    if selected_patches.intersection(candidate.patch_ids):
        return False
    if candidate.operation != "SUBTRACT":
        return True
    candidate_box = candidate_module.primitive_bounds(candidate)
    if results is not None:
        points, _ = candidate_module._sample_patch_points(results, candidate.patch_ids, 2500)
        covered = np.zeros(len(points), dtype=bool)
        model_scale = max(np.linalg.norm(np.ptp(points, axis=0)), 1.0)
        tolerance = 1e-4 * model_scale
        for item in selected:
            if item.operation != "ADD":
                continue
            covered |= candidate_module.primitive_contains(
                item.primitive_type,
                item.parameters,
                points,
                tolerance=tolerance,
            )
        if len(points) and np.mean(covered) >= 0.9:
            return True
        # A concave transition can lie mostly outside each mother solid's
        # fitted extent.  Its observed wall still borders those ADD patches,
        # and the cutter intersects enough of them to shape the union.
        if (
            candidate.primitive_type == "CYLINDER"
            and candidate.metadata.get("orientation_score", 0.0) < -0.8
            and len(points)
            and np.mean(covered) >= 0.3
        ):
            lookup = candidate_module._result_lookup(results)
            neighbors = set()
            for patch_id in candidate.patch_ids:
                neighbors.update(lookup[patch_id].get("neighboring_patches", []))
            adjacent_additive = {
                patch_id
                for item in selected if item.operation == "ADD"
                for patch_id in item.patch_ids
            }
            if neighbors & adjacent_additive:
                return True
    for item in selected:
        if item.operation != "ADD":
            continue
        bounds = candidate_module.primitive_bounds(item)
        tolerance = 1e-4 * max(np.linalg.norm(bounds[1] - bounds[0]), 1.0)
        if np.all(candidate_box[0] >= bounds[0] - tolerance) and np.all(
            candidate_box[1] <= bounds[1] + tolerance
        ):
            return True
    return False


def _selection_score(selected, patch_areas, complexity_penalty):
    total_area = max(sum(patch_areas.values()), candidate_module.EPSILON)
    covered = {patch for candidate in selected for patch in candidate.patch_ids}
    coverage = sum(patch_areas.get(patch, 0.0) for patch in covered) / total_area
    if selected:
        weighted_error = np.average(
            [candidate.fitting_error for candidate in selected],
            weights=[
                sum(
                    patch_areas.get(patch, 0.0)
                    for patch in candidate.patch_ids
                )
                for candidate in selected
            ],
        )
        confidence = np.mean([candidate.confidence for candidate in selected])
    else:
        weighted_error = 1.0
        confidence = 0.0
    score = (
        coverage
        + 0.15 * confidence
        - 0.4 * weighted_error
        - complexity_penalty * len(selected)
    )
    return float(score), float(coverage)


def reconstruct_csg_tree(
    candidates,
    results,
    beam_width=12,
    maximum_primitives=64,
    complexity_penalty=0.025,
    intersection_coverage_threshold=0.85,
):
    """Use beam search to select a compact, high-coverage CSG hypothesis."""
    patch_areas = {
        int(result.get("patch_index", index)): float(
            result.get("area", result["mesh"].area)
        )
        for index, result in enumerate(results)
    }
    ranked = sorted(
        generate_boolean_hypotheses(candidates),
        key=lambda candidate: (
            -(
                sum(
                    patch_areas.get(patch, 0.0)
                    for patch in candidate.patch_ids
                )
                * candidate.confidence
            ),
            candidate.fitting_error,
        ),
    )
    beam = [([], 0.0, 0.0)]
    best = beam[0]
    for _ in range(min(maximum_primitives, len(ranked))):
        expanded = list(beam)
        for selected, _, _ in beam:
            for candidate in ranked:
                if any(candidate is item for item in selected) or not _can_append_candidate(
                    selected, candidate, results
                ):
                    continue
                trial = selected + [candidate]
                score, coverage = _selection_score(trial, patch_areas, complexity_penalty)
                expanded.append((trial, score, coverage))
        unique = {}
        for state in expanded:
            key = tuple(sorted(id(candidate) for candidate in state[0]))
            if key not in unique or state[1] > unique[key][1]:
                unique[key] = state
        beam = sorted(
            unique.values(),
            key=lambda state: (state[2], state[1]),
            reverse=True,
        )[:beam_width]
        if (beam[0][2], beam[0][1]) > (best[2], best[1]):
            best = beam[0]
        if best[2] >= 1.0 - 1e-9:
            break
    selected, score, coverage = best
    csg_ir = build_csg_ir(
        selected,
        results,
        intersection_coverage_threshold=intersection_coverage_threshold,
    )
    return csg_ir, selected, {"search_score": score, "patch_coverage": coverage}


def _candidate_to_ir(candidate):
    parameters = {}
    for key, value in candidate.parameters.items():
        if isinstance(value, np.ndarray):
            parameters[key] = np.asarray(value).tolist()
        elif isinstance(value, np.floating):
            parameters[key] = float(value)
        else:
            parameters[key] = value
    return {
        "op": candidate.primitive_type,
        "parameters": parameters,
        "patch_ids": list(candidate.patch_ids),
        "fitting_error": float(candidate.fitting_error),
        "confidence": float(candidate.confidence),
    }


def _intersection_pair_score(
    first,
    second,
    results,
    tolerance,
    coverage_threshold,
):
    """Score whether two ADD candidates explain an intersection boundary."""
    if set(first.patch_ids).intersection(second.patch_ids):
        return None

    first_bounds = candidate_module.primitive_bounds(first)
    second_bounds = candidate_module.primitive_bounds(second)
    overlap_extent = np.minimum(first_bounds[1], second_bounds[1]) - np.maximum(
        first_bounds[0], second_bounds[0]
    )
    if np.any(overlap_extent <= tolerance):
        return None

    result_lookup = candidate_module._result_lookup(results)
    first_vertices = np.vstack(
        [
            np.asarray(result_lookup[patch_id]["mesh"].vertices, dtype=float)
            for patch_id in first.patch_ids
        ]
    )
    second_vertices = np.vstack(
        [
            np.asarray(result_lookup[patch_id]["mesh"].vertices, dtype=float)
            for patch_id in second.patch_ids
        ]
    )
    second_tree = scipy.spatial.cKDTree(second_vertices)
    shared_boundary_vertices = sum(
        bool(matches)
        for matches in second_tree.query_ball_point(first_vertices, r=tolerance)
    )
    if shared_boundary_vertices < 2:
        return None

    first_points, _ = candidate_module._sample_patch_points(
        results, first.patch_ids, 2500
    )
    second_points, _ = candidate_module._sample_patch_points(
        results, second.patch_ids, 2500
    )
    first_coverage = float(
        np.mean(
            candidate_module.primitive_contains(
                second.primitive_type,
                second.parameters,
                first_points,
                tolerance=tolerance,
            )
        )
    )
    second_coverage = float(
        np.mean(
            candidate_module.primitive_contains(
                first.primitive_type,
                first.parameters,
                second_points,
                tolerance=tolerance,
            )
        )
    )
    mutual_coverage = min(first_coverage, second_coverage)
    if mutual_coverage < coverage_threshold:
        return None

    score = mutual_coverage + 0.25 * (first_coverage + second_coverage)
    return {
        "score": float(score),
        "first_coverage": first_coverage,
        "second_coverage": second_coverage,
        "shared_boundary_vertices": shared_boundary_vertices,
    }


def infer_intersection_groups(
    additive_candidates,
    results,
    coverage_threshold=0.85,
):
    """Group ADD candidates whose visible surfaces imply set intersection.

    For an intersection ``A & B``, the visible surface contributed by ``A``
    lies inside ``B`` and vice versa, and both surfaces meet on a shared
    boundary.  A genuine volume overlap is also required, which prevents
    merely touching primitives from becoming intersections.  Groups with more
    than two candidates are constructed as cliques so a transitive chain
    cannot create an invalid n-ary intersection.
    """
    additive_candidates = list(additive_candidates)
    if len(additive_candidates) < 2 or results is None:
        return [[candidate] for candidate in additive_candidates]

    all_points = np.vstack(
        [np.asarray(result["mesh"].vertices, dtype=float) for result in results]
    )
    model_scale = max(float(np.linalg.norm(np.ptp(all_points, axis=0))), 1.0)
    tolerance = model_scale * 1e-4
    pair_scores = {}
    for first_index in range(len(additive_candidates)):
        for second_index in range(first_index + 1, len(additive_candidates)):
            relation = _intersection_pair_score(
                additive_candidates[first_index],
                additive_candidates[second_index],
                results,
                tolerance,
                coverage_threshold,
            )
            if relation is not None:
                pair_scores[(first_index, second_index)] = relation

    remaining = set(range(len(additive_candidates)))
    groups = []
    while remaining:
        available_pairs = [
            (relation["score"], first, second)
            for (first, second), relation in pair_scores.items()
            if first in remaining and second in remaining
        ]
        if not available_pairs:
            groups.extend([[additive_candidates[index]] for index in sorted(remaining)])
            break

        _, first, second = max(available_pairs)
        group_indices = [first, second]
        remaining.difference_update(group_indices)
        expandable = sorted(
            remaining,
            key=lambda index: sum(
                pair_scores.get(tuple(sorted((index, member))), {}).get("score", 0.0)
                for member in group_indices
            ),
            reverse=True,
        )
        for index in expandable:
            if all(tuple(sorted((index, member))) in pair_scores for member in group_indices):
                group_indices.append(index)
                remaining.remove(index)

        group = [additive_candidates[index] for index in group_indices]
        groups.append(group)
        for index in group_indices:
            candidate = additive_candidates[index]
            partners = [
                additive_candidates[other].patch_ids
                for other in group_indices
                if other != index
            ]
            candidate.metadata["intersection_partner_patch_ids"] = partners
    return groups


def _build_additive_ir(candidate_nodes, results, coverage_threshold):
    if not candidate_nodes:
        return {"op": "EMPTY", "children": []}
    transition_nodes = [
        item for item in candidate_nodes if item[0].metadata.get("transition")
    ]
    base_nodes = [
        item for item in candidate_nodes if not item[0].metadata.get("transition")
    ]
    if not base_nodes:
        return {"op": "EMPTY", "children": []}
    groups = infer_intersection_groups(
        [candidate for candidate, _ in base_nodes],
        results,
        coverage_threshold=coverage_threshold,
    )
    terms = []
    for group in groups:
        children = [
            next(node for candidate, node in base_nodes if candidate is member)
            for member in group
        ]
        terms.append(
            children[0]
            if len(children) == 1
            else {"op": "INTERSECTION", "children": children}
        )

    # A fillet/chamfer mother surface is a local transition, not an
    # independent additive solid. Intersect it with the neighboring base
    # primitive so only the clipped transition volume is retained.
    for transition, transition_node in transition_nodes:
        support_ids = set(transition.metadata.get("support_patch_ids", []))
        supporting = [
            (candidate, node)
            for candidate, node in base_nodes
            if support_ids.intersection(candidate.patch_ids)
        ]
        if not supporting:
            transition_bounds = candidate_module.primitive_bounds(transition)
            for candidate, node in base_nodes:
                bounds = candidate_module.primitive_bounds(candidate)
                overlap = np.minimum(bounds[1], transition_bounds[1]) - np.maximum(
                    bounds[0], transition_bounds[0]
                )
                if np.all(overlap > 0.0):
                    supporting.append((candidate, node))
        if not supporting:
            continue
        supporting.sort(
            key=lambda item: (
                -sum(patch_id in item[0].patch_ids for patch_id in support_ids),
                item[0].fitting_error,
            )
        )
        support_bounds = candidate_module.primitive_bounds(supporting[0][0])
        transition_bounds = candidate_module.primitive_bounds(transition)
        overlap = np.minimum(support_bounds[1], transition_bounds[1]) - np.maximum(
            support_bounds[0], transition_bounds[0]
        )
        if np.all(overlap > 0.0):
            terms.append(
                {
                    "op": "INTERSECTION",
                    "children": [supporting[0][1], transition_node],
                    "transition_patch_ids": list(transition.patch_ids),
                }
            )
        else:
            # The clipped mother surface does not overlap its support (for
            # example a rim band sitting exactly between two body
            # primitives); intersecting would erase it entirely, so keep it
            # as a standalone additive term.
            terms.append(transition_node)
    return terms[0] if len(terms) == 1 else {"op": "UNION", "children": terms}


def _stitch_pocket_cutters(candidates, nodes, model_scale):
    """Close gaps between a pocket floor cutter and its adjacent groove prisms.

    A groove profile can taper inward as it rises, while its central floor
    cutter has a fixed footprint. Extend only the inner side of selected
    adjacent subtractive prisms to the floor cutter's face, with a tiny
    overlap for robust Boolean evaluation. Their outer observed profiles
    remain unchanged.
    """
    overlap = max(1e-4 * model_scale, 1e-8)
    seam_tolerance = max(1e-3 * model_scale, 1e-7)
    for pocket in candidates:
        if not pocket.metadata.get("pocket_floor") or pocket.primitive_type != "CUBE":
            continue
        connected = {
            tuple(patch_ids) for patch_ids in pocket.metadata.get("connected_prisms", [])
        }
        if not connected:
            continue
        rotation = np.asarray(pocket.parameters["rotation"], dtype=float)
        normal_axis = int(np.argmax(np.abs(rotation[:, 2])))
        if abs(rotation[normal_axis, 2]) < 0.999:
            continue
        pocket_bounds = candidate_module.primitive_bounds(pocket)
        for prism, node in zip(candidates, nodes):
            if prism.primitive_type != "EXTRUSION" or tuple(prism.patch_ids) not in connected:
                continue
            parameters = prism.parameters
            axis = np.asarray(parameters["axis"], dtype=float)
            axial_axis = int(np.argmax(np.abs(axis)))
            if axial_axis == normal_axis or abs(axis[axial_axis]) < 0.999:
                continue
            radial_axis = next(index for index in range(3) if index not in {normal_axis, axial_axis})
            bounds = candidate_module.primitive_bounds(prism)
            if (
                bounds[1, normal_axis] < pocket_bounds[0, normal_axis]
                or bounds[0, normal_axis] > pocket_bounds[1, normal_axis]
                or bounds[1, axial_axis] < pocket_bounds[0, axial_axis]
                or bounds[0, axial_axis] > pocket_bounds[1, axial_axis]
            ):
                continue
            if abs(bounds[0, radial_axis] - pocket_bounds[1, radial_axis]) <= seam_tolerance:
                wall = pocket_bounds[1, radial_axis] - overlap
            elif abs(bounds[1, radial_axis] - pocket_bounds[0, radial_axis]) <= seam_tolerance:
                wall = pocket_bounds[0, radial_axis] + overlap
            else:
                continue
            origin = np.asarray(parameters["axis_point"], dtype=float)
            basis = np.asarray(parameters["basis"], dtype=float)
            polygon = np.asarray(parameters["polygon"], dtype=float)
            extra = []
            for station in bounds[:, normal_axis]:
                point = origin.copy()
                point[radial_axis] = wall
                point[normal_axis] = station
                extra.append((point - origin) @ basis)
            extended = np.vstack([polygon, extra])
            if np.max(np.linalg.norm(extended[-2:, None] - polygon[None, :], axis=2).min(axis=1)) > 0.1 * model_scale:
                continue
            hull = scipy.spatial.ConvexHull(extended)
            node["parameters"]["polygon"] = extended[hull.vertices].tolist()


def _overlap_coaxial_cutters(candidates, nodes, model_scale):
    """Give touching stepped-hole cutters a small shared axial interval."""
    overlap = max(1e-5 * model_scale, 1e-8)
    seam_tolerance = max(1e-4 * model_scale, 1e-7)
    cylinders = [
        (candidate, node)
        for candidate, node in zip(candidates, nodes)
        if candidate.primitive_type == "CYLINDER"
    ]
    for index, (first, first_node) in enumerate(cylinders):
        first_parameters = first_node["parameters"]
        first_axis = candidate_module._normalize(first_parameters["axis"])
        first_origin = np.asarray(first_parameters["axis_point"], dtype=float)
        if first_axis is None:
            continue
        for second, second_node in cylinders[index + 1 :]:
            second_parameters = second_node["parameters"]
            second_axis = candidate_module._normalize(second_parameters["axis"])
            second_origin = np.asarray(second_parameters["axis_point"], dtype=float)
            if (
                second_axis is None
                or abs(float(np.dot(first_axis, second_axis))) < np.cos(np.deg2rad(1.0))
                or np.linalg.norm(np.cross(second_origin - first_origin, first_axis))
                > seam_tolerance
            ):
                continue
            first_extent = np.asarray(first_parameters["extent"], dtype=float)
            second_extent = np.asarray(second_parameters["extent"], dtype=float)
            first_stations = (first_origin @ first_axis) + first_extent
            second_stations = (second_origin @ first_axis) + second_extent * np.dot(
                second_axis, first_axis
            )
            first_low, first_high = float(first_stations.min()), float(first_stations.max())
            second_low, second_high = float(second_stations.min()), float(second_stations.max())
            if abs(first_high - second_low) <= seam_tolerance:
                first_end = int(np.argmax(first_stations))
                second_end = int(np.argmin(second_stations))
                first_extent[first_end] += overlap
                second_extent[second_end] -= overlap / float(np.dot(second_axis, first_axis))
            elif abs(second_high - first_low) <= seam_tolerance:
                first_end = int(np.argmin(first_stations))
                second_end = int(np.argmax(second_stations))
                first_extent[first_end] -= overlap
                second_extent[second_end] += overlap / float(np.dot(second_axis, first_axis))
            else:
                continue
            first_parameters["extent"] = first_extent.tolist()
            second_parameters["extent"] = second_extent.tolist()


def _extend_through_hole_cutters(candidates, nodes, results, model_points, model_scale):
    """Carry observed through holes beyond both exterior end faces.

    Boolean tolerances or an extended additive body can otherwise leave a
    very thin cap over a hole even when the fitted hole wall spans the model.
    Only cylinders whose observed wall reaches both model extrema along the
    cylinder axis qualify; blind holes keep their fitted depth.
    """
    lookup = candidate_module._result_lookup(results)
    seam_tolerance = max(1e-4 * model_scale, 1e-7)
    overlap = max(5e-4 * model_scale, 1e-7)
    for candidate, node in zip(candidates, nodes):
        if candidate.primitive_type != "CYLINDER":
            continue
        axis = candidate_module._normalize(node["parameters"]["axis"])
        if axis is None:
            continue
        wall_patch_ids = set(candidate.patch_ids) - set(
            candidate.metadata.get("cap_patch_ids", [])
        )
        if not wall_patch_ids:
            continue
        wall_points = np.vstack(
            [np.asarray(lookup[patch_id]["mesh"].vertices, dtype=float) for patch_id in wall_patch_ids]
        )
        model_stations = model_points @ axis
        wall_stations = wall_points @ axis
        if (
            abs(float(wall_stations.min() - model_stations.min())) > seam_tolerance
            or abs(float(wall_stations.max() - model_stations.max())) > seam_tolerance
        ):
            continue
        origin = np.asarray(node["parameters"]["axis_point"], dtype=float)
        current = np.asarray(node["parameters"]["extent"], dtype=float)
        offset = float(origin @ axis)
        node["parameters"]["extent"] = [
            float(min(current.min(), model_stations.min() - offset - overlap)),
            float(max(current.max(), model_stations.max() - offset + overlap)),
        ]


def _extend_cutters_at_additive_mouths(
    candidates, nodes, additive_candidates, results, model_scale
):
    """Pass a local exterior face when a cutter wall ends at its opening.

    A stepped hole can open on a flange face well inside the model's global
    bounds.  The wall/face graph identifies that opening without extending a
    blind hole through its (subtractive) floor.
    """
    lookup = candidate_module._result_lookup(results)
    additive_patches = {
        patch_id for candidate in additive_candidates for patch_id in candidate.patch_ids
    }
    seam_tolerance = max(1e-4 * model_scale, 1e-7)
    overlap = max(5e-4 * model_scale, 1e-7)
    for candidate, node in zip(candidates, nodes):
        if candidate.primitive_type != "CYLINDER":
            continue
        axis = candidate_module._normalize(node["parameters"]["axis"])
        if axis is None:
            continue
        wall_ids = set(candidate.patch_ids) - set(
            candidate.metadata.get("cap_patch_ids", [])
        )
        neighboring = set()
        for patch_id in wall_ids:
            neighboring.update(lookup[patch_id].get("neighboring_patches", []))
        opening_ids = (neighboring & additive_patches) - set(candidate.patch_ids)
        if not opening_ids:
            continue
        parameters = node["parameters"]
        extent = np.asarray(parameters["extent"], dtype=float)
        offset = float(np.asarray(parameters["axis_point"], dtype=float) @ axis)
        stations = offset + extent
        for patch_id in opening_ids:
            face = lookup[patch_id]
            fit_rate, _ = candidate_module._seed_fit(face, "PLANE")
            if fit_rate < 0.8:
                continue
            mesh = face["mesh"]
            normal = candidate_module._normalize(
                np.average(mesh.face_normals, axis=0, weights=mesh.area_faces)
            )
            if normal is None:
                continue
            face_stations = np.asarray(mesh.vertices, dtype=float) @ axis
            if np.ptp(face_stations) > seam_tolerance:
                continue
            face_station = float(np.mean(face_stations))
            for end, direction in ((int(np.argmin(stations)), -1), (int(np.argmax(stations)), 1)):
                if (
                    np.dot(normal, axis) * direction > 0.95
                    and abs(face_station - stations[end]) <= seam_tolerance
                ):
                    extent[end] = face_station - offset + direction * overlap
        parameters["extent"] = extent.tolist()


def _apply_conical_end_constraints(additive, results, model_scale):
    """Trim a polygonal body with its fitted conical end surfaces.

    Adding a cone to a full prism cannot form an end chamfer: the prism fills
    the volume that the cone should remove. Extend the prism to the observed
    end faces, extend each cone inward until it contains the whole opposite
    end of the prism, then intersect these solids.
    """
    tolerance = max(1e-4 * model_scale, 1e-7)
    lookup = candidate_module._result_lookup(results)
    extrusions = [
        (index, item[0])
        for index, item in enumerate(additive)
        if item[0].primitive_type == "EXTRUSION"
    ]
    matches = {}
    for cone_index, (cone, _) in enumerate(additive):
        if cone.primitive_type != "CONE":
            continue
        cone_axis = candidate_module._normalize(cone.parameters["axis"])
        cone_apex = np.asarray(cone.parameters["apex"], dtype=float)
        points = np.vstack(
            [np.asarray(lookup[patch_id]["mesh"].vertices, dtype=float) for patch_id in cone.patch_ids]
        )
        for extrusion_index, extrusion in extrusions:
            axis = candidate_module._normalize(extrusion.parameters["axis"])
            if (
                axis is None
                or cone_axis is None
                or abs(float(np.dot(axis, cone_axis))) < np.cos(np.deg2rad(3.0))
            ):
                continue
            if not any(
                set(lookup[patch_id].get("neighboring_patches", []))
                .intersection(extrusion.patch_ids)
                for patch_id in cone.patch_ids
            ):
                continue
            origin = np.asarray(extrusion.parameters["axis_point"], dtype=float)
            if np.linalg.norm(np.cross(cone_apex - origin, axis)) > 0.02 * model_scale:
                continue
            basis = np.asarray(extrusion.parameters["basis"], dtype=float)
            polygon = np.asarray(extrusion.parameters["polygon"], dtype=float)
            local = (points - origin) @ basis
            inside = candidate_module._point_in_polygon(local, polygon)
            if np.any(~inside) and np.max(
                candidate_module._polygon_boundary_distance(local[~inside], polygon)
            ) > tolerance:
                continue
            matches.setdefault(extrusion_index, []).append((cone_index, points))
            break

    replacements = {}
    consumed = set()
    for extrusion_index, linked_cones in matches.items():
        extrusion, body = additive[extrusion_index]
        parameters = extrusion.parameters
        origin = np.asarray(parameters["axis_point"], dtype=float)
        axis = candidate_module._normalize(parameters["axis"])
        basis = np.asarray(parameters["basis"], dtype=float)
        polygon = np.asarray(parameters["polygon"], dtype=float)
        original_extent = np.asarray(body["parameters"]["extent"], dtype=float)
        accepted = []
        for cone_index, points in linked_cones:
            cone, _ = additive[cone_index]
            cone_axis = candidate_module._normalize(cone.parameters["axis"])
            apex = np.asarray(cone.parameters["apex"], dtype=float)
            trial_extent = original_extent.copy()
            station = (points - origin) @ axis
            trial_extent[0] = min(trial_extent[0], float(station.min()) - tolerance)
            trial_extent[1] = max(trial_extent[1], float(station.max()) + tolerance)
            body_ends = origin + trial_extent[:, None] * axis
            cone_stations = (body_ends - apex) @ cone_axis
            far_end = body_ends[int(np.argmax(cone_stations))]
            far_polygon = far_end + polygon @ basis.T
            radial = np.linalg.norm(np.cross(far_polygon - apex, cone_axis), axis=1)
            far_radius = float(cone_stations.max() * np.tan(cone.parameters["angle"]))
            if far_radius < float(radial.max()) - tolerance:
                continue
            accepted.append((cone_index, points))
        if not accepted:
            continue
        extent = original_extent.copy()
        for _, points in accepted:
            station = (points - origin) @ axis
            extent[0] = min(extent[0], float(station.min()) - tolerance)
            extent[1] = max(extent[1], float(station.max()) + tolerance)
        constraints = []
        for cone_index, _ in accepted:
            cone, cone_node = additive[cone_index]
            cone_axis = candidate_module._normalize(cone.parameters["axis"])
            apex = np.asarray(cone.parameters["apex"], dtype=float)
            cone_stations = (origin + extent[:, None] * axis - apex) @ cone_axis
            cone_extent = np.asarray(cone_node["parameters"]["extent"], dtype=float)
            cone_node["parameters"]["extent"] = [
                float(min(cone_extent.min(), cone_stations.min()) - tolerance),
                float(max(cone_extent.max(), cone_stations.max()) + tolerance),
            ]
            constraints.append(cone_node)
            consumed.add(cone_index)
        body["parameters"]["extent"] = extent.tolist()
        replacements[extrusion_index] = (
            extrusion,
            {"op": "INTERSECTION", "children": [body, *constraints]},
        )
    additive[:] = [
        replacements.get(index, item)
        for index, item in enumerate(additive)
        if index not in consumed
    ]


def _join_inward_torus_supports(additive, model_scale):
    """Move a tangent fillet's padding inside its supporting cylinder."""
    overlap = max(1e-4 * model_scale, 1e-8)
    cylinders = [item for item in additive if item[0].primitive_type == "CYLINDER"]
    for fillet, fillet_node in additive:
        if fillet.primitive_type != "INNER_TORUS":
            continue
        parameters = fillet_node["parameters"]
        center = np.asarray(parameters["center"], dtype=float)
        axis = candidate_module._normalize(parameters["axis"])
        extent = np.asarray(parameters["extent"], dtype=float)
        near = int(np.argmin(np.abs(extent)))
        far = 1 - near
        if abs(extent[near]) > 1.1 * overlap or abs(extent[far]) < 5.0 * overlap:
            continue
        inner_radius = float(parameters["major_radius"] - parameters["minor_radius"])
        for support, support_node in cylinders:
            support_parameters = support_node["parameters"]
            support_axis = candidate_module._normalize(support_parameters["axis"])
            support_origin = np.asarray(support_parameters["axis_point"], dtype=float)
            if (
                support_axis is None
                or abs(float(np.dot(axis, support_axis))) < np.cos(np.deg2rad(1.0))
                or abs(float(support_parameters["radius"]) - inner_radius) > overlap
                or np.linalg.norm(np.cross(center - support_origin, support_axis)) > overlap
            ):
                continue
            support_extent = np.asarray(support_parameters["extent"], dtype=float)
            support_ends = support_origin + support_extent[:, None] * support_axis
            touching = int(np.argmin(np.linalg.norm(support_ends - center, axis=1)))
            if np.linalg.norm(support_ends[touching] - center) > overlap:
                continue
            to_fillet = (center + extent[far] * axis) - support_ends[touching]
            to_support = support_ends[1 - touching] - support_ends[touching]
            if float(np.dot(to_fillet, to_support)) >= 0.0:
                continue
            direction = np.sign(float(np.dot(to_fillet, support_axis)))
            extent[near] = 0.0
            support_extent[touching] += direction * overlap
            parameters["extent"] = extent.tolist()
            support_parameters["extent"] = support_extent.tolist()
            break


def _bridge_additive_cone_seams(additive, model_scale):
    """Overlap tangent cone/cylinder solids without adding an outer lip."""
    overlap = max(1e-5 * model_scale, 1e-8)
    seam_tolerance = max(1e-4 * model_scale, 1e-7)
    cylinders = [item for item in additive if item[0].primitive_type == "CYLINDER"]
    for index, (cone, cone_node) in enumerate(additive):
        if cone.primitive_type != "CONE" or cone_node["op"] != "CONE":
            continue
        parameters = cone_node["parameters"]
        cone_axis = candidate_module._normalize(parameters["axis"])
        apex = np.asarray(parameters["apex"], dtype=float)
        extent = np.asarray(parameters["extent"], dtype=float)
        cone_ends = apex + extent[:, None] * cone_axis
        radii = np.maximum(0.0, extent) * np.tan(parameters["angle"])
        for support, support_node in cylinders:
            support_parameters = support_node["parameters"]
            support_axis = candidate_module._normalize(support_parameters["axis"])
            support_origin = np.asarray(support_parameters["axis_point"], dtype=float)
            if (
                support_axis is None
                or abs(float(np.dot(cone_axis, support_axis))) < np.cos(np.deg2rad(1.0))
                or np.linalg.norm(np.cross(apex - support_origin, support_axis))
                > seam_tolerance
            ):
                continue
            support_radius = float(support_parameters["radius"])
            support_extent = np.asarray(support_parameters["extent"], dtype=float)
            support_ends = support_origin + support_extent[:, None] * support_axis
            distances = np.linalg.norm(cone_ends[:, None] - support_ends[None, :], axis=2)
            cone_end, support_end = np.unravel_index(np.argmin(distances), distances.shape)
            if (
                distances[cone_end, support_end] > seam_tolerance
                or abs(float(radii[cone_end]) - support_radius) > seam_tolerance
                or radii[1 - cone_end] > support_radius + seam_tolerance
            ):
                continue
            to_support = support_ends[1 - support_end] - cone_ends[cone_end]
            to_cone = cone_ends[1 - cone_end] - cone_ends[cone_end]
            if float(np.dot(to_support, to_cone)) >= 0.0:
                continue
            direction = np.sign(float(np.dot(to_support, cone_axis)))
            extended = extent.copy()
            extended[cone_end] += direction * overlap
            parameters["extent"] = extended.tolist()
            radial_clip = {
                "op": "CYLINDER",
                "parameters": {
                    "axis_point": apex.tolist(),
                    "axis": cone_axis.tolist(),
                    "extent": extended.tolist(),
                    "radius": support_radius,
                },
            }
            additive[index] = (
                cone,
                {"op": "INTERSECTION", "children": [cone_node, radial_clip]},
            )
            break


def _concave_cylinder_fillet_supports(additive, cutters, results, model_scale):
    """Fill the tangent wedge between a round boss and a planar body.

    The selected inward cylinder remains the cutter.  This adds only the
    missing local mother volume below its observed arc; patch growing and
    candidate selection are unchanged.
    """
    lookup = candidate_module._result_lookup(results)
    tolerance = max(5e-4 * model_scale, 1e-7)
    join_overlap = max(1e-4 * model_scale, 1e-8)
    supports = []
    convex_cylinders = [
        candidate for candidate, _ in additive if candidate.primitive_type == "CYLINDER"
    ]
    planar_bodies = [
        candidate for candidate, _ in additive if candidate.primitive_type == "CUBE"
    ]
    for cutter in cutters:
        if (
            cutter.primitive_type != "CYLINDER"
            or cutter.metadata.get("orientation_score", 0) >= -0.8
        ):
            continue
        wall_ids = set(cutter.patch_ids) - set(cutter.metadata.get("cap_patch_ids", []))
        neighbors = set()
        for patch_id in wall_ids:
            neighbors.update(lookup[patch_id].get("neighboring_patches", []))
        if not neighbors:
            continue
        inward_axis = candidate_module._normalize(cutter.parameters["axis"])
        inward_origin = np.asarray(cutter.parameters["axis_point"], dtype=float)
        inward_radius = float(cutter.parameters["radius"])
        if inward_axis is None or inward_radius <= tolerance:
            continue
        wall_points, _ = candidate_module._sample_patch_points(
            results, list(wall_ids), 1000
        )
        matches = []
        for convex in convex_cylinders:
            if not neighbors.intersection(convex.patch_ids):
                continue
            convex_axis = candidate_module._normalize(convex.parameters["axis"])
            if (
                convex_axis is None
                or abs(float(np.dot(convex_axis, inward_axis)))
                < np.cos(np.deg2rad(1.0))
            ):
                continue
            convex_origin = np.asarray(convex.parameters["axis_point"], dtype=float)
            convex_radius = float(convex.parameters["radius"])
            center_delta = inward_origin - convex_origin
            center_delta -= float(center_delta @ convex_axis) * convex_axis
            center_distance = float(np.linalg.norm(center_delta))
            tangent_error = abs(center_distance - convex_radius - inward_radius)
            if tangent_error > tolerance:
                continue
            for body in planar_bodies:
                for patch_id in neighbors.intersection(body.patch_ids):
                    face = lookup[patch_id]
                    fit_rate, _ = candidate_module._seed_fit(face, "PLANE")
                    if fit_rate < 0.8:
                        continue
                    mesh = face["mesh"]
                    normal = candidate_module._normalize(
                        np.average(mesh.face_normals, axis=0, weights=mesh.area_faces)
                    )
                    if normal is None or abs(float(normal @ convex_axis)) > 0.02:
                        continue
                    face_center = np.mean(np.asarray(mesh.vertices, dtype=float), axis=0)
                    plane_from_convex = float((face_center - convex_origin) @ normal)
                    cutter_height = float(center_delta @ normal)
                    if (
                        abs(plane_from_convex) > tolerance
                        or abs(cutter_height - inward_radius) > tolerance
                    ):
                        continue
                    axis = convex_axis.copy()
                    normal = candidate_module._normalize(
                        normal - float(normal @ axis) * axis
                    )
                    radial = np.cross(normal, axis)
                    if float(center_delta @ radial) < 0:
                        axis = -axis
                        radial = -radial
                    tangent = center_delta * (
                        convex_radius / (convex_radius + inward_radius)
                    )
                    radial_low = float(tangent @ radial) - join_overlap
                    radial_high = float(center_delta @ radial)
                    normal_low = plane_from_convex - join_overlap
                    normal_high = cutter_height
                    if (
                        radial_high - radial_low <= tolerance
                        or normal_high - normal_low <= tolerance
                    ):
                        continue
                    convex_ends = (
                        convex_origin
                        + np.asarray(convex.parameters["extent"])[:, None] * convex_axis
                    )
                    inward_ends = (
                        inward_origin
                        + np.asarray(cutter.parameters["extent"])[:, None] * inward_axis
                    )
                    face_stations = np.asarray(mesh.vertices, dtype=float) @ axis
                    axial_low = max(
                        float((convex_ends @ axis).min()),
                        float((inward_ends @ axis).min()),
                        float(face_stations.min()),
                    )
                    axial_high = min(
                        float((convex_ends @ axis).max()),
                        float((inward_ends @ axis).max()),
                        float(face_stations.max()),
                    )
                    if axial_high - axial_low <= 2 * tolerance:
                        continue
                    center = (
                        convex_origin
                        + ((axial_low + axial_high) / 2 - float(convex_origin @ axis)) * axis
                        + (radial_low + radial_high) / 2 * radial
                        + (normal_low + normal_high) / 2 * normal
                    )
                    rotation = np.column_stack((axis, radial, normal))
                    support = {
                        "op": "CUBE",
                        "parameters": {
                            "center": center.tolist(),
                            "size": [
                                axial_high - axial_low,
                                radial_high - radial_low,
                                normal_high - normal_low,
                            ],
                            "rotation": rotation.tolist(),
                            "euler_xyz_degrees": candidate_module._matrix_to_euler_xyz(
                                rotation
                            ).tolist(),
                        },
                        "patch_ids": [],
                        "feature": "concave_cylinder_fillet_support",
                        "source_patch_ids": list(cutter.patch_ids),
                    }
                    contained = candidate_module.primitive_contains(
                        "CUBE", support["parameters"], wall_points,
                        tolerance=tolerance,
                    )
                    if float(np.mean(contained)) < 0.9:
                        continue
                    score = (
                        tangent_error + abs(plane_from_convex)
                        + abs(cutter_height - inward_radius)
                    )
                    matches.append((score, support))
        if matches:
            supports.append(min(matches, key=lambda item: item[0])[1])
    return supports


def build_csg_ir(
    selected_candidates,
    results=None,
    intersection_coverage_threshold=0.85,
):
    """Build UNION/INTERSECTION/DIFFERENCE IR from selected candidates."""
    additive_candidates = [
        candidate for candidate in selected_candidates if candidate.operation == "ADD"
    ]
    additive = [
        (candidate, _candidate_to_ir(candidate)) for candidate in additive_candidates
    ]
    subtractive_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate.operation == "SUBTRACT"
    ]
    subtractive = [_candidate_to_ir(candidate) for candidate in subtractive_candidates]
    if not additive:
        return {"op": "EMPTY", "children": []}
    restorative = []
    if results is not None and subtractive:
        retained_additive = []
        for candidate, node in additive:
            # Restoration coverage is measured on the primitive's own curved
            # surface; attached flat cap patches belong to the surrounding
            # body and would dilute the ratio.
            surface_patch_ids = sorted(
                set(candidate.patch_ids)
                - set(candidate.metadata.get("cap_patch_ids", []))
            )
            if not surface_patch_ids:
                surface_patch_ids = list(candidate.patch_ids)
            points, _ = candidate_module._sample_patch_points(
                results, surface_patch_ids, 2500
            )
            covered = np.zeros(len(points), dtype=bool)
            for cutter in subtractive_candidates:
                covered |= candidate_module.primitive_contains(
                    cutter.primitive_type,
                    cutter.parameters,
                    points,
                    tolerance=1e-4,
                )
            coverage = float(np.mean(covered)) if len(points) else 0.0
            partial_cylinder_restoration = False
            restoration_cutter = None
            restoration_cutter_index = None
            if coverage >= 0.5 and candidate.primitive_type == "CYLINDER":
                candidate_radius = float(candidate.parameters.get("radius", 0.0))
                candidate_axis = candidate_module._normalize(candidate.parameters.get("axis", []))
                for cutter_index, cutter in enumerate(subtractive_candidates):
                    if cutter.primitive_type != "CYLINDER":
                        continue
                    cutter_radius = float(cutter.parameters.get("radius", 0.0))
                    cutter_axis = candidate_module._normalize(cutter.parameters.get("axis", []))
                    if candidate_axis is None or cutter_axis is None:
                        continue
                    if (
                        cutter_radius <= candidate_radius
                        or abs(np.dot(candidate_axis, cutter_axis)) < np.cos(np.deg2rad(8.0))
                    ):
                        continue
                    cutter_covered = candidate_module.primitive_contains(
                        cutter.primitive_type,
                        cutter.parameters,
                        points,
                        tolerance=1e-4,
                    )
                    if float(np.mean(cutter_covered)) >= 0.5:
                        partial_cylinder_restoration = True
                        restoration_cutter = cutter
                        restoration_cutter_index = cutter_index
                        break
            if partial_cylinder_restoration and coverage < 0.9:
                candidate_axis = candidate_module._normalize(candidate.parameters["axis"])
                cutter_axis = candidate_module._normalize(restoration_cutter.parameters["axis"])
                candidate_origin = np.asarray(
                    candidate.parameters["axis_point"], dtype=float
                )
                candidate_extent = np.asarray(
                    candidate.parameters["extent"], dtype=float
                )
                candidate_endpoints = (
                    candidate_origin + candidate_extent[:, None] * candidate_axis
                )
                cutter_origin = np.asarray(
                    restoration_cutter.parameters["axis_point"], dtype=float
                )
                projected_extent = (
                    candidate_endpoints - cutter_origin
                ) @ cutter_axis
                subtractive[restoration_cutter_index]["parameters"]["extent"] = [
                    float(projected_extent.min()),
                    float(projected_extent.max()),
                ]
            if coverage >= 0.9 or partial_cylinder_restoration:
                restorative.append(node)
            else:
                retained_additive.append((candidate, node))
        additive = retained_additive

    if not additive:
        return {"op": "EMPTY", "children": []}
    if results is not None:
        points = np.vstack([np.asarray(result["mesh"].vertices, dtype=float) for result in results])
        model_scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), candidate_module.EPSILON)
        _apply_conical_end_constraints(additive, results, model_scale)
        _join_inward_torus_supports(additive, model_scale)
        _bridge_additive_cone_seams(additive, model_scale)
    base = _build_additive_ir(
        additive,
        results,
        coverage_threshold=intersection_coverage_threshold,
    )
    if results is not None and subtractive:
        supports = _concave_cylinder_fillet_supports(
            additive, subtractive_candidates, results, model_scale
        )
        if supports:
            base = {"op": "UNION", "children": [base, *supports]}
    if not subtractive:
        return (
            base
            if not restorative
            else {"op": "UNION", "children": [base, *restorative]}
        )
    if results is not None:
        _stitch_pocket_cutters(subtractive_candidates, subtractive, model_scale)
        _overlap_coaxial_cutters(subtractive_candidates, subtractive, model_scale)
        _extend_through_hole_cutters(
            subtractive_candidates, subtractive, results, points, model_scale
        )
        _extend_cutters_at_additive_mouths(
            subtractive_candidates, subtractive, additive_candidates, results, model_scale
        )
    subtraction = (
        subtractive[0]
        if len(subtractive) == 1
        else {"op": "UNION", "children": subtractive}
    )
    difference = {"op": "DIFFERENCE", "children": [base, subtraction]}
    if restorative:
        return {"op": "UNION", "children": [difference, *restorative]}
    return difference


class BooleanCSGReconstructor:
    """Classify operations, infer intersections, and search for a CSG tree."""

    def __init__(
        self,
        containment_tolerance=1e-4,
        beam_width=12,
        maximum_primitives=64,
        complexity_penalty=0.025,
        intersection_coverage_threshold=0.85,
    ):
        self.containment_tolerance = float(containment_tolerance)
        self.beam_width = int(beam_width)
        self.maximum_primitives = int(maximum_primitives)
        self.complexity_penalty = float(complexity_penalty)
        self.intersection_coverage_threshold = float(
            intersection_coverage_threshold
        )

    def classify(self, primitive_candidates, results):
        return classify_boolean_operations(
            primitive_candidates,
            results,
            containment_tolerance=self.containment_tolerance,
        )

    @staticmethod
    def hypotheses(primitive_candidates, relative_score_threshold=0.35):
        return generate_boolean_hypotheses(
            primitive_candidates,
            relative_score_threshold=relative_score_threshold,
        )

    def reconstruct(self, primitive_candidates, results):
        return reconstruct_csg_tree(
            primitive_candidates,
            results,
            beam_width=self.beam_width,
            maximum_primitives=self.maximum_primitives,
            complexity_penalty=self.complexity_penalty,
            intersection_coverage_threshold=self.intersection_coverage_threshold,
        )

    @staticmethod
    def build_ir(
        selected_candidates,
        results=None,
        intersection_coverage_threshold=0.85,
    ):
        return build_csg_ir(
            selected_candidates,
            results=results,
            intersection_coverage_threshold=intersection_coverage_threshold,
        )


__all__ = [
    "BooleanCSGReconstructor",
    "build_csg_ir",
    "classify_boolean_operations",
    "generate_boolean_hypotheses",
    "infer_intersection_groups",
    "reconstruct_csg_tree",
]
