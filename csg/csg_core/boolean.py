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
    base = _build_additive_ir(
        additive,
        results,
        coverage_threshold=intersection_coverage_threshold,
    )
    if not subtractive:
        return (
            base
            if not restorative
            else {"op": "UNION", "children": [base, *restorative]}
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
