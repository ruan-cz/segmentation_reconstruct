from pathlib import Path

import numpy as np
import trimesh

from csg.graph_construct import prepare_patch_graph
from csg.csg_reconstruction import (
    BooleanCSGReconstructor,
    CSGValidator,
    OpenSCADExporter,
    PrimitiveCandidate,
    PrimitiveCandidateGenerator,
    PrimitiveFitter,
    build_csg_ir,
    classify_boolean_operations,
    csg_contains,
    csg_to_openscad,
    fit_cone,
    fit_cylinder,
    fit_sphere,
    fit_torus,
    generate_extrusion_candidates,
    generate_cube_candidates,
    generate_primitive_candidates,
    generate_spline_extrusion_candidates,
    primitive_contains,
    reconstruct_csg_tree,
)


def _rotated_box_patches(rotated=True):
    box = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    z_angle = np.deg2rad(27.0)
    x_angle = 0.3
    z_rotation = np.array(
        [
            [np.cos(z_angle), -np.sin(z_angle), 0.0],
            [np.sin(z_angle), np.cos(z_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    x_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x_angle), -np.sin(x_angle)],
            [0.0, np.sin(x_angle), np.cos(x_angle)],
        ]
    )
    if rotated:
        box.vertices = box.vertices @ (z_rotation @ x_rotation).T
    box.vertices += np.array([3.0, -2.0, 5.0])
    groups = []
    for face_id, normal in enumerate(box.face_normals):
        for group in groups:
            if np.dot(group[0], normal) > 0.99:
                group[1].append(face_id)
                break
        else:
            groups.append([normal.copy(), [face_id]])
    results = []
    for patch_id, (normal, face_ids) in enumerate(groups):
        faces = box.faces[face_ids]
        vertex_ids, inverse = np.unique(faces, return_inverse=True)
        mesh = trimesh.Trimesh(
            vertices=box.vertices[vertex_ids],
            faces=inverse.reshape((-1, 3)),
            process=False,
        )
        plane = np.concatenate([normal, [-np.dot(normal, mesh.vertices.mean(axis=0))]])
        mesh.face_normals = np.tile(
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
            (len(mesh.faces), 1),
        )
        results.append(
            {
                "patch_index": patch_id,
                "mesh": mesh,
                "area": mesh.area,
                "params": plane,
                "fits": {"plane": {"fit_rate": 1.0, "params": plane}},
            }
        )
    edges = []
    for first in range(6):
        for second in range(first + 1, 6):
            if abs(np.dot(groups[first][0], groups[second][0])) < 0.1:
                edges.append(
                    {
                        "patch0": first,
                        "patch1": second,
                        "length": 1.0,
                        "dihedral_angle": np.pi / 2.0,
                    }
                )
    graph = {"nodes": [{"patch_index": index} for index in range(6)], "edges": edges}
    return results, graph


def test_rotated_cube_reconstruction():
    results, graph = _rotated_box_patches()
    candidates = generate_cube_candidates(results, graph)
    complete = [candidate for candidate in candidates if len(candidate.patch_ids) == 6]
    assert complete
    candidate = min(complete, key=lambda item: item.fitting_error)
    assert not candidate.metadata["axis_aligned"]
    assert np.allclose(np.sort(candidate.parameters["size"]), [2.0, 3.0, 4.0], atol=1e-6)
    assert candidate.fitting_error < 1e-6


def test_axis_aligned_cube_reuses_existing_subgraph():
    results, graph = _rotated_box_patches(rotated=False)
    candidates = generate_cube_candidates(
        results,
        graph,
    )
    complete = [candidate for candidate in candidates if len(candidate.patch_ids) == 6]
    assert complete
    candidate = min(complete, key=lambda item: item.fitting_error)
    assert candidate.metadata["axis_aligned"]
    assert np.allclose(np.sort(candidate.parameters["size"]), [2.0, 3.0, 4.0], atol=1e-6)
    assert candidate.fitting_error < 1e-6


def test_axis_aligned_multi_cube_partition_regression():
    ply_path = Path(
        "example_data/final/ply/"
        "00140750_0da4d13f288d4cd70deecf20_trimesh_001/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)

    candidates = generate_cube_candidates(results, graph)
    partition = {frozenset(candidate.patch_ids) for candidate in candidates}
    # The four disjoint winning regions must all be present; additional
    # losing-trial alternatives may also be emitted for the global search.
    expected = {
        frozenset([1, 2, 10, 19, 20]),
        frozenset([4, 5, 7, 15, 16, 17]),
        frozenset([0, 3, 9, 12]),
        frozenset([6, 8, 11, 13, 14, 18]),
    }

    assert expected <= partition


def test_00023435_classifies_inner_cube_as_subtract():
    ply_path = Path(
        "example_data/final/ply/"
        "00023435_385b221a0d58490b84f3edee_trimesh_000/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_cube_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    inner_patch_ids = frozenset([10, 16, 18, 24])
    inner = next(
        candidate
        for candidate in candidates
        if frozenset(candidate.patch_ids) == inner_patch_ids
    )
    assert inner.operation == "SUBTRACT"
    assert inner.metadata["orientation_score"] < -0.9
    assert inner.metadata["containment_count"] >= 1

    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)
    assert csg_ir["op"] == "DIFFERENCE"
    assert report["patch_coverage"] > 0.98
    assert sum(candidate.operation == "SUBTRACT" for candidate in selected) == 1


def test_00024448_rejects_degenerate_large_spheres():
    ply_path = Path(
        "example_data/final/ply/"
        "00024448_274a7c9d6def4a699961a747_trimesh_002/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)

    assert not [
        candidate
        for candidate in candidates
        if candidate.primitive_type == "SPHERE"
    ]

    classify_boolean_operations(candidates, results)
    csg_ir, selected, _ = reconstruct_csg_tree(candidates, results)
    assert csg_ir["op"] == "UNION"
    assert all(candidate.primitive_type == "CUBE" for candidate in selected)


def test_00025710_keeps_both_subtractive_cylinders():
    ply_path = Path(
        "example_data/final/ply/"
        "00025710_6fe81cf35e2740b3bbde9aa6_trimesh_020/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    cylinders = [
        candidate
        for candidate in candidates
        if candidate.primitive_type == "CYLINDER"
    ]
    assert len(cylinders) == 2
    assert all(candidate.operation == "SUBTRACT" for candidate in cylinders)

    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)
    selected_cylinders = [
        candidate
        for candidate in selected
        if candidate.primitive_type == "CYLINDER"
    ]
    assert len(selected_cylinders) == 2
    assert all(candidate.operation == "SUBTRACT" for candidate in selected_cylinders)
    assert csg_ir["op"] == "DIFFERENCE"
    assert report["patch_coverage"] > 0.999


def test_00023471_classifies_cylinder_inside_additive_union_as_subtract():
    ply_path = Path(
        "example_data/final/ply/"
        "00023471_7f45ff9e8c754def8bb4b1cb_trimesh_000/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    cutter = next(
        candidate
        for candidate in candidates
        if candidate.primitive_type == "CYLINDER"
        and candidate.patch_ids == [4]
    )
    assert cutter.operation == "SUBTRACT"
    assert cutter.metadata["containment_coverage"] >= 0.9

    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)
    assert csg_ir["op"] == "DIFFERENCE"
    assert any(
        candidate.primitive_type == "CYLINDER"
        and candidate.patch_ids == [4]
        and candidate.operation == "SUBTRACT"
        for candidate in selected
    )
    assert report["patch_coverage"] > 0.6


def test_00023792_preserves_additive_inner_cylinders():
    ply_path = Path(
        "example_data/final/ply/"
        "00023792_30c31f050d2c40139e9b36ca_trimesh_001/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    # Curved side patches 1/2/4/7/8 identify the five cylinders; flat end
    # caps attached to them must not change their Boolean operations.
    key_patches = {1, 2, 4, 7, 8}

    def keyed_operations(items):
        result = {}
        for candidate in items:
            if candidate.primitive_type != "CYLINDER":
                continue
            keys = [
                patch_id
                for patch_id in candidate.patch_ids
                if patch_id in key_patches
            ]
            if len(keys) == 1:
                result[keys[0]] = candidate.operation
        return result

    expected_operations = {
        8: "ADD",
        2: "SUBTRACT",
        7: "SUBTRACT",
        1: "ADD",
        4: "ADD",
    }
    assert keyed_operations(candidates) == expected_operations

    csg_ir, selected, _ = reconstruct_csg_tree(candidates, results)
    assert csg_ir["op"] == "UNION"
    assert csg_ir["children"][0]["op"] == "DIFFERENCE"
    assert keyed_operations(selected) == expected_operations


def test_00025611_groups_disconnected_sphere_and_preserves_torus_csg():
    ply_path = Path(
        "example_data/final/ply/"
        "00025611_37f3a717a9b842cfbe5f6005_trimesh_002/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    sphere = next(
        candidate
        for candidate in candidates
        if candidate.primitive_type == "SPHERE"
    )
    torus = next(
        candidate
        for candidate in candidates
        if candidate.primitive_type == "TORUS"
    )

    assert sphere.patch_ids == [0, 12]
    assert torus.patch_ids == [3, 9]
    assert torus.operation == "SUBTRACT"

    # Flat end caps attached to the cylinders must not change their Boolean
    # operations: the (2, 11) hole stays SUBTRACT, the post on 10 stays ADD.
    subtractive_cylinder = next(
        candidate
        for candidate in candidates
        if candidate.primitive_type == "CYLINDER"
        and {2, 11}.issubset(candidate.patch_ids)
    )
    additive_cylinder = next(
        candidate
        for candidate in candidates
        if candidate.primitive_type == "CYLINDER"
        and 10 in candidate.patch_ids
        and not {2, 11}.issubset(candidate.patch_ids)
    )
    assert subtractive_cylinder.operation == "SUBTRACT"
    assert additive_cylinder.operation == "ADD"

    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)
    selected_kinds = set()
    for candidate in selected:
        if candidate.primitive_type == "CYLINDER":
            operation = (
                "SUBTRACT"
                if {2, 11}.issubset(candidate.patch_ids)
                else "ADD"
            )
            selected_kinds.add((candidate.primitive_type, operation))
        else:
            selected_kinds.add((candidate.primitive_type, candidate.operation))
    assert selected_kinds == {
        ("SPHERE", "ADD"),
        ("TORUS", "SUBTRACT"),
        ("CYLINDER", "SUBTRACT"),
        ("CYLINDER", "ADD"),
    }
    assert csg_ir["op"] == "UNION"
    assert csg_ir["children"][0]["op"] == "DIFFERENCE"
    assert csg_ir["children"][0]["children"][0]["op"] == "SPHERE"
    subtraction = csg_ir["children"][0]["children"][1]
    subtractive_cylinder = next(
        child for child in subtraction["children"] if child["op"] == "CYLINDER"
    )
    additive_cylinder = csg_ir["children"][1]
    subtractive_height = np.ptp(
        subtractive_cylinder["parameters"]["extent"]
    )
    additive_height = np.ptp(additive_cylinder["parameters"]["extent"])
    assert np.isclose(subtractive_height, additive_height, rtol=1e-5)
    assert report["patch_coverage"] > 0.98


def test_00023582_selects_polygonal_extrusion_candidate():
    ply_path = Path(
        "example_data/final/ply/"
        "00023582_9c917172a61b472fb0e6ae3c_trimesh_002/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    extrusion = next(
        candidate
        for candidate in candidates
        if candidate.primitive_type == "EXTRUSION"
    )
    # The hexagonal socket cut into the top flange is recovered as a clean
    # polygonal prism and classified SUBTRACT from its reversed normals and
    # full containment in the spindle body.
    assert len(extrusion.metadata["side_patch_ids"]) >= 3
    assert set(extrusion.patch_ids).issuperset(extrusion.metadata["side_patch_ids"])
    assert extrusion.operation == "SUBTRACT"
    assert len(extrusion.parameters["polygon"]) >= 3

    csg_ir, selected, _ = reconstruct_csg_tree(candidates, results)
    assert any(
        candidate.primitive_type == "EXTRUSION"
        and candidate.patch_ids == extrusion.patch_ids
        for candidate in selected
    )
    assert "linear_extrude" in csg_to_openscad(csg_ir)


def test_00024792_screw_threads_and_cross_recess():
    ply_path = Path(
        "example_data/final/ply/"
        "00024792_34a17822747a4b20a8c2954b_trimesh_009/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)

    # The cross recess in the domed nut is removed by four triangular
    # prisms, one per arm, plus the center pocket cube enclosed by them.
    subtractive_extrusions = {
        tuple(candidate.patch_ids)
        for candidate in selected
        if candidate.primitive_type == "EXTRUSION"
        and candidate.operation == "SUBTRACT"
    }
    assert {
        (3, 5, 63),
        (13, 24, 50),
        (16, 22, 31),
        (40, 42, 59),
    }.issubset(subtractive_extrusions)
    pocket = next(
        (candidate for candidate in selected if candidate.metadata.get("pocket_floor")),
        None,
    )
    assert pocket is not None and pocket.operation == "SUBTRACT"
    assert pocket.patch_ids == [58]

    # The thread is a stack of thin cone/cylinder rings and the domed nut
    # top is a clipped sphere.
    kinds = {(candidate.primitive_type, candidate.operation) for candidate in selected}
    assert ("SPHERE", "ADD") in kinds
    assert sum(
        candidate.primitive_type == "CONE" and candidate.operation == "ADD"
        for candidate in selected
    ) >= 20
    assert sum(
        candidate.primitive_type == "CYLINDER" and candidate.operation == "ADD"
        for candidate in selected
    ) >= 10
    assert report["patch_coverage"] > 0.99
    assert "difference()" in csg_to_openscad(csg_ir)

    # The domed nut stays round; the recess center column opens through the
    # dome top, while the diagonal regions between groove arms stay solid.
    probes = np.array(
        [
            [0.005, 0.005, 0.30],  # recess center column -> void
            [0.0, 0.0, 0.36],      # center opens through the dome top -> void
            [0.06, 0.06, 0.36],    # between groove arms -> solid
            [0.15, 0.0, 0.25],     # nut wall -> solid
        ]
    )
    assert list(csg_contains(csg_ir, probes)) == [False, False, True, True]


def test_00140553_recovers_beam_between_rotated_arms():
    ply_path = Path(
        "example_data/final/ply/00140553_e2c0841b6c86e3bfdcc8c477_trimesh_000/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)
    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)

    cubes = [
        candidate
        for candidate in selected
        if candidate.primitive_type == "CUBE" and candidate.operation == "ADD"
    ]
    # post + horizontal beam + two rotated arms
    assert len(cubes) == 4
    patch_sets = {frozenset(candidate.patch_ids) for candidate in cubes}
    assert frozenset({5, 7, 9, 14, 19}) in patch_sets
    assert frozenset({16, 17, 18}) in patch_sets
    assert frozenset({1, 8, 11, 20, 21}) in patch_sets
    assert frozenset({0, 10, 12, 13, 15}) in patch_sets
    assert report["patch_coverage"] > 0.9

    # The arm plates end flush with the beam's bottom plane (z = 152.4):
    # both arm cubes must be clipped there instead of leaking below it.
    arms = [
        candidate
        for candidate in cubes
        if frozenset(candidate.patch_ids)
        in {frozenset({1, 8, 11, 20, 21}), frozenset({0, 10, 12, 13, 15})}
    ]
    for arm in arms:
        clip_bounds = arm.parameters.get("clip_bounds")
        assert clip_bounds is not None
        assert np.isclose(clip_bounds[0][2], 152.4, atol=0.5)
    probes = np.array(
        [
            [-66.5, 15.0, 150.5],  # below the beam bottom inside the unclipped box
            [66.5, 15.0, 150.5],
            [-81.0, 15.0, 185.0],  # arm interiors stay solid
            [81.0, 15.0, 185.0],
        ]
    )
    assert list(csg_contains(csg_ir, probes)) == [False, False, True, True]


def test_00025728_cylinder_with_chord_notch():
    ply_path = Path(
        "example_data/final/ply/"
        "00025728_6fe81cf35e2740b3bbde9aa6_trimesh_038/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    # The chord notch milled into the top cylinder (patch 1 is the flat
    # chord face) must become a subtractive box cutter.
    notch = next(
        candidate
        for candidate in candidates
        if candidate.primitive_type == "CUBE"
        and candidate.metadata.get("notch_of_cylinder")
    )
    assert notch.operation == "SUBTRACT"

    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)
    assert any(
        candidate.metadata.get("notch_of_cylinder")
        and candidate.operation == "SUBTRACT"
        for candidate in selected
    )
    # The removed chord segment is open while the rest of the cylinder
    # and the material below the notch floor stay solid.
    points = np.array(
        [
            [0.0, 0.030, 0.20],
            [0.0, 0.030, 0.05],
            [0.0, 0.010, 0.20],
        ]
    )
    assert list(csg_contains(csg_ir, points)) == [False, True, True]


def test_00025183_cube_with_bosses_and_through_slot():
    ply_path = Path(
        "example_data/final/ply/"
        "00025183_021c990acfc648618a49bd47_trimesh_000/gt.ply"
    )
    _, _, results, graph = prepare_patch_graph(ply_path)
    candidates = generate_primitive_candidates(results, graph)
    classify_boolean_operations(candidates, results)

    csg_ir, selected, report = reconstruct_csg_tree(candidates, results)
    selected_kinds = {
        (candidate.primitive_type, candidate.operation) for candidate in selected
    }
    assert ("CUBE", "ADD") in selected_kinds
    assert ("EXTRUSION", "ADD") in selected_kinds
    assert ("EXTRUSION", "SUBTRACT") in selected_kinds
    assert ("SPLINE_EXTRUSION", "ADD") in selected_kinds
    assert report["patch_coverage"] > 0.99

    openscad = csg_to_openscad(csg_ir)
    assert "difference()" in openscad
    assert openscad.count("linear_extrude") >= 3


def test_free_curve_spline_extrusion_candidate_parameters():
    sample_count = 48
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    radii = 1.0 + 0.22 * np.cos(3.0 * angles)
    profile = np.column_stack(
        [radii * np.cos(angles), 0.75 * radii * np.sin(angles)]
    )
    side_vertices = np.vstack(
        [
            np.column_stack([profile, np.zeros(sample_count)]),
            np.column_stack([profile, np.full(sample_count, 2.0)]),
        ]
    )
    side_faces = []
    for index in range(sample_count):
        following = (index + 1) % sample_count
        side_faces.extend(
            [
                [index, following, sample_count + following],
                [index, sample_count + following, sample_count + index],
            ]
        )
    side_mesh = trimesh.Trimesh(
        vertices=side_vertices,
        faces=np.asarray(side_faces),
        process=False,
    )

    def cap_mesh(height, reverse=False):
        vertices = np.vstack(
            [[0.0, 0.0, height], np.column_stack([profile, np.full(sample_count, height)])]
        )
        faces = []
        for index in range(sample_count):
            following = (index + 1) % sample_count
            face = [0, index + 1, following + 1]
            faces.append(face[::-1] if reverse else face)
        return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)

    def fit(fit_rate, parameters):
        return {"fit_rate": fit_rate, "params": np.asarray(parameters, dtype=float)}

    results = [
        {
            "patch_index": 0,
            "mesh": side_mesh,
            "area": side_mesh.area,
            "type": "spline_extrusion",
            "fit_rate": 1.0,
            "params": np.array([0.0, 0.0, 1.0]),
            "fits": {
                "spline_extrusion": fit(1.0, [0.0, 0.0, 1.0]),
                "extrusion": fit(1.0, [0.0, 0.0, 1.0]),
                "plane": fit(0.0, []),
                "cylinder": fit(0.2, []),
                "cone": fit(0.1, []),
                "sphere": fit(0.1, []),
            },
        },
    ]
    for patch_index, height in enumerate((0.0, 2.0), start=1):
        mesh = cap_mesh(height, reverse=bool(patch_index - 1))
        results.append(
            {
                "patch_index": patch_index,
                "mesh": mesh,
                "area": mesh.area,
                "type": "plane",
                "fit_rate": 1.0,
                "params": np.array([0.0, 0.0, 1.0, -height]),
                "fits": {"plane": fit(1.0, [0.0, 0.0, 1.0, -height])},
            }
        )
    graph = {
        "nodes": [{"patch_index": index} for index in range(3)],
        "edges": [
            {"patch0": 0, "patch1": 1, "length": 1.0, "dihedral_angle": np.pi / 2.0},
            {"patch0": 0, "patch1": 2, "length": 1.0, "dihedral_angle": np.pi / 2.0},
        ],
    }

    candidates = generate_spline_extrusion_candidates(results, graph)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.patch_ids == [0, 1, 2]
    assert candidate.parameters["profile_kind"] == "PERIODIC_BSPLINE"
    assert candidate.parameters["spline_degree"] == 3
    assert len(candidate.parameters["spline_knots"]) > 4
    assert candidate.parameters["spline_control_points"].shape[1] == 2
    assert candidate.parameters["surface_control_net"].shape[1:] == (2, 3)
    assert np.array_equal(
        candidate.parameters["surface_knots_u"],
        candidate.parameters["spline_knots"],
    )
    assert primitive_contains(
        candidate.primitive_type,
        candidate.parameters,
        np.array([[0.0, 0.0, 1.0]]),
    )[0]
    assert "linear_extrude" in csg_to_openscad(
        {
            "op": candidate.primitive_type,
            "parameters": candidate.parameters,
        }
    )


def test_curved_primitive_fitting():
    theta = np.linspace(0.0, 1.6 * np.pi, 60)
    height = np.linspace(-2.0, 3.0, 25)
    theta_grid, height_grid = np.meshgrid(theta, height)
    cylinder_points = np.column_stack(
        [
            2.0 * np.cos(theta_grid.ravel()),
            2.0 * np.sin(theta_grid.ravel()),
            height_grid.ravel(),
        ]
    )
    cylinder_normals = np.column_stack(
        [
            np.cos(theta_grid.ravel()),
            np.sin(theta_grid.ravel()),
            np.zeros(theta_grid.size),
        ]
    )
    cylinder, cylinder_error = fit_cylinder(cylinder_points, cylinder_normals)
    assert np.isclose(cylinder["radius"], 2.0, atol=1e-6)
    assert cylinder_error < 1e-6

    polar = np.linspace(0.1, 1.7, 40)
    azimuth = np.linspace(0.0, 1.7 * np.pi, 35)
    polar_grid, azimuth_grid = np.meshgrid(polar, azimuth)
    sphere_center = np.array([1.0, 2.0, -1.0])
    sphere_points = sphere_center + 3.0 * np.column_stack(
        [
            np.sin(polar_grid.ravel()) * np.cos(azimuth_grid.ravel()),
            np.sin(polar_grid.ravel()) * np.sin(azimuth_grid.ravel()),
            np.cos(polar_grid.ravel()),
        ]
    )
    sphere, sphere_error = fit_sphere(sphere_points)
    assert np.allclose(sphere["center"], sphere_center, atol=1e-6)
    assert np.isclose(sphere["radius"], 3.0, atol=1e-6)
    assert sphere_error < 1e-6

    cone_height = np.linspace(1.0, 4.0, 30)
    cone_theta = np.linspace(0.0, 1.5 * np.pi, 60)
    cone_theta_grid, cone_height_grid = np.meshgrid(cone_theta, cone_height)
    cone_angle = 0.35
    cone_points = np.column_stack(
        [
            cone_height_grid.ravel() * np.tan(cone_angle) * np.cos(cone_theta_grid.ravel()),
            cone_height_grid.ravel() * np.tan(cone_angle) * np.sin(cone_theta_grid.ravel()),
            cone_height_grid.ravel(),
        ]
    )
    cone_normals = np.column_stack(
        [
            np.cos(cone_theta_grid.ravel()),
            np.sin(cone_theta_grid.ravel()),
            np.full(cone_theta_grid.size, -np.tan(cone_angle)),
        ]
    )
    cone_normals /= np.linalg.norm(cone_normals, axis=1, keepdims=True)
    cone, cone_error = fit_cone(cone_points, cone_normals)
    assert np.isclose(cone["angle"], cone_angle, atol=1e-6)
    assert cone_error < 1e-6

    major_radius = 4.0
    minor_radius = 1.0
    major_angle = np.linspace(0.0, 1.8 * np.pi, 45)
    minor_angle = np.linspace(0.0, 2.0 * np.pi, 28)
    major_grid, minor_grid = np.meshgrid(major_angle, minor_angle)
    torus_points = np.column_stack(
        [
            (major_radius + minor_radius * np.cos(minor_grid.ravel())) * np.cos(major_grid.ravel()),
            (major_radius + minor_radius * np.cos(minor_grid.ravel())) * np.sin(major_grid.ravel()),
            minor_radius * np.sin(minor_grid.ravel()),
        ]
    )
    torus, torus_error = fit_torus(torus_points)
    assert np.isclose(torus["major_radius"], major_radius, atol=1e-6)
    assert np.isclose(torus["minor_radius"], minor_radius, atol=1e-6)
    assert torus_error < 1e-6


def test_polygonal_extrusion_candidate_and_openscad_output():
    polygon = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.5, 1.0], [1.0, 2.0], [-0.5, 1.0]]
    )
    lower, upper = -1.0, 3.0
    vertices = np.vstack(
        [
            np.column_stack([polygon, np.full(len(polygon), lower)]),
            np.column_stack([polygon, np.full(len(polygon), upper)]),
        ]
    )
    results = []
    for patch_id in range(len(polygon)):
        next_id = (patch_id + 1) % len(polygon)
        local_ids = [patch_id, next_id, len(polygon) + next_id, len(polygon) + patch_id]
        mesh = trimesh.Trimesh(
            vertices=vertices[local_ids],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
        normal = mesh.face_normals.mean(axis=0)
        normal /= np.linalg.norm(normal)
        results.append(
            {
                "patch_index": patch_id,
                "mesh": mesh,
                "area": mesh.area,
                "type": "plane",
                "fits": {
                    "plane": {
                        "fit_rate": 1.0,
                        "params": np.r_[normal, -normal @ mesh.vertices.mean(axis=0)],
                    }
                },
            }
        )
    graph = {
        "nodes": [{"patch_index": index} for index in range(len(polygon))],
        "edges": [
            {
                "patch0": index,
                "patch1": (index + 1) % len(polygon),
                "length": 1.0,
                "dihedral_angle": np.pi / len(polygon),
            }
            for index in range(len(polygon))
        ],
    }

    candidates = generate_extrusion_candidates(results, graph)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.primitive_type == "EXTRUSION"
    assert candidate.patch_ids == list(range(len(polygon)))
    assert np.allclose(np.sort(candidate.parameters["extent"]), [lower, upper])
    assert len(candidate.parameters["polygon"]) == len(polygon)
    openscad = csg_to_openscad(
        {
            "op": "EXTRUSION",
            "parameters": candidate.parameters,
        }
    )
    assert "linear_extrude" in openscad
    assert "polygon(points=" in openscad


def test_difference_hole_csg():
    box = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    height = np.linspace(-2.0, 2.0, 20)
    theta_grid, height_grid = np.meshgrid(theta, height)
    vertices = np.column_stack(
        [np.cos(theta_grid.ravel()), np.sin(theta_grid.ravel()), height_grid.ravel()]
    )
    faces = []
    columns = len(theta)
    for row in range(len(height) - 1):
        for column in range(columns):
            next_column = (column + 1) % columns
            first = row * columns + column
            second = row * columns + next_column
            third = (row + 1) * columns + column
            fourth = (row + 1) * columns + next_column
            faces.extend([[first, third, second], [second, third, fourth]])
    wall = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    wall.invert()
    results = [
        {"patch_index": 0, "mesh": box, "area": box.area},
        {"patch_index": 1, "mesh": wall, "area": wall.area},
    ]
    cube = PrimitiveCandidate(
        "CUBE",
        {
            "center": np.zeros(3),
            "size": np.array([4.0, 4.0, 4.0]),
            "rotation": np.eye(3),
            "euler_xyz_degrees": np.zeros(3),
            "local_bounds": np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]),
        },
        [0],
        0.0,
        1.0,
        "ADD",
    )
    cylinder = PrimitiveCandidate(
        "CYLINDER",
        {
            "axis_point": np.zeros(3),
            "axis": np.array([0.0, 0.0, 1.0]),
            "radius": 1.0,
            "extent": np.array([-2.0, 2.0]),
        },
        [1],
        0.0,
        0.95,
    )
    classified = classify_boolean_operations([cube, cylinder], results)
    assert classified[1].operation == "SUBTRACT"
    csg_ir, selected, report = reconstruct_csg_tree(classified, results)
    assert csg_ir["op"] == "DIFFERENCE"
    assert report["patch_coverage"] == 1.0
    openscad = csg_to_openscad(csg_ir)
    assert "difference()" in openscad
    assert "cylinder(" in openscad


def _intersection_cube_candidate(center, patch_id):
    center = np.asarray(center, dtype=float)
    size = np.array([4.0, 4.0, 4.0])
    return PrimitiveCandidate(
        "CUBE",
        {
            "center": center,
            "size": size,
            "rotation": np.eye(3),
            "euler_xyz_degrees": np.zeros(3),
            "local_bounds": np.stack([-size / 2.0, size / 2.0]),
        },
        [patch_id],
        0.0,
        1.0,
        "ADD",
    )


def _square_patch(x_position, patch_id):
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [x_position, -2.0, -2.0],
                [x_position, 2.0, -2.0],
                [x_position, 2.0, 2.0],
                [x_position, -2.0, 2.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    return {"patch_index": patch_id, "mesh": mesh, "area": mesh.area}


def _corner_patch(x_position, y_position, x_range, y_range, patch_id):
    lower_x, upper_x = x_range
    lower_y, upper_y = y_range
    vertices = np.array(
        [
            [x_position, lower_y, -2.0],
            [x_position, upper_y, -2.0],
            [x_position, upper_y, 2.0],
            [x_position, lower_y, 2.0],
            [lower_x, y_position, -2.0],
            [upper_x, y_position, -2.0],
            [upper_x, y_position, 2.0],
            [lower_x, y_position, 2.0],
        ]
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]),
        process=False,
    )
    return {"patch_index": patch_id, "mesh": mesh, "area": mesh.area}


def test_intersection_relation_builds_ir_and_openscad():
    first = _intersection_cube_candidate([0.0, 0.0, 0.0], 0)
    second = _intersection_cube_candidate([1.0, 1.0, 0.0], 1)
    intersection_results = [
        _corner_patch(2.0, 2.0, (-1.0, 2.0), (-1.0, 2.0), 0),
        _corner_patch(-1.0, -1.0, (-1.0, 2.0), (-1.0, 2.0), 1),
    ]

    csg_ir = build_csg_ir([first, second], intersection_results)

    assert csg_ir["op"] == "INTERSECTION"
    points = np.array([[0.0, 0.0, 0.0], [-1.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
    assert np.array_equal(csg_contains(csg_ir, points), [True, False, False])
    assert "intersection()" in csg_to_openscad(csg_ir)


def test_overlapping_union_is_not_misclassified_as_intersection():
    first = _intersection_cube_candidate([0.0, 0.0, 0.0], 0)
    second = _intersection_cube_candidate([1.0, 1.0, 0.0], 1)
    union_results = [
        _square_patch(-2.0, 0),
        _square_patch(3.0, 1),
    ]

    csg_ir = build_csg_ir([first, second], union_results)

    assert csg_ir["op"] == "UNION"


def test_class_based_api_preserves_functional_pipeline():
    results, graph = _rotated_box_patches(rotated=False)
    generator = PrimitiveCandidateGenerator(include_torus=False)
    candidates = generator.generate_cube(
        results,
        graph,
    )

    complete = [candidate for candidate in candidates if len(candidate.patch_ids) == 6]
    assert complete
    assert PrimitiveFitter.sphere is fit_sphere

    reconstructor = BooleanCSGReconstructor()
    classified = reconstructor.classify(complete, results)
    csg_ir, selected, report = reconstructor.reconstruct(classified, results)

    assert csg_ir["op"] == "CUBE"
    assert report["patch_coverage"] == 1.0
    assert CSGValidator.contains(csg_ir, np.array([[3.0, -2.0, 5.0]])).all()
    assert OpenSCADExporter.generate(csg_ir) == csg_to_openscad(csg_ir)
    assert selected
