import numpy as np
import pytest
import trimesh

from primitive_fitting.fitting import PrimitiveFitting
from primitive_fitting.geometry_primitive import (
    Torus,
    is_cylinder_same,
    is_plane_same,
)
from primitive_fitting.primitives_2d import fit_circle


def _fitting(mesh):
    return PrimitiveFitting(
        np.asarray(mesh.vertices, dtype=float),
        np.asarray(mesh.faces, dtype=int),
    )


def _side_faces(mesh):
    return np.flatnonzero(np.abs(np.asarray(mesh.face_normals)[:, 2]) < 0.9)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_plane_is_scale_invariant_and_rejects_other_primitives(scale):
    mesh = trimesh.creation.box(extents=[2.0 * scale, 3.0 * scale, 0.2 * scale])
    face_ids = np.flatnonzero(np.asarray(mesh.face_normals)[:, 2] > 0.9)
    fitting = _fitting(mesh)

    plane_rate, params = fitting.fit_planar(face_ids)
    cylinder_rate, _ = fitting.fit_cylinder(face_ids)
    sphere_rate, _ = fitting.fit_sphere(face_ids)
    cone_rate, _ = fitting.fit_cone(face_ids)

    assert plane_rate == pytest.approx(1.0)
    assert abs(np.dot(params[:3], [0.0, 0.0, 1.0])) > 0.999
    assert max(cylinder_rate, sphere_rate, cone_rate) < 0.5


@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_cylinder_fit_is_scale_invariant_and_has_correct_parameters(scale):
    radius = 2.0 * scale
    mesh = trimesh.creation.cylinder(radius=radius, height=5.0 * scale, sections=64)
    face_ids = _side_faces(mesh)
    fitting = _fitting(mesh)

    cylinder_rate, params = fitting.fit_cylinder(face_ids)
    plane_rate, _ = fitting.fit_planar(face_ids)
    sphere_rate, _ = fitting.fit_sphere(face_ids)

    assert cylinder_rate > 0.99
    assert plane_rate < 0.5
    assert sphere_rate < 0.5
    assert abs(np.dot(params[3:6], [0.0, 0.0, 1.0])) > 0.999
    assert params[6] == pytest.approx(radius, rel=1e-4)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_sphere_fit_is_scale_invariant_and_has_correct_parameters(scale):
    radius = 2.3 * scale
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    face_ids = np.arange(len(mesh.faces))
    fitting = _fitting(mesh)

    sphere_rate, params = fitting.fit_sphere(face_ids)
    plane_rate, _ = fitting.fit_planar(face_ids)
    cylinder_rate, _ = fitting.fit_cylinder(face_ids)

    assert sphere_rate > 0.99
    assert plane_rate < 0.5
    assert cylinder_rate < 0.5
    np.testing.assert_allclose(params[:3], np.zeros(3), atol=max(scale * 1e-8, 1e-12))
    assert params[3] == pytest.approx(radius, rel=1e-8)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_cone_fit_is_scale_invariant_and_has_correct_parameters(scale):
    radius = 2.0 * scale
    height = 4.0 * scale
    mesh = trimesh.creation.cone(radius=radius, height=height, sections=64)
    face_ids = _side_faces(mesh)
    fitting = _fitting(mesh)

    cone_rate, params = fitting.fit_cone(face_ids)
    plane_rate, _ = fitting.fit_planar(face_ids)
    cylinder_rate, _ = fitting.fit_cylinder(face_ids)
    sphere_rate, _ = fitting.fit_sphere(face_ids)

    assert cone_rate > 0.99
    assert max(plane_rate, cylinder_rate, sphere_rate) < 0.5
    assert abs(np.dot(params[3:6], [0.0, 0.0, 1.0])) > 0.999
    assert params[6] == pytest.approx(np.arctan(radius / height), rel=1e-3)


def test_partial_surfaces_still_fit_their_primitives():
    cylinder = trimesh.creation.cylinder(radius=2.0, height=5.0, sections=64)
    cylinder_centers = np.asarray(cylinder.triangles_center)
    cylinder_angles = np.arctan2(cylinder_centers[:, 1], cylinder_centers[:, 0])
    cylinder_faces = np.flatnonzero(
        (np.abs(np.asarray(cylinder.face_normals)[:, 2]) < 0.5)
        & (np.abs(cylinder_angles) < 0.8)
    )
    cylinder_rate, _ = _fitting(cylinder).fit_cylinder(cylinder_faces)

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=2.3)
    sphere_centers = np.asarray(sphere.triangles_center)
    sphere_faces = np.flatnonzero((sphere_centers[:, 0] > 0) & (sphere_centers[:, 2] > 0))
    sphere_rate, _ = _fitting(sphere).fit_sphere(sphere_faces)

    cone = trimesh.creation.cone(radius=2.0, height=4.0, sections=128)
    cone_centers = np.asarray(cone.triangles_center)
    cone_angles = np.arctan2(cone_centers[:, 1], cone_centers[:, 0])
    cone_faces = np.flatnonzero(
        (np.abs(np.asarray(cone.face_normals)[:, 2]) < 0.9)
        & (np.abs(cone_angles) < 1.5)
    )
    cone_rate, _ = _fitting(cone).fit_cone(cone_faces)

    assert cylinder_rate > 0.99
    assert sphere_rate > 0.99
    assert cone_rate > 0.99


def test_vertex_id_api_fits_each_supported_surface():
    plane = trimesh.creation.box(extents=[2.0, 3.0, 0.2])
    plane_faces = np.flatnonzero(np.asarray(plane.face_normals)[:, 2] > 0.9)
    plane = plane.submesh([plane_faces], append=True, repair=False)

    cylinder = trimesh.creation.cylinder(radius=2.0, height=5.0, sections=64)
    cylinder = cylinder.submesh([_side_faces(cylinder)], append=True, repair=False)

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=2.3)

    cone = trimesh.creation.cone(radius=2.0, height=4.0, sections=64)
    cone = cone.submesh([_side_faces(cone)], append=True, repair=False)

    for mesh, method_name in (
        (plane, "fit_planar"),
        (cylinder, "fit_cylinder"),
        (sphere, "fit_sphere"),
        (cone, "fit_cone"),
    ):
        fitting = _fitting(mesh)
        rate, _ = getattr(fitting, method_name)(
            np.arange(len(mesh.vertices)),
            is_v_id=True,
        )
        assert rate > 0.99


def test_small_vertex_noise_is_tolerated():
    rng = np.random.default_rng(7)
    cases = (
        (trimesh.creation.cylinder(radius=2.0, height=5.0, sections=64), "fit_cylinder"),
        (trimesh.creation.icosphere(subdivisions=2, radius=2.3), "fit_sphere"),
        (trimesh.creation.cone(radius=2.0, height=4.0, sections=64), "fit_cone"),
    )

    for mesh, method_name in cases:
        mesh.vertices += rng.normal(scale=1e-3, size=mesh.vertices.shape)
        face_ids = (
            np.arange(len(mesh.faces))
            if method_name == "fit_sphere"
            else _side_faces(mesh)
        )
        rate, _ = getattr(_fitting(mesh), method_name)(face_ids)
        assert rate > 0.95


def test_extrusion_fits_the_gauss_normal_plane():
    mesh = trimesh.creation.box(extents=[2.0, 3.0, 5.0])
    side_faces = np.flatnonzero(np.abs(np.asarray(mesh.face_normals)[:, 2]) < 0.5)

    rate, direction = _fitting(mesh).fit_extrusion(side_faces)

    assert rate > 0.99
    assert abs(np.dot(direction, [0.0, 0.0, 1.0])) > 0.999


@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_torus_fit_recovers_center_axis_and_radii(scale):
    u = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    v = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    major, minor = 3.0 * scale, 0.8 * scale
    points = np.c_[
        ((major + minor * np.cos(vv)) * np.cos(uu)).ravel(),
        ((major + minor * np.cos(vv)) * np.sin(uu)).ravel(),
        (minor * np.sin(vv)).ravel(),
    ]

    rate, params = Torus(points).fit()

    assert rate > 0.99
    np.testing.assert_allclose(params[:3], np.zeros(3), atol=max(scale * 1e-8, 1e-12))
    assert abs(np.dot(params[3:6], [0.0, 0.0, 1.0])) > 0.999
    assert params[6] == pytest.approx(major, rel=1e-8)
    assert params[7] == pytest.approx(minor, rel=1e-8)


def test_dense_planar_patch_is_not_accepted_as_cylinder_or_torus():
    grid_size = 12
    xx, yy = np.meshgrid(
        np.linspace(-2.0, 2.0, grid_size),
        np.linspace(-3.0, 3.0, grid_size),
        indexing="ij",
    )
    vertices = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            a = i * grid_size + j
            b = a + 1
            c = a + grid_size
            d = c + 1
            faces.extend(([a, c, b], [b, c, d]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    fitting = _fitting(mesh)
    face_ids = np.arange(len(mesh.faces))

    plane_rate, _ = fitting.fit_planar(face_ids)
    cylinder_rate, cylinder_params = fitting.fit_cylinder(face_ids)
    torus_rate, torus_params = fitting.fit_torus(face_ids)

    assert plane_rate == pytest.approx(1.0)
    assert cylinder_rate == 0.0
    assert torus_rate == 0.0
    assert np.all(np.isnan(cylinder_params))
    assert np.all(np.isnan(torus_params))


def test_precomputed_face_attributes_are_used_in_the_documented_order():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals)
    centers = np.asarray(mesh.triangles_center)
    areas = np.asarray(mesh.area_faces)

    fitting = PrimitiveFitting(vertices, faces, normals, centers, areas)

    np.testing.assert_allclose(fitting.f_normal, normals)
    np.testing.assert_allclose(fitting.f_center, centers)
    np.testing.assert_allclose(fitting.f_area, areas)


def test_degenerate_selections_return_a_stable_failure_result():
    mesh = trimesh.creation.box()
    fitting = _fitting(mesh)

    for method, size in (
        (fitting.fit_planar, 4),
        (fitting.fit_cylinder, 7),
        (fitting.fit_sphere, 4),
        (fitting.fit_cone, 7),
        (fitting.fit_extrusion, 3),
        (fitting.fit_torus, 8),
    ):
        rate, params = method(np.array([], dtype=int))
        assert rate == 0.0
        assert np.asarray(params).shape == (size,)
        assert np.all(np.isnan(params))


def test_legacy_public_entry_points_use_the_robust_fitters():
    cylinder = trimesh.creation.cylinder(radius=2.0, height=5.0, sections=32)
    cylinder_faces = _side_faces(cylinder)
    fitting = _fitting(cylinder)
    sphere_rate, sphere_params = fitting.fit_sphere(cylinder_faces)

    from primitive_fitting.geometry_primitive import Cone, Sphere

    sphere = Sphere(
        fitting.v,
        fitting.f,
        fitting.f_normal,
        fitting.f_area,
        fitting.f_center,
        fitting.v_normal,
    )
    alias_rate, alias_params = sphere.optmize_fit(cylinder_faces, False)
    assert alias_rate == pytest.approx(sphere_rate)
    np.testing.assert_allclose(alias_params, sphere_params)

    cone_mesh = trimesh.creation.cone(radius=2.0, height=4.0, sections=32)
    cone_fitting = _fitting(cone_mesh)
    cone_faces = _side_faces(cone_mesh)
    cone = Cone(
        cone_fitting.v,
        cone_fitting.f,
        cone_fitting.f_normal,
        cone_fitting.f_center,
        cone_fitting.v_normal,
    )
    direct_rate, direct_params = cone.fit(cone_faces, False)
    alias_rate, alias_params = cone.fit_optimization(cone_faces, False)
    assert alias_rate == pytest.approx(direct_rate)
    np.testing.assert_allclose(alias_params, direct_params)


def test_patch_classification_keeps_a_successful_cone_result():
    from patch_graph.patch_mesh import BorderType, PatchMesh

    mesh = trimesh.creation.cone(radius=2.0, height=4.0, sections=32)
    face_ids = _side_faces(mesh)
    patch_mesh = PatchMesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        {0: face_ids},
    )

    BorderType(patch_mesh).get_patch_type()

    assert patch_mesh.graph.nodes[0]["type"] == "Cone"
    assert np.asarray(patch_mesh.graph.nodes[0]["params"]).shape == (7,)


def test_circle_fit_and_degenerate_circle_input():
    angles = np.linspace(-0.8, 0.8, 20)
    center = np.array([2.0, -3.0])
    radius = 4.0
    points = center + radius * np.c_[np.cos(angles), np.sin(angles)]

    fitted_center, fitted_radius = fit_circle(points)

    np.testing.assert_allclose(fitted_center, center, atol=1e-7)
    assert fitted_radius == pytest.approx(radius, rel=1e-7)
    with pytest.raises(ValueError, match="non-collinear"):
        fit_circle(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def test_primitive_equivalence_is_sign_invariant_but_not_offset_invariant():
    assert is_plane_same([0, 0, 1, -2], [0, 0, -2, 4])
    assert not is_plane_same([0, 0, 1, -2], [0, 0, 1, -2.1])

    cylinder = np.array([0, 0, 0, 0, 0, 1, 2.0], dtype=float)
    same_line = np.array([0, 0, 4, 0, 0, -1, 2.0], dtype=float)
    offset_line = np.array([0.2, 0, 0, 0, 0, 1, 2.0], dtype=float)
    assert is_cylinder_same(cylinder, same_line)
    assert not is_cylinder_same(cylinder, offset_line)
