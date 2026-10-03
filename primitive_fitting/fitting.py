import igl
import numpy as np

from primitive_fitting.geometry_primitive import (
    Cone,
    Cylinder,
    Extrusion,
    Plane,
    Sphere,
    Torus,
)
from primitive_fitting.primitives_2d import fit_circle as fit_circle_2d


def _normalized_rows(values):
    values = np.asarray(values, dtype=float)
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    result = np.zeros_like(values)
    valid = lengths[:, 0] > np.finfo(float).eps
    result[valid] = values[valid] / lengths[valid]
    return result


class PrimitiveFitting:
    """Fit analytic primitives to a mesh face or vertex subset.

    Face-based calls use unique face vertices for geometric residuals and the
    selected face centers/normals for orientation validation. Parameter
    conventions are documented by the corresponding primitive classes.
    """

    def __init__(self, v, f, f_normal=None, f_center=None, f_area=None):
        self.v = np.asarray(v, dtype=float)
        self.f = np.asarray(f, dtype=int)
        if self.v.ndim != 2 or self.v.shape[1] != 3:
            raise ValueError("v must have shape (n, 3)")
        if self.f.ndim != 2 or self.f.shape[1] != 3:
            raise ValueError("f must have shape (m, 3)")

        self.v_normal = _normalized_rows(
            np.asarray(igl.per_vertex_normals(self.v, self.f)).reshape((-1, 3))
        )
        self.f_center = (
            np.mean(self.v[self.f], axis=1)
            if f_center is None
            else np.asarray(f_center, dtype=float)
        )
        self.f_area = (
            igl.doublearea(self.v, self.f) / 2
            if f_area is None
            else np.asarray(f_area, dtype=float)
        )
        self.f_area = np.asarray(self.f_area, dtype=float).reshape(-1)
        self.f_normal = (
            igl.per_face_normals(self.v, self.f, np.array([1.0, 0.0, 0.0]))
            if f_normal is None
            else np.asarray(f_normal, dtype=float)
        )
        self.f_normal = _normalized_rows(
            np.asarray(self.f_normal).reshape((-1, 3))
        )

        if self.f_center.shape != (len(self.f), 3):
            raise ValueError("f_center must have shape (m, 3)")
        if self.f_area.shape != (len(self.f),):
            raise ValueError("f_area must have shape (m,)")
        if self.f_normal.shape != (len(self.f), 3):
            raise ValueError("f_normal must have shape (m, 3)")

    def fit_planar(self, f_id, is_v_id=False):
        plane = Plane(self.v, self.f, self.f_normal, self.f_area, self.f_center)
        return plane.fit(f_id, is_v_id)

    def fit_extrusion(self, f_id, is_v_id=False, visualize=False):
        extrusion = Extrusion(self.v, self.f, self.f_normal, self.v_normal)
        return extrusion.fit(f_id, is_v_id, visualize)

    def fit_cylinder(self, f_id, is_v_id=False, visualize=False):
        cylinder = Cylinder(
            self.v,
            self.f,
            self.f_normal,
            self.f_center,
            self.f_area,
            self.v_normal,
        )
        return cylinder.fit(f_id, is_v_id, visualize=visualize)

    def fit_circle(self, points):
        return fit_circle_2d(points)

    def fit_sphere(self, f_id, is_v_id=False):
        sphere = Sphere(
            self.v,
            self.f,
            self.f_normal,
            self.f_area,
            self.f_center,
            self.v_normal,
        )
        return sphere.fit(f_id, is_v_id)

    def fit_cone(self, f_id, is_v_id=False):
        cone = Cone(
            self.v,
            self.f,
            self.f_normal,
            self.f_center,
            self.v_normal,
        )
        return cone.fit_on_all_points(f_id, is_v_id)

    def fit_torus(self, f_id, is_v_id=False):
        torus = Torus(self.v, self.f, self.f_normal)
        return torus.fit(f_id, is_v_id)
