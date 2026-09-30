'''
    TODO
        + input f_id => v_id, do duplicate check
        + pre compute vertex&face normal, face area, face center
        + difference of fitting vertex and face
'''

# import scipy.optimize
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize, least_squares
import sklearn
import igl
import meshplot as mp
import scipy
import sklearn
from scipy.special import comb
# mp.offline()
# import open3d as o3d



_EPS = np.finfo(float).eps


def _index_array(ids):
    if isinstance(ids, np.ndarray):
        result = ids
    else:
        result = np.asarray(list(ids) if not np.isscalar(ids) else [ids])
    return np.asarray(result, dtype=int).reshape(-1)


def _selected_points(vertices, faces, ids, is_v_id):
    indices = _index_array(ids)
    if indices.size == 0:
        return np.empty((0, 3), dtype=float), indices
    if is_v_id:
        return np.asarray(vertices[indices], dtype=float), indices
    vertex_ids = np.unique(faces[indices].reshape(-1))
    return np.asarray(vertices[vertex_ids], dtype=float), indices


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length <= _EPS:
        return None
    return vector / length


def _unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    result = np.zeros_like(vectors)
    valid = np.isfinite(lengths) & (lengths > _EPS)
    result[valid] = vectors[valid] / lengths[valid, None]
    return result, valid


def _characteristic_scale(points):
    if len(points) == 0:
        return 0.0
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def _distance_tolerance(points, relative=1e-2):
    scale = _characteristic_scale(points)
    coordinate_scale = max(float(np.max(np.abs(points))) if len(points) else 0.0, 1.0)
    return max(relative * scale, 100.0 * _EPS * coordinate_scale)


def _failure(size):
    return 0.0, np.full(size, np.nan)



def vector_same_direction(v1, v2):
    v1 = _unit(v1)
    v2 = _unit(v2)
    if v1 is None or v2 is None:
        return False
    return np.linalg.norm(np.cross(v1, v2)) < 1e-2
    

def is_plane_same(plane1, plane2):
    plane1 = np.asarray(plane1, dtype=float).reshape(-1)
    plane2 = np.asarray(plane2, dtype=float).reshape(-1)
    if plane1.size != 4 or plane2.size != 4:
        return False
    norm1 = np.linalg.norm(plane1[:3])
    norm2 = np.linalg.norm(plane2[:3])
    if norm1 <= _EPS or norm2 <= _EPS:
        return False
    normalized1 = plane1 / norm1
    normalized2 = plane2 / norm2
    if np.dot(normalized1[:3], normalized2[:3]) < 0:
        normalized2 = -normalized2
    return np.allclose(normalized1, normalized2, rtol=1e-3, atol=1e-6)



def is_cylinder_same(cylinder1, cylinder2):
    cylinder1 = np.asarray(cylinder1, dtype=float).reshape(-1)
    cylinder2 = np.asarray(cylinder2, dtype=float).reshape(-1)
    if cylinder1.size != 7 or cylinder2.size != 7:
        return False
    c1, axis1, r1 = cylinder1[:3], _unit(cylinder1[3:6]), cylinder1[6]
    c2, axis2, r2 = cylinder2[:3], _unit(cylinder2[3:6]), cylinder2[6]
    if axis1 is None or axis2 is None or r1 <= 0 or r2 <= 0:
        return False
    if not np.isclose(r1, r2, rtol=1e-2, atol=1e-8):
        return False
    if np.linalg.norm(np.cross(axis1, axis2)) >= 1e-2:
        return False
    axis_distance = np.linalg.norm(np.cross(c2 - c1, axis1))
    return axis_distance <= 1e-2 * max(r1, r2)


class Plane:
    def __init__(self, v, f, f_normal, f_area, f_center):
        self.v = v
        self.f = f
        self.f_normal = f_normal
        self.f_area = f_area
        self.f_center = f_center
        
        self.threshold = 1e-2
        
    def fit(self, id, is_v_id):
        points, indices = _selected_points(self.v, self.f, id, is_v_id)
        if len(points) < 3:
            return _failure(4)

        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, singular_values, vh = np.linalg.svd(centered_points, full_matrices=False)
        if len(singular_values) < 2 or singular_values[1] <= _EPS:
            return _failure(4)
        normal = _unit(vh[-1])
        if normal is None:
            return _failure(4)

        d = -float(np.dot(normal, centroid))
        params = np.r_[normal, d]
        distances = np.abs(centered_points @ normal)
        tolerance = _distance_tolerance(points, self.threshold)
        fit_rate = float(np.mean(distances <= tolerance))
        if not is_v_id:
            measured, valid = _unit_rows(self.f_normal[indices])
            if np.any(valid):
                alignment = np.abs(measured[valid] @ normal)
                normal_rate = float(np.mean(alignment >= np.cos(np.deg2rad(10.0))))
                fit_rate = min(fit_rate, normal_rate)
        return fit_rate, params

    


class Sphere:
    """Sphere parameters are ``[cx, cy, cz, radius]``."""

    def __init__(self, v, f, f_normal, f_area, f_center, v_normal=None):
        self.v = v
        self.f = f
        self.f_normal = f_normal
        self.f_area = f_area
        self.f_center = f_center
        self.v_normal = (
            igl.per_vertex_normals(v, f)
            if v_normal is None
            else np.asarray(v_normal, dtype=float)
        )
        
        self.threshold = 1e-2
        self.normal_angle = np.deg2rad(20.0)

    def _legacy_optmize_fit(self, id, is_v_id):
        if is_v_id:
            points = np.asarray(self.v[id])
        else:
            v_id = np.unique(self.f[id].flatten())
            points = np.asarray(self.v[v_id])
            
        initial_center = np.mean(points, axis=0)
        distances = np.sqrt(np.sum((points - initial_center)**2, axis=1))
        initial_radius = np.mean(distances)
        initial_params = np.append(initial_center, initial_radius)
        
        def objective(params):
            x_c, y_c, z_c, r = params
            distances = np.sqrt(np.sum((points - [x_c, y_c, z_c])**2, axis=1))
            return np.sum((distances - r)**2)
        
        result = scipy.optimize.minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': 1000
            }
        )
        
        if not result.success:
            return 0., []
        
        center = result.x[:3].reshape((1, 3))
        radius = result.x[3]
        
        
        dist_pt = np.linalg.norm(points - center, axis=1)
        dist_r = np.abs(dist_pt - radius)
        eps = 5*1e-2
        n_points = points.shape[0]
        fit_rate = np.sum(dist_r < eps * radius) / n_points
        
        return fit_rate, result.x

            
    

    def fit(self, id, is_v_id):
        points, indices = _selected_points(self.v, self.f, id, is_v_id)
        if len(points) < 4 or _characteristic_scale(points) <= _EPS:
            return _failure(4)

        design = np.c_[2.0 * points, np.ones(len(points))]
        target = np.einsum("ij,ij->i", points, points)
        solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < 4:
            return _failure(4)

        center = solution[:3]
        radius_squared = float(solution[3] + np.dot(center, center))
        if not np.isfinite(radius_squared) or radius_squared <= _EPS:
            return _failure(4)
        radius = np.sqrt(radius_squared)
        tolerance = max(
            self.threshold * max(radius, _characteristic_scale(points)),
            _distance_tolerance(points, 0.0),
        )

        def residual(params):
            return np.linalg.norm(points - params[:3], axis=1) - params[3]

        result = least_squares(
            residual,
            np.r_[center, radius],
            bounds=(np.r_[[-np.inf] * 3, _EPS], np.inf),
            loss="soft_l1",
            f_scale=tolerance,
            max_nfev=1000,
        )
        params = result.x if result.success else np.r_[center, radius]
        params[3] = abs(params[3])
        point_rate = float(np.mean(np.abs(residual(params)) <= tolerance))

        if is_v_id:
            sample_points = points
            sample_normals = self.v_normal[indices]
        else:
            sample_points = np.asarray(self.f_center[indices], dtype=float)
            sample_normals = np.asarray(self.f_normal[indices], dtype=float)
        predicted, valid_predicted = _unit_rows(sample_points - params[:3])
        measured, valid_measured = _unit_rows(sample_normals)
        valid = valid_predicted & valid_measured
        if np.any(valid):
            alignment = np.abs(np.einsum("ij,ij->i", predicted[valid], measured[valid]))
            normal_rate = float(np.mean(alignment >= np.cos(self.normal_angle)))
            point_rate = min(point_rate, normal_rate)

        return point_rate, params

    def optmize_fit(self, id, is_v_id):
        """Backward-compatible alias for the historical misspelled API."""
        return self.fit(id, is_v_id)
            
                
class Extrusion:
    def __init__(self, v, f, f_normal=None, v_normal=None):
        self.v, self.f = v, f
        self.v_normal = (
            igl.per_vertex_normals(v, f)
            if v_normal is None
            else np.asarray(v_normal, dtype=float)
        )
        self.f_normal = (
            igl.per_face_normals(v, f, np.array([1.0, 0.0, 0.0]))
            if f_normal is None
            else np.asarray(f_normal, dtype=float)
        )
    
        self.threshold_normal_planar = 5 * 1e-2


    def fit_normal_to_plane(self, normals):
        normals, valid = _unit_rows(normals)
        normals = normals[valid]
        if len(normals) < 3:
            return _failure(4)

        centroid = np.mean(normals, axis=0)
        centered = normals - centroid
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if len(singular_values) < 2 or singular_values[1] <= _EPS:
            return _failure(4)
        plane_normal = _unit(vh[-1])
        if plane_normal is None:
            return _failure(4)

        offset = -float(np.dot(plane_normal, centroid))
        param = np.r_[plane_normal, offset]
        distances = np.abs(normals @ plane_normal + offset)
        fit_rate = float(np.mean(distances <= self.threshold_normal_planar))
        return fit_rate, param
    
    def fit(self, id, is_v_id, visualize=False):
        indices = _index_array(id)
        if indices.size == 0:
            return _failure(3)
        if is_v_id:
            normals = self.v_normal[indices]
        else:
            normals = self.f_normal[indices]
        
        # normals on same plane on Gauss sphere
        normal_plane_rate, normal_plane_param = self.fit_normal_to_plane(normals)
        if normal_plane_rate <= 0.9 or not np.all(np.isfinite(normal_plane_param)):
            return _failure(3)
        
        return normal_plane_rate, normal_plane_param[:3]





class Cone:
    """Cone parameters are ``[apex, unit_axis, half_angle]``."""

    def __init__(self, v, f, f_normal=None, f_center=None, v_normal=None):
        self.v = np.asarray(v, dtype=float)
        self.f = np.asarray(f, dtype=int)
        self.v_normal = (
            igl.per_vertex_normals(v, f)
            if v_normal is None
            else np.asarray(v_normal, dtype=float)
        )
        self.f_normal = (
            igl.per_face_normals(v, f, np.array([1.0, 0.0, 0.0]))
            if f_normal is None
            else np.asarray(f_normal, dtype=float)
        )
        self.f_center = (
            np.mean(self.v[self.f], axis=1)
            if f_center is None
            else np.asarray(f_center, dtype=float)
        )

        self.threshold = 1e-2
        self.normal_angle = np.deg2rad(20.0)

    def point_to_cone(self, point, p, axis, theta):
        vec = point - p
        dot = np.dot(vec, axis)
        
        v_perp = vec - dot * axis
        r = np.linalg.norm(v_perp)
        
        r_expected = dot * np.tan(theta)
        
        distance = np.abs(r - r_expected)
        return distance
    
    def points_to_cone(self, points, p, axis, theta):
        vec = points - p
        dot = np.dot(vec, axis)
        
        v_perp = vec - dot[:, np.newaxis] * axis
        r = np.linalg.norm(v_perp, axis=1)
        
        r_expected = dot * np.tan(theta)
        
        distances = np.abs(r - r_expected)
        
        return np.sum(distances**2)
    
    def objective_function(self, params, points, apex):
        axis = params[:3]
        axis /= np.linalg.norm(np.array(axis))
        theta = params[3]
        return self.points_to_cone(points, apex, axis, theta)
    

    def _legacy_fit_optimization(self, id, is_v_id, visualize=False):
        if is_v_id:
            points = self.v[id]
        else:
            v_id = np.unique(self.f[id].flatten())
            points = self.v[v_id]
        
        centroid = np.mean(points, axis=0)
        initial_axis = np.array([0, 0, 1])
        initial_theta = np.pi / 8
        initial_params = np.concatenate([centroid, initial_axis, [initial_theta]])
        
        # Constraints: axis should be normalized
        constraints = {
            'type': 'eq', 
            'fun': lambda params: np.linalg.norm(params[3:6]) - 1
        }
        
        # Optimize
        result = minimize(
            self.objective_function,
            initial_params,
            args=(points,),
            constraints=constraints,
            method='SLSQP',
            options={'disp': True}
        )
        
        if result.success:
            p = result.x[:3]
            axis = result.x[3:6] / np.linalg.norm(result.x[3:6])  # Ensure normalized
            theta = result.x[6]
            # return vertex, axis, theta

            vec = points - p
            dot = np.dot(vec, axis)
            v_perp = vec - dot[:, np.newaxis] * axis
            r = np.linalg.norm(v_perp, axis=1)
            r_expected = dot * np.tan(theta)
            
            distances = np.abs(r - r_expected)
            fit_rate = np.sum(distances < self.threshold) / points.shape[0]
            return fit_rate, np.concatenate(p, axis, theta)
        else:
            # raise RuntimeError("Optimization failed")
            return 0, []
        
    
    def fit_on_all_points(self, id, is_v_id, visualize=False):
        points, indices = _selected_points(self.v, self.f, id, is_v_id)
        if len(points) < 4 or _characteristic_scale(points) <= _EPS:
            return _failure(7)

        if is_v_id:
            tangent_points = points
            tangent_normals = self.v_normal[indices]
        else:
            tangent_points = np.asarray(self.f_center[indices], dtype=float)
            tangent_normals = np.asarray(self.f_normal[indices], dtype=float)

        tangent_normals, valid_normals = _unit_rows(tangent_normals)
        tangent_points = tangent_points[valid_normals]
        tangent_normals = tangent_normals[valid_normals]
        if len(tangent_points) < 3 or np.linalg.matrix_rank(tangent_normals) < 3:
            return _failure(7)

        tangent_offsets = np.einsum("ij,ij->i", tangent_normals, tangent_points)
        apex, _, rank, _ = np.linalg.lstsq(tangent_normals, tangent_offsets, rcond=None)
        if rank < 3 or not np.all(np.isfinite(apex)):
            return _failure(7)

        rays = points - apex
        ray_directions, valid_rays = _unit_rows(rays)
        if np.sum(valid_rays) < 3:
            return _failure(7)
        ray_directions = ray_directions[valid_rays]
        axis = _unit(np.mean(ray_directions, axis=0))
        if axis is None:
            _, _, vh = np.linalg.svd(ray_directions, full_matrices=False)
            axis = _unit(vh[0])
        if axis is None:
            return _failure(7)
        if np.median(ray_directions @ axis) < 0:
            axis = -axis

        angles = np.arccos(np.clip(ray_directions @ axis, -1.0, 1.0))
        theta = float(np.median(angles))
        if not np.isfinite(theta) or not np.deg2rad(0.1) < theta < np.deg2rad(89.0):
            return _failure(7)

        tolerance = _distance_tolerance(points, self.threshold)

        def residual(params):
            candidate_apex = params[:3]
            candidate_axis = _unit(params[3:6])
            if candidate_axis is None:
                return np.full(len(points) + 1, 1e6)
            candidate_theta = params[6]
            q = points - candidate_apex
            axial = q @ candidate_axis
            radial = np.linalg.norm(q - np.outer(axial, candidate_axis), axis=1)
            geometric = (radial - axial * np.tan(candidate_theta)) * np.cos(candidate_theta)
            unit_penalty = (np.linalg.norm(params[3:6]) - 1.0) * max(
                _characteristic_scale(points), 1.0
            )
            return np.r_[geometric, unit_penalty]

        initial = np.r_[apex, axis, theta]
        lower = np.r_[[-np.inf] * 6, np.deg2rad(0.1)]
        upper = np.r_[[np.inf] * 6, np.deg2rad(89.0)]
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=tolerance,
            max_nfev=2000,
        )
        params = result.x if result.success else initial
        fitted_axis = _unit(params[3:6])
        if fitted_axis is None:
            return _failure(7)
        params[3:6] = fitted_axis
        if np.median((points - params[:3]) @ fitted_axis) < 0:
            params[3:6] = -fitted_axis
            fitted_axis = params[3:6]

        q = points - params[:3]
        axial = q @ fitted_axis
        radial_vectors = q - np.outer(axial, fitted_axis)
        radial = np.linalg.norm(radial_vectors, axis=1)
        distances = np.abs(
            (radial - axial * np.tan(params[6])) * np.cos(params[6])
        )
        forward = axial >= -tolerance
        point_rate = float(np.mean((distances <= tolerance) & forward))

        q_normal = tangent_points - params[:3]
        axial_normal = q_normal @ fitted_axis
        radial_normal, valid_radial = _unit_rows(
            q_normal - np.outer(axial_normal, fitted_axis)
        )
        predicted = (
            radial_normal * np.cos(params[6])
            - fitted_axis[None, :] * np.sin(params[6])
        )
        measured, valid_measured = _unit_rows(tangent_normals)
        valid = valid_radial & valid_measured
        if np.any(valid):
            alignment = np.abs(np.einsum("ij,ij->i", predicted[valid], measured[valid]))
            normal_rate = float(np.mean(alignment >= np.cos(self.normal_angle)))
            point_rate = min(point_rate, normal_rate)

        return point_rate, params
        
        
    
    def get_cone_param_single(self, points, normals):
        # apex
        B = np.zeros(3)
        B[:] = normals[:, 0] * points[:, 0] + normals[:, 1] * points[:, 1] + normals[:, 2] * points[:, 2]
        try:
            apex = np.linalg.solve(normals, B)
        except:
            return None, None, None
        
        # axis
        P = points - apex
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        u, v = P[1] - P[0], P[2] - P[0]
        axis = np.cross(u, v)
        axis /= np.linalg.norm(axis)
        
        # theta
        theta = np.abs(np.arccos(np.dot(points[0] - apex, axis) / (np.linalg.norm(points[0] - apex) / np.linalg.norm(axis))))
        if theta > np.pi / 2:
            return None, None, None

        return apex, axis, theta
        
        
        
    def _legacy_fit(self, id, is_v_id):
        if is_v_id:
            points = self.v[id]
            points_normal = self.v_normal[id]
        else:
            v_id = np.unique(self.f[id].flatten())
            points = self.v[v_id]
            points_normal = self.v_normal[v_id]
            
        
        n_points = points.shape[0]
        
        eps = 1e-3
        sample_radio = 1e-1
        max_iter = 1000
        min_sample_num = 100
        
        if n_points < min_sample_num:
            sample_points = np.copy(points)
        else:
            sample_num = np.max(min_sample_num, int(sample_radio * n_points))
            sample_points = points[np.random.choice(n_points, sample_num, replace=False)]
            
        best_radio = 0
        best_apex, best_axis, best_theta = None, None, None
        for _ in range(0, max_iter):
            index = np.random.choice(n_points, 3, replace=False)
            cur_normals = points_normal[index]
            cur_points = points[index]
            
            cur_apex, cur_axis, cur_theta = self.get_cone_param_single(cur_points, cur_normals)
            if cur_apex is None:
                continue
            
            # compute distance on all points
            u = sample_points - cur_apex
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            v = cur_axis / np.linalg.norm(cur_axis)
            angle = np.abs(np.arccos(np.dot(u, v)))
            angle_diff = np.abs(angle - cur_theta)
            mask = angle_diff > (np.pi / 2)
            
            dist_apex = np.sqrt(np.sum((sample_points - cur_apex)**2, axis=1))
            res = mask * dist_apex + (1 - mask) * dist_apex * np.sin(angle_diff)
            radio = np.sum(res - eps) / res.shape[0]
            
            if best_radio < radio:
                best_radio = radio
                best_apex = cur_apex
                best_axis = cur_axis
                best_theta = cur_theta
        best_params = np.concatenate([best_apex, best_axis, np.array([best_theta])])
        return best_radio, best_params

    def fit_optimization(self, id, is_v_id, visualize=False):
        """Backward-compatible entry point using the robust cone fitter."""
        return self.fit_on_all_points(id, is_v_id, visualize=visualize)

    def fit(self, id, is_v_id):
        return self.fit_on_all_points(id, is_v_id)
                


class Torus:
    """Least-squares torus fitting for a 3-D point set.

    Parameters are returned as ``[cx, cy, cz, ax, ay, az, R, r]`` where the
    axis is unit length, ``R`` is the major radius and ``r`` the minor radius.
    """

    def __init__(self, v, f=None, f_normal=None, threshold=None):
        self.v = np.asarray(v, dtype=float)
        self.f = f
        self.f_normal = f_normal
        self.threshold = threshold

    def fit(self, id=None, is_v_id=True):
        if id is None:
            points = self.v
        elif is_v_id:
            points = self.v[_index_array(id)]
        else:
            face_ids = _index_array(id)
            points = self.v[np.unique(np.asarray(self.f)[face_ids].reshape(-1))]
        points = np.asarray(points, dtype=float)
        if points.shape[0] < 4 or _characteristic_scale(points) <= _EPS:
            return _failure(8)

        center = points.mean(axis=0)
        centered = points - center
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        rank_tolerance = max(singular_values[0], 1.0) * 100.0 * _EPS
        if singular_values[-1] <= rank_tolerance:
            return _failure(8)
        axis = vt[-1]
        axis /= max(np.linalg.norm(axis), np.finfo(float).eps)
        axial = centered @ axis
        radial = np.linalg.norm(centered - np.outer(axial, axis), axis=1)
        major = max(float(np.median(radial)), 1e-8)
        minor = max(float(np.median(np.sqrt((radial - major) ** 2 + axial ** 2))), 1e-8)

        # Axis is represented by an unconstrained vector and normalized in the
        # residual, avoiding singular angle parameterizations.
        x0 = np.r_[center, axis, major, minor]
        scale = max(np.linalg.norm(np.ptp(points, axis=0)), minor, 1e-8)

        def residual(params):
            c = params[:3]
            a = params[3:6]
            a = a / max(np.linalg.norm(a), np.finfo(float).eps)
            R, r = params[6:8]
            q = points - c
            z = q @ a
            rho = np.linalg.norm(q - np.outer(z, a), axis=1)
            return np.sqrt((rho - R) ** 2 + z ** 2) - r

        result = least_squares(
            residual,
            x0,
            bounds=(np.r_[[-np.inf] * 6, 1e-10, 1e-10], np.inf),
            max_nfev=3000,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return _failure(8)
        params = result.x.copy()
        params[3:6] /= max(np.linalg.norm(params[3:6]), np.finfo(float).eps)
        params[6:8] = np.abs(params[6:8])
        errors = np.abs(residual(params))
        tolerance = (
            self.threshold
            if self.threshold is not None
            else _distance_tolerance(points, 1e-2)
        )
        return float(np.mean(errors <= tolerance)), params


# Legacy prototype retained below for reference.
#     def __init__(self, v, f, v_normal):
#         self.v = v
#         self.f = f
#         self.v_normal = v_normal
        
#         self.threshold = 1e-1
    
    
#     def fit(self, id, is_v_id, weights=None, initial_guess: geom3d.Torus = None) -> geom3d.Torus:
#         if is_v_id:
#             points = self.v[id]
#         else:
#             v_id = np.unique(self.f[id]).flatten()
#             points = self.v[v_id]
            
#         if initial_guess is None:
#             initial_guess = geom3d.Torus(
#                 [0, 0, 0],
#                 [0, 0, 1],
#                 1, 0.1
#             )
        
#         def torus_fit_residuals(circle_params, points, weights):
#             circle = geom3d.Circle3D(
#                 circle_params[:3], circle_params[3:], np.linalg.norm(circle_params[3:])
#             )
#             distance = circle.distance_to_point(points)
#             radius = np.average(distance, weights=weights)
#             weights = np.sqrt(weights) if weights is not None else 1.0
#             return (distance - radius) * weights

#         x0 = np.concatenate(
#             [initial_guess.center, initial_guess.direction * initial_guess.major_radius]
#         )
#         results = optimize.least_squares(
#             torus_fit_residuals,
#             x0=x0,
#             args=(points, weights),
#             ftol=1e-10,
#             max_nfev=10000
#         )
#         if not results.success:
#             return RuntimeError(results.message)
        
#         circle = geom3d.Circle3D(
#             results.x[:3], results.x[3:], np.linalg.norm(results.x[3:])
#         )
#         distance = circle.distance_to_point(points)
#         minor_radius = np.average(distance, weights=weights)
#         # return geom3d.Torus(
#         #     results.x[:3],
#         #     results.x[3:],
#         #     np.linalg.norm(results.x[3:]),
#         #     minor_radius
#         # )
#         return results.x[:3], results.x[3:], np.linalg.norm(results.x[3:]), minor_radius
        
    
class _LegacyCylinder:
    def __init__(self, v, f, f_center, f_area, f_normal):
        self.v = v
        self.f = f
        
        # self.f_center = f_center
        # self.f_area = f_area
        # self.f_normal = f_normal
        # self.f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        
        self.n_sample_f_normal = 20
        self.f_center = np.mean(v[f], axis=1)
        self.f_area = igl.doublearea(v, f) / 2
        self.f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        self.f_normal = self.f_normal / np.linalg.norm(self.f_normal, axis=1).reshape(-1, 1)
        self.v_normal = igl.per_vertex_normals(v, f)
        
        
    def fit(self, id, is_v_id, visualize=False):
        if is_v_id:
            v_id = id
        else:
            v_id = np.unique(self.f[id].flatten())
        points = np.asarray(self.v[v_id])
        
        min_coord, max_coord = np.min(points, axis=0), np.max(points, axis=0)
        diagonal_len = np.linalg.norm(max_coord - min_coord)
        
        
        # get axis by pca
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        covariance = np.cov(points_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        pca_axis = eigenvectors[:, np.argmin(eigenvalues)]
        pca_axis /= np.linalg.norm(pca_axis)

        # get axis by normal
        if is_v_id:
            normals = self.v_normal[id]
        else:
            normals = self.f_normal[id]
        
        M = np.zeros((3, 3))
        for n in normals:
            M += np.outer(n, n)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        normal_axis = eigenvectors[:, 0]
        normal_axis /= np.linalg.norm(normal_axis)

        '''
        # normalize normals
        normals = normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)
        sample_axis = []
        sample_num = int(min(comb(normals.shape[0], 2), self.n_sample_f_normal))
        points_center = np.mean(points, axis=0)
        plot1 = mp.plot(self.v, self.f[id], shading={"wireframe": True})
        for _ in range(sample_num):
            index = np.random.choice(normals.shape[0], 2, replace=True)
            cur_normal = normals[index]
            cur_axis = np.cross(cur_normal[0], cur_normal[1])
            if np.linalg.norm(cur_axis) < 1e-3:
                continue
            cur_axis /= np.linalg.norm(cur_axis)
            sample_axis.append(cur_axis)
            
            plot1.add_lines(
                np.array([points_center]),
                np.array([points_center + cur_normal[0] * 10]),
                shading={
                    "line_color": "blue",
                    "line_width": 5
                }
            )
            plot1.add_lines(
                np.array([points_center]),
                np.array([points_center + cur_normal[1] * 10]),
                shading={
                    "line_color": "blue",
                    "line_width": 5
                }
            )
            plot1.add_lines(
                np.array([points_center]),
                np.array([points_center + cur_axis * 10]),
                shading={
                    "line_color": "red",
                    "line_width": 5
                }
            )
        normal_axis = []
        if len(sample_axis) >= 1:
            sample_axis = np.array(sample_axis)
            ms = sklearn.cluster.MeanShift()
            ms.fit(sample_axis)
            
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(sample_axis[:, 0], sample_axis[:, 1], sample_axis[:, 2])
            # ax.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1], ms.cluster_centers_[:, 2], c='r')
            # plt.show()
            
            for center in ms.cluster_centers_:
                center = center / np.linalg.norm(center)
                normal_axis.append(center)

            
            # sample_angles = np.zeros((sample_axis.shape[0], 2))
            # sample_angles[:, 0] = np.arccos(sample_axis[:, 2])
            # sample_angles[:, 1] = np.arctan2(sample_axis[:, 1], sample_axis[:, 0])

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(sample_angles[:, 0], sample_angles[:, 1])
            # ms = sklearn.cluster.MeanShift()
            # ms.fit(sample_angles)
            # cluster_centers = ms.cluster_centers_
            # ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r')
            # plt.show()
                        
            # for center in cluster_centers:
            #     axis = np.array([
            #         np.sin(center[0]) * np.cos(center[1]),
            #         np.sin(center[0]) * np.sin(center[1]),
            #         np.cos(center[0])
            #     ])
            #     normal_axis.append(axis)
        '''      
        
        if visualize:
            plot = mp.plot(self.v, self.f[id], shading={"wireframe": True})
            # show normal of all faces
            # plot.add_lines(
            #     np.array([self.f_center[id]]),
            #     np.array([self.f_center[id] + normals * 10]),
            #     shading={
            #         "line_color": "green",
            #         "line_width": 5
            #     }
            # )

            points_center = np.mean(points, axis=0)
            plot.add_lines(
                np.array([points_center]),
                np.array([points_center + pca_axis * 10]),
                shading={
                    "line_color": "blue",
                    "line_width": 5
                }
            )
            
            plot.add_lines(
                np.array([points_center]),
                np.array([points_center + normal_axis * 10]),
                shading={
                    "line_color": "red",
                    "line_width": 5
                }
            )
        
        # combine 2 lists to one
        candidate_axis = [pca_axis, normal_axis]
        # candidate_axis = [normal_axis]
        
        best_fit_rate = 0.
        best_params = []

        if visualize:
            p = mp.plot(self.v, self.f[id], shading={"wireframe": True})
        for i in range(len(candidate_axis)):
            axis_normal = candidate_axis[i]
        
            # get orthonormal basis
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(axis_normal, ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(axis_normal, ref)) > 0.9:
                ref = np.array([0.0, 0.0, 1.0])
            u = ref - np.dot(ref, axis_normal) * axis_normal
            u /= np.linalg.norm(u)
            v = np.cross(axis_normal, u)
            uv_basis = np.column_stack((u, v))
            
            # project to 2d        
            points_2d = self.project_points(points, axis_normal, uv_basis)
            # print(points_2d.shape)
            
            
            # fit 2d circle 
            center, radius = self.fit_circle(points_2d)
            
            # show 2d points by plt
            if visualize:
                if i == 0:
                    print('pca axis projection')
                else:
                    print('normal axis projection')
                plt.scatter(points_2d[:, 0], points_2d[:, 1])
                plt.scatter([center[0], center[0], center[0]+radius], [center[1], center[1]+radius, center[1]], c='red')
                plt.show()
            
            
            # reproject to 3d
            center = center @ uv_basis.T
            
            params = np.concatenate((
                center.reshape((-1, 1)), 
                axis_normal.reshape((-1, 1)), 
                np.array([radius]).reshape((-1, 1))
            ))

            
            # if radius > bbox diagonal len, thought to be wrong fitting
            if radius > diagonal_len * 2:
                continue
            
            # compute distance
            distance = self.dist_point_to_center(points, center, axis_normal)
            errors = abs(distance - radius)

            fit_rate = np.sum(errors < radius * 1e-2) / points.shape[0]
            # print(f'Axis {i}: Fit Rate: {fit_rate}, Center: {center}, Radius: {radius}')
            # color = 'red'
            
            
            
            if fit_rate > best_fit_rate:
                best_fit_rate = fit_rate
                best_params = params
                # color = 'green'
            
            a = 1
            # show results
            # if visualize:
            #     reference_vectors = [
            #         np.array([1.0, 0.0, 0.0]), 
            #         np.array([0.0, 1.0, 0.0]), 
            #         np.array([0.0, 0.0, 1.0])
            #     ]
            #     perp_vector = np.zeros(3)
            #     for ref_vector in reference_vectors:
            #         perp_vector = np.cross(axis_normal, ref_vector)
            #         if np.linalg.norm(perp_vector) > 1e-5:
            #             perp_vector = radius * perp_vector / np.linalg.norm(perp_vector)
            #             break
            #     p.add_lines(
            #         np.array([center]),
            #         np.array([center + perp_vector]),
            #         shading={
            #             "line_color": 'red',
            #             "line_width": 5
            #         }
            #     )

 
        return best_fit_rate, best_params

    def project_points(self, points, n, uv_basis):
            project_matrix = np.eye(3) - np.outer(n, n)
            project_points = points @ project_matrix
            points_2d = project_points @ uv_basis
            return points_2d

    def dist_point_to_center(self, points, c, n):
        distance = np.linalg.norm(np.cross(points - c, n), axis=1)
        return distance
    
    def fit_circle(self, points):
        x, y = points[:, 0], points[:, 1]
        n = points.shape[0]
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_xy = np.sum(x*y)
        sum_S = np.sum(x**2 + y**2)
        sum_S2 = np.sum((x**2 + y**2)**2)
        sum_xS = np.sum(x * (x**2 + y**2))
        sum_yS = np.sum(y * (x**2 + y**2))

        M11 = sum_S2 - (sum_S**2)/n
        M12 = sum_xS - (sum_x * sum_S)/n
        M13 = sum_yS - (sum_y * sum_S)/n
        M22 = sum_x2 - (sum_x**2)/n
        M23 = sum_xy - (sum_x * sum_y)/n
        M33 = sum_y2 - (sum_y**2)/n
        M = np.array([[M11, M12, M13],
                      [M12, M22, M23],
                      [M13, M23, M33]])
        
        N11 = 4 * sum_S
        N12 = 2 * sum_x
        N13 = 2 * sum_y
        N = np.array([[N11, N12, N13],
                      [N12, n, 0],
                      [N13, 0, n]])
        
        eigenvalues, eigenvectors = scipy.linalg.eig(M, N)
        min_idx = np.argmin(eigenvalues)
        a, b, c = np.real(eigenvectors[:, min_idx])
        d = -(a * sum_S + b * sum_x + c * sum_y) / n
        
        a += 1e-8 if a == 0 else 0
        B, C, D = b/a, c/a, d/a
        
        center = np.array([-B/2, -C/2])
        radius = np.sqrt(B**2 + C**2 - 4*D) / 2

        return center, radius
# v, f = igl.read_triangle_mesh('cylinder2.obj')
# cylinder_fit = _LegacyCylinder(v, f)
# cylinder_fit.fit()


class Cylinder:
    """Cylinder parameters are ``[axis_point, unit_axis, radius]``.

    ``axis_point`` is the point on the fitted axis closest to the coordinate
    origin. Candidate axes come from both surface normals and point PCA; each
    candidate is scored using radial distance and normal consistency.
    """

    def __init__(
        self,
        v,
        f,
        f_normal=None,
        f_center=None,
        f_area=None,
        v_normal=None,
    ):
        self.v = np.asarray(v, dtype=float)
        self.f = np.asarray(f, dtype=int)
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
        self.f_normal = (
            igl.per_face_normals(self.v, self.f, np.array([1.0, 0.0, 0.0]))
            if f_normal is None
            else np.asarray(f_normal, dtype=float)
        )
        self.v_normal = (
            igl.per_vertex_normals(self.v, self.f)
            if v_normal is None
            else np.asarray(v_normal, dtype=float)
        )
        self.threshold = 1e-2
        self.normal_angle = np.deg2rad(20.0)

    @staticmethod
    def _basis(axis):
        reference = np.eye(3)[np.argmin(np.abs(axis))]
        u = _unit(reference - np.dot(reference, axis) * axis)
        if u is None:
            return None
        return np.column_stack((u, np.cross(axis, u)))

    @staticmethod
    def _fit_circle(points):
        if len(points) < 3:
            return None, None
        initial_center = np.mean(points, axis=0)

        def residual(center):
            radii = np.linalg.norm(points - center, axis=1)
            return radii - np.mean(radii)

        result = least_squares(residual, initial_center, max_nfev=1000)
        center = result.x if result.success else initial_center
        radius = float(np.mean(np.linalg.norm(points - center, axis=1)))
        if not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= _EPS:
            return None, None
        return center, radius

    def fit(self, id, is_v_id, visualize=False):
        points, indices = _selected_points(self.v, self.f, id, is_v_id)
        if len(points) < 5 or _characteristic_scale(points) <= _EPS:
            return _failure(7)

        if is_v_id:
            sample_points = points
            sample_normals = self.v_normal[indices]
        else:
            sample_points = np.asarray(self.f_center[indices], dtype=float)
            sample_normals = np.asarray(self.f_normal[indices], dtype=float)
        measured_normals, valid_normals = _unit_rows(sample_normals)

        candidates = []
        if np.sum(valid_normals) >= 2:
            normal_matrix = measured_normals[valid_normals].T @ measured_normals[valid_normals]
            _, eigenvectors = np.linalg.eigh(normal_matrix)
            candidates.append(eigenvectors[:, 0])

        centered = points - np.mean(points, axis=0)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if len(singular_values) == 3:
            rank_tolerance = max(singular_values[0], 1.0) * 100.0 * _EPS
            if singular_values[-1] <= rank_tolerance:
                return _failure(7)
            candidates.extend(vh)

        unique_candidates = []
        for candidate in candidates:
            candidate = _unit(candidate)
            if candidate is None:
                continue
            if not any(abs(np.dot(candidate, other)) > 1.0 - 1e-6 for other in unique_candidates):
                unique_candidates.append(candidate)

        best_rate = 0.0
        best_rmse = np.inf
        best_params = None
        for axis in unique_candidates:
            basis = self._basis(axis)
            if basis is None:
                continue
            projected = points @ basis
            center_2d, radius = self._fit_circle(projected)
            if center_2d is None:
                continue
            axis_point = center_2d @ basis.T

            q = points - axis_point
            axial = q @ axis
            radial = np.linalg.norm(q - np.outer(axial, axis), axis=1)
            errors = np.abs(radial - radius)
            tolerance = max(
                self.threshold * radius,
                _distance_tolerance(points, 0.0),
            )
            point_rate = float(np.mean(errors <= tolerance))

            normal_rate = 1.0
            if np.any(valid_normals):
                q_normal = sample_points - axis_point
                axial_normal = q_normal @ axis
                predicted, valid_predicted = _unit_rows(
                    q_normal - np.outer(axial_normal, axis)
                )
                valid = valid_normals & valid_predicted
                if np.any(valid):
                    alignment = np.abs(
                        np.einsum("ij,ij->i", predicted[valid], measured_normals[valid])
                    )
                    normal_rate = float(
                        np.mean(alignment >= np.cos(self.normal_angle))
                    )
            rate = min(point_rate, normal_rate)
            rmse = float(np.sqrt(np.mean(errors**2)))
            if rate > best_rate or (np.isclose(rate, best_rate) and rmse < best_rmse):
                best_rate = rate
                best_rmse = rmse
                best_params = np.r_[axis_point, axis, radius]

        if best_params is None:
            return _failure(7)

        if visualize:
            faces = self.f[indices] if not is_v_id else self.f
            plot = mp.plot(self.v, faces, shading={"wireframe": True})
            extent = max(_characteristic_scale(points), best_params[6])
            plot.add_lines(
                np.array([best_params[:3] - best_params[3:6] * extent]),
                np.array([best_params[:3] + best_params[3:6] * extent]),
                shading={"line_color": "red", "line_width": 3},
            )

        return best_rate, best_params

        

    
def plane_eigen_fitting(X):
    centroid = np.mean(X, axis=0)
    X_centered = X - centroid
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    
    a, b, c = normal
    d = -np.dot(normal, centroid)
    
    errors = np.mean(
        np.abs(
            a*X[:,0] + b*X[:,1] + c*X[:,2] + d 
        )
    )
    print(errors)
    
    rho = d
    phi = np.arccos(c)
    theta = np.arctan2(b, a)
    
    return np.array([[rho, theta, phi]]), 0
    


def plane_HT0_search(X):    
    rho_init = np.max(np.linalg.norm(X, axis=1))
    
    rho_range = np.linspace(0, rho_init, 50)
    theta_range = np.deg2rad(np.arange(0, 359, 1))
    phi_range = np.deg2rad(np.arange(0, 181, 1))
    
    H = np.zeros((rho_range.shape[0], theta_range.shape[0], phi_range.shape[0]))
    threshold = rho_init / 200
    
    for i, rho in enumerate(rho_range):
        for j, theta in enumerate(theta_range):
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for k, phi in enumerate(phi_range):
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                errors = np.abs(
                    rho - X[:,0] * cos_theta * sin_phi - X[:,1] * sin_theta * sin_phi - X[:,2] * cos_phi
                )
                H[i, j, k] = np.sum(errors < threshold)

    max_coord = np.max(H)
    coords = np.argwhere(H == max_coord)
    
    result = []
    for idx in coords:
        result.append([rho_range[idx[0]], theta_range[idx[1]], phi_range[idx[2]]])

    return np.array(result), max_coord



class PointFittting:
    def __init__(self, v, f, f_center, f_area, f_normal):
        self.v = v
        self.f = f
        self.f_center = f_center
        self.f_area = f_area
        self.f_normal = f_normal
        
        


r'''
    params \in R^10
'''
class QuadricSurface:
    def __init__(self):
        pass
    
    def fit(self, points):
        
        M = np.zeros((10, 10))
        N = np.zeros((10, 10))
        _lambda = 0.1
        
        for p in points:
            x, y, z = p
            l = np.array([
                x**2, y**2, z**2, x*y, y*z, z*x, x, y, z, 1
            ])
            l_x = np.array([
                2*x, 0, 0, y, 0, z, 1, 0, 0, 0
            ])
            l_y = np.array([
                0, 2*y, 0, x, z, 0, 0, 1, 0, 0
            ])
            l_z = np.array([
                0, 0, 2*z, 0, y, x, 0, 0, 1, 0
            ])
            
            M += l * l.T
            N += l_x * l_x.T + l_y * l_y.T + l_z * l_z.T
            
        
        # compute eigenvector of min eigenvalue of M-lambda * N
        MAT = M - _lambda * N
        eigenvalues, eigenvectors = np.linalg.eigh(MAT)
        params = eigenvectors[:, np.argmin(eigenvalues)]
        
        return params
