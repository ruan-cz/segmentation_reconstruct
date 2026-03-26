# import scipy.optimize
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize
import sklearn
import igl
import meshplot as mp
import scipy
import sklearn
from scipy.special import comb
# mp.offline()
import open3d as o3d

from sklearn.neighbors import NearestNeighbors

def estimate_normals(points, k=30):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros_like(points)

    for i in range(points.shape[0]):
        neighbors = points[indices[i]]

        # center neighborhood
        neighbors = neighbors - neighbors.mean(axis=0)

        # covariance
        cov = neighbors.T @ neighbors

        # eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # smallest eigenvector = normal
        normals[i] = eigvecs[:, 0]

    return normals

def vector_same_direction(v1, v2):
    # v1, v2 = np.array(v1).reshape()
    thredhold_cross = 1e-2
    # print(v1.shape, v2.shape)
    cross = np.cross(v1, v2)
    
    return np.all(np.abs(cross) < thredhold_cross)
    

def is_plane_same(plane1, plane2):
    n1, d1 = plane1[:3].squeeze(), float(plane1[3])
    n2, d2 = plane2[:3].squeeze(), float(plane2[3])

    if not vector_same_direction(n1, n2):
        return False

    tol = 1e-3
    idx = np.argmax(np.abs(n2))
    lam = n1[idx] / n2[idx]

    return np.allclose(n1, lam*n2, atol=tol) and np.isclose(d1, lam*d2, atol=tol)



def is_cylinder_same(cylinder1, cylinder2):
    c1, axis1, r1 = cylinder1[:3].squeeze(), cylinder1[3:6].squeeze(), float(cylinder1[6])
    c2, axis2, r2 = cylinder2[:3].squeeze(), cylinder2[3:6].squeeze(), float(cylinder2[6])

    threshold_r = 1e-1
    # same r -> same axis -> same diff(c1, c2) & axis
    if not (np.abs(r1 - r2) < threshold_r):
        return False
    
    if not vector_same_direction(axis1, axis2):
        return False
    
    if not vector_same_direction(axis1, c1 - c2):
        return False

    return True

def get_bbox_diagnal(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    diagnal = np.linalg.norm(max_bound - min_bound)
    return diagnal


# not need f
class Plane:
    def __init__(self, v, f=None, f_normal=None, f_area=None, f_center=None):
        self.v = v
        self.f = f
        if f is not None:
            self.f_normal = f_normal
            self.f_area = f_area
            self.f_center = f_center
        
        self.threshold = 1e-2 * get_bbox_diagnal(v)

    def fit(self, id, is_v_id):
        if is_v_id:
            points = np.asarray(self.v[id])
        else:
            v_id = np.unique(self.f[id].flatten())
            points = np.asarray(self.v[v_id])
        
        
            
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        cov = np.cov(centered_points, rowvar=False)
        eigen_value, eigen_vector = np.linalg.eig(cov)
        normal = eigen_vector[:, np.argmin(eigen_value)]
        
        a, b, c = normal
        d = -np.dot(normal, centroid)
        param = np.array([a, b, c, d])
        
        dist = np.abs(
            (normal[0] * points[:, 0] + normal[1] * points[:, 1] + normal[2] * points[:, 2] + d) / np.linalg.norm(normal) 
        )
        fit_rate = np.sum(dist < self.threshold) / points.shape[0]

        return fit_rate, param



# not need face
class Sphere:
    def __init__(self, v, f=None, f_normal=None, f_area=None, f_center=None):
        self.v = v
        self.f = f
        self.f_normal = f_normal
        self.f_area = f_area
        self.f_center = f_center
        
        self.threshold = 1e-1 * get_bbox_diagnal(v)

    def optmize_fit(self, id, is_v_id):
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
        if is_v_id:
            points = np.asarray(self.v[id])
        else:
            v_id = np.unique(self.f[id].flatten())
            points = np.asarray(self.v[v_id])
            
        min_coord, max_coord = np.min(points, axis=0), np.max(points, axis=0)
        diagonal_len = np.linalg.norm(max_coord - min_coord)
        # print(f'diagonal {diagonal_len}')
        

        
            
        n_points = points.shape[0]

        eps = 10*1e-2
        sample_radio = 1e-1
        max_iter = 100
        
        best_center = []
        best_radius = 0.
        if n_points < 100:
            px, py, pz = points[:, 0], points[:, 1], points[:, 2]
            A = np.zeros((n_points, 4))
            A[:, 0] = px * 2
            A[:, 1] = py * 2
            A[:, 2] = pz * 2
            A[:, 3] = 1
            b = np.zeros((n_points, 1))
            b[:, 0] = px * px + py * py + pz * pz
            C, resid1, rank1, s1 = np.linalg.lstsq(A, b, rcond=None)
            
            radius_pow = C[0]**2 + C[1]**2 + C[2]**2 + C[3]
            if radius_pow <= 0:
                return 0, np.array([0., 0., 0., 0.])
            center = np.array([C[0], C[1], C[2]]).reshape(1, 3)
            radius = np.sqrt(radius_pow)[0]
            
            if radius > diagonal_len * 2:
                return 0, np.array([0., 0., 0., 0.])
                
            
            # return center, radius
            best_center, best_radius = center, radius
        else:
            best_inliers_len = 0
            for _ in range(max_iter):
                sample_num = int(sample_radio * n_points)
                sample_id = np.random.choice(n_points, sample_num)
                sample_points = points[sample_id]
    
                px, py, pz = sample_points[:, 0], sample_points[:, 1], sample_points[:, 2]
                A = np.zeros((sample_num, 4))
                A[:, 0] = px * 2
                A[:, 1] = py * 2
                A[:, 2] = pz * 2
                A[:, 3] = 1
                b = np.zeros((sample_num, 1))
                b[:, 0] = px * px + py * py + pz * pz
                C, resid1, rank1, s1 = np.linalg.lstsq(A, b, rcond=None)
                
                radius_pow = C[0]**2 + C[1]**2 + C[2]**2 + C[3]
                if radius_pow <= 0:
                    continue
                center = np.array([C[0], C[1], C[2]]).reshape(1, 3)
                radius = np.sqrt(radius_pow)[0]
                
                # if radius > diagonal_len:
                #     continue
                
                dist_pt = np.linalg.norm(points - center, axis=1)
                radiu_threshold = eps
                cur_inliers_len = len(np.where(np.abs(dist_pt - radius) <= radiu_threshold)[0])
                if cur_inliers_len > best_inliers_len:
                    best_inliers_len = cur_inliers_len
                    best_center = center
                    best_radius = radius
        
        if len(best_center) == 0:
            return 0, np.array([0., 0., 0., 0.])
        
        
        best_params = np.array([
            best_center[0, 0], 
            best_center[0, 1], 
            best_center[0, 2], 
            best_radius
        ])
        dist_pt = np.linalg.norm(points - best_center, axis=1)
        dist_r = np.abs(dist_pt - best_radius)
        fit_rate = np.sum(dist_r < eps * best_radius) / n_points

            
        return fit_rate, best_params
            
                
class Extrusion:
    def __init__(self, v, f):
        self.v, self.f = v, f
        self.v_normal = igl.per_vertex_normals(v, f)
        self.f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
    
        self.threshold_normal_planar = 5 * 1e-2


    def fit_normal_to_plane(self, normals):
        centroid = np.mean(normals, axis=0)
        centroid_n = normals - centroid

        cov = np.cov(centroid_n, rowvar=False)
        eigen_value, eigen_vector = np.linalg.eig(cov)
        
        n = eigen_vector[:, np.argmin(eigen_value)]
        a, b, c = n
        d = -np.dot(n, centroid)
        param = np.array([a, b, c, d])

        # fit rate
        dist = np.abs(
            (n[0] * normals[:, 0] + n[1] * normals[:, 1] + n[2] * normals[:, 2] + d) / np.linalg.norm(n)
        )
        fit_rate = np.sum(dist < self.threshold_normal_planar) / normals.shape[0]

        return fit_rate, param
    
    def fit(self, id, is_v_id, visualize=False):
        if is_v_id:
            points = np.asarray(self.v[id])
            normals = self.v_normal[id]
        else:
            v_id = np.unique(self.f[id].flatten())
            points = np.asarray(self.v[v_id])
            normals = self.f_normal[id]


        # visualize = True
        
        # normals on same plane on Gauss sphere
        normal_plane_rate, normal_plane_param = self.fit_normal_to_plane(normals)

        # if visualize:
        #     plot = mp.plot(normals)

        if normal_plane_rate <= 0.9:
            return 0, []
        
        return normal_plane_rate, normal_plane_param[:3].squeeze()



class Cone:
    def __init__(self, v, f=None):
        self.v = v
        self.f = f
        self.v_normal = igl.per_vertex_normals(v, f)
        
        self.threshold = 1e-2

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
    

    def fit_optimization(self, id, is_v_id, visualize=False):
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
            print(vec.shape, dot.shape, axis.shape)
            v_perp = vec - dot[:, np.newaxis] * axis
            r = np.linalg.norm(v_perp, axis=1)
            
            print()
            r_expected = dot * np.tan(theta)
            
            distances = np.abs(r - r_expected)
            fit_rate = np.sum(distances < self.threshold) / points.shape[0]
            return fit_rate, np.concatenate(p, axis, theta)
        else:
            # raise RuntimeError("Optimization failed")
            return 0, []
        
    
    def fit_on_all_points(self, id, is_v_id, visualize=False):
        if is_v_id:
            points = self.v[id]
            points_normal = self.v_normal[id]
        else:
            v_id = np.unique(self.f[id].flatten())
            points = self.v[v_id]
            points_normal = self.v_normal[v_id]
        
        n_points = points.shape[0]
        
        # apex: ni * (p_i - apex) = 0
        b = np.zeros(n_points)
        b[:] = points_normal[:, 0] * points[:, 0] + points_normal[:, 1] * points[:, 1] + points_normal[:, 2] * points[:, 2]


            
        try:
            apex, residuals, rank, s = np.linalg.lstsq(points_normal, b)
        except:
            if visualize:
                if is_v_id == False:
                    plot = mp.plot(self.v, self.f[id], np.ones((self.f[id].shape[0], 3)))
                    plot.add_lines(
                        self.v[np.unique(self.f[id].flatten())],
                        self.v[np.unique(self.f[id].flatten())] + self.v_normal[np.unique(self.f[id].flatten())] * 5,
                        shading={
                            'line_width': 3,
                            'line_color': 'red'
                        }
                    )    
                # print('ERROR: apex fitting')
            return 0, []

        initial_axis = np.array([0, 0, 1])
        initial_theta = np.pi / 8
        initial_params = np.concatenate([initial_axis, [initial_theta]])
        
        # Constraints: axis should be normalized
        constraints = {
            'type': 'eq', 
            'fun': lambda params: np.linalg.norm(params[:3]) - 1
        }
        
        # Optimize
        result = minimize(
            self.objective_function,
            initial_params,
            args=(points, apex),
            constraints=constraints,
            method='SLSQP',
            options={'disp': False}
        )
        
        if result.success:
            axis = result.x[:3]
            axis /= np.linalg.norm(axis)
            theta = result.x[3]

            vec = points - apex
            dot = np.dot(vec, axis)
            print(vec.shape, dot.shape, axis.shape)
            
            v_perp = vec - dot[:, np.newaxis] * axis
            r = np.linalg.norm(v_perp, axis=1)
            r_expected = dot * np.tan(theta)
            
            distances = np.abs(r - r_expected)
            fit_rate = np.sum(distances < self.threshold) / points.shape[0]
            return fit_rate, np.concatenate([apex, axis, [theta]])
        else:
            return 0, []
        
        # axis: circle axis fitting
        vec = points - apex
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        # fit vec to plane
        vec = vec - np.mean(vec, axis=0)
        cov = np.cov(vec, rowvar=False)
        eigen_value, eigen_vector = np.linalg.eig(cov)
        axis = eigen_vector[:, np.argmin(eigen_value)]
        axis = np.squeeze(axis)
        
        # theta: mean angle<axis, vec>
        angles = np.arccos(np.clip(np.dot(vec, axis), -1, 1))
        theta = np.mean(angles) * 0.5
        
        params = np.array([
            apex[0], apex[1], apex[2],
            axis[0], axis[1], axis[2],
            theta
        ])

        plot = mp.plot(self.v, self.f[id], np.ones((self.f[id].shape[0], 3)))
        plot.add_points(
            np.array([apex]),
            shading={
                'point_size': 5,
                'point_color': 'red'
            }
        )
        plot.add_lines(
            np.array([apex]),
            np.array([apex]) + np.array([axis]) * 5,
            shading={
                'line_width': 3,
                'line_color': 'red'
            }
        )
        
        
        # compute distance
        distances = np.array([
            self.point_to_cone(point, params[:3], params[3:6], params[6]) for point in points
        ])
        fit_rate = np.sum(distances < self.threshold) / points.shape[0]
        # threshold_angle = 1 * 1e-1
        # dis_angle = np.abs(angles - theta)
        # fit_rate = np.sum(dis_angle < threshold_angle) / points.shape[0]

        
        return fit_rate, params
        
        
    
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
        
        
        
    def fit(self, id, is_v_id):
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
                


# class Torus:
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
        
# not need face  
class Cylinder:
    def __init__(self, v, f=None, f_center=None, f_area=None, f_normal=None):
        self.v = v
        self.f = f
        
        self.v_normal = estimate_normals(v)
        # self.f_center = f_center
        # self.f_area = f_area
        # self.f_normal = f_normal
        # self.f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        
        # self.n_sample_f_normal = 20
        # self.f_center = np.mean(v[f], axis=1)
        # self.f_area = igl.doublearea(v, f) / 2
        # self.f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        # self.f_normal = self.f_normal / np.linalg.norm(self.f_normal, axis=1).reshape(-1, 1)
        # self.v_normal = igl.per_vertex_normals(v, f)
        
        
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
# cylinder_fit = Cylinder(v, f)
# cylinder_fit.fit()        

        

    
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