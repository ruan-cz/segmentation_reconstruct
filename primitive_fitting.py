import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import igl
# from queue import PriorityQueue
# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# from distinctipy import distinctipy
import meshplot as mp
# import timeit
# import openmesh as om
import scipy
import sklearn
from scipy.special import comb

from utils import *
from fitting_geometric_primitives.geometry_primitive import *

    
class PrimitiveFitting:
    def __init__(self, v, f, f_normal=None, f_center=None, f_area=None):
        self.v = v
        self.f = f
        # self.f_normal = f_normal
        # self.f_center = f_center
        # self.f_area = f_area
        self.v_normal = igl.per_vertex_normals(v, f)

        self.f_center = np.mean(v[f], axis=1)
        self.f_area = igl.doublearea(v, f) / 2
        self.f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        self.f_normal = self.f_normal / np.linalg.norm(self.f_normal, axis=1).reshape(-1, 1)
    
    
    def fit_planar(self, f_id, is_v_id = False):
        plane = Plane(self.v, self.f, self.f_normal, self.f_area, self.f_center)
        return plane.fit(f_id, is_v_id)
        
        # if is_v_id:
        #     v_id = f_id
        # else:
        #     v_id = list(set(self.f[f_id].flatten()))
        # points = self.v[v_id]
        
        # centroid = np.mean(points, axis=0)
        # points_centered = points - centroid
        
        # covariance = np.cov(points_centered, rowvar=False)
        
        # eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # normal = eigenvectors[:, np.argmin(eigenvalues)]
        
        # a, b, c = normal
        # d = -np.dot(normal, centroid)
        # params = np.array([a, b, c, d])
        
        # distances = np.abs(a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / np.sqrt(a**2 + b**2 + c**2)
        # # fit = np.all(distances < 1e-3)
        # fit_rate = np.sum(distances < 5 * 1e-3) / len(v_id)
        # return fit_rate, params
        
        
    def fit_extrusion(self, f_id, is_v_id = False, visualize = False):
        extrusion = Extrusion(self.v, self.f)
        return extrusion.fit(f_id, is_v_id, visualize)
    
    def fit_cylinder(self, f_id, is_v_id = False, visualize=False):
        
        cylinder = Cylinder(self.v, self.f, self.f_normal, self.f_area, self.f_center)
        return cylinder.fit(f_id, is_v_id, visualize=False)
        
        if is_v_id:
            v_id = f_id
        else:
            v_id = list(set(self.f[f_id].flatten()))
        points = self.v[v_id]
        
        # get axis direction by mesh normal
        if is_v_id:
            normals = self.v_normal[v_id]
        else:
            normals = self.f_normal[f_id]
        sample_axis = []
        sample_num = int(min(comb(normals.shape[0], 3), 200))
        for _ in range(sample_num):
            index = np.random.choice(normals.shape[0], 3, replace=False)
            cur_normal = normals[index]
            cur_axis = np.cross(cur_normal[1] - cur_normal[0], cur_normal[2] - cur_normal[0])
            if np.linalg.norm(cur_axis) < 1e-3:
                continue
            cur_axis /= np.linalg.norm(cur_axis)
            sample_axis.append(cur_axis)
        if len(sample_axis) <= 1:
            return 0., 0
            
        sample_axis = np.array(sample_axis)
        sample_angles = np.zeros((sample_axis.shape[0], 2))
        sample_angles[:, 0] = np.arccos(sample_axis[:, 2])
        sample_angles[:, 1] = np.arctan2(sample_axis[:, 1], sample_axis[:, 0])
        
        ms = sklearn.cluster.MeanShift(
            cluster_all=False
        )
        ms.fit(sample_angles)
        cluster_centers = ms.cluster_centers_
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(sample_angles[:, 0], sample_angles[:, 1])
        # ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r')
        # plt.show()
        # print(cluster_centers)
        # axis_normal = np.array([
        #     np.sin(cluster_centers[0, 0]) * np.cos(cluster_centers[0, 1]),
        #     np.sin(cluster_centers[0, 0]) * np.sin(cluster_centers[0, 1]),
        #     np.cos(cluster_centers[0, 0])
        # ])


        
        
        def project_points(points, n, uv_basis):
            project_matrix = np.eye(3) - np.outer(n, n)
            project_points = points @ project_matrix
            points_2d = project_points @ uv_basis
            return points_2d


        def dist_point_to_center(points, c, n):
            distance = np.linalg.norm(np.cross(points - c, n), axis=1)
            return distance
        
        # plot = mp.plot(self.v, self.f, shading={"wireframe": True})
        best_fit_rate = 0.
        best_params = []
        for i in range(cluster_centers.shape[0]):
            axis_normal = np.array([
                np.sin(cluster_centers[i, 0]) * np.cos(cluster_centers[i, 1]),
                np.sin(cluster_centers[i, 0]) * np.sin(cluster_centers[i, 1]),
                np.cos(cluster_centers[i, 0])
            ])
        
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
            points_2d = project_points(points, axis_normal, uv_basis)
            
            # find circle, reproject to 3d
            center, radius = self.fit_circle(points_2d)
            center = center @ uv_basis.T
            
            params = np.concatenate((
                center.reshape((-1, 1)), 
                axis_normal.reshape((-1, 1)), 
                np.array([radius]).reshape((-1, 1))
            ))
            
            # compute distance
            distance = dist_point_to_center(points, center, axis_normal)
            errors = abs(distance - radius)

            # fit = np.mean(errors) < 1e-2
            # fit = np.all(errors < 2.)
            fit_rate = np.sum(errors < 1e-1) / points.shape[0]
            # print(fit_rate)
            # print(np.sum(errors < 1e-2) / points.shape[0])
            
            if fit_rate > best_fit_rate:
                best_fit_rate = fit_rate
                best_params = params
        
        
            # points_center = np.mean(points, axis=0)
            # plot.add_lines(
            #     np.array([points_center]),
            #     np.array([points_center + axis_normal * 10]),
            #     shading={
            #         "line_color": "red",
            #         "line_width": 5
            #     }
            # )

        # get axis normal by PCA of all points
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        covariance = np.cov(points_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis_normal_pca = eigenvectors[:, np.argmax(eigenvalues)]
        axis_normal_pca = np.array(axis_normal)
        axis_normal_pca /= np.linalg.norm(axis_normal)

        # get orthonormal basis
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis_normal_pca, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(axis_normal_pca, ref)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        u = ref - np.dot(ref, axis_normal_pca) * axis_normal_pca
        u /= np.linalg.norm(u)
        v = np.cross(axis_normal_pca, u)
        uv_basis = np.column_stack((u, v))
        
        # project to 2d        
        points_2d = project_points(points, axis_normal_pca, uv_basis)
        
        # find circle, reproject to 3d
        center, radius = self.fit_circle(points_2d)
        center = center @ uv_basis.T
        
        params = np.concatenate((
            center.reshape((-1, 1)), 
            axis_normal_pca.reshape((-1, 1)), 
            np.array([radius]).reshape((-1, 1))
        ))
        
        # compute distance
        distance = dist_point_to_center(points, center, axis_normal_pca)
        errors = abs(distance - radius)

        # fit = np.mean(errors) < 1e-2
        # fit = np.all(errors < 2.)
        fit_rate = np.sum(errors < 1e-1) / points.shape[0]
        # print(fit_rate)
        # print(np.sum(errors < 1e-2) / points.shape[0])
        
        if fit_rate > best_fit_rate:
            best_fit_rate = fit_rate
            best_params = params

        print(best_fit_rate, best_params)
        return best_fit_rate, best_params
        # return fit_rate, params
    
    
    
    def fit_circle(self, points):
        '''
            Taubin's method
        '''
        
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
        

    def fit_sphere(self, f_id, is_v_id = False):
        sphere = Sphere(self.v, self.f, self.f_normal, self.f_area, self.f_center)
        # return sphere.optmize_fit(f_id, is_v_id)
        return sphere.fit(f_id, is_v_id)
        
        # if is_v_id:
        #     v_id = f_id
        # else:
        #     v_id = list(set(self.f[f_id].flatten()))
        # points = self.v[v_id]
        
        
        # '''
        # x^2 + y^2 + z^2 + A^2 + B^2 + C^2 - 2 * (Ax + By + Cz) - r^2 = 0
        # ==> (2x, 2y, 2z, 1) (A, B, C, r^2 - A^2 - B^2 - C^2).T = x^2+y^2+z^2
        # ==> Aw = b, w = (A^TA)^{-1}b
        # '''
        
        # A = np.concatenate((
        #     2 * points,
        #     np.ones((points.shape[0], 1))
        # ), axis=1)
        # b = points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2

        # w = np.linalg.inv(A.T @ A) @ A.T @ b
        
        # A, B, C, r2 = w
        # r = np.sqrt(r2 + A**2 + B**2 + C**2)

        # params = np.array([A, B, C, r])
        
        # # compute distance
        # distance = np.linalg.norm(points - np.array([A, B, C]), axis=1)
        # errors = abs(distance - r)

        # # fit = np.mean(errors < 1e-2)
        # fit_rate = np.sum(errors < 1e-2) / points.shape[0]
        
        # return fit_rate, params
        
    def fit_cone(self, f_id, is_v_id = False):
        cone = Cone(self.v, self.f)
        # return cone.fit(f_id, is_v_id)
        return cone.fit_on_all_points(f_id, is_v_id)
        # return cone.fit_optimization(f_id, is_v_id)
         
        
    def fit_torus(self, f_id, is_v_id = False):
        if is_v_id:
            v_id = f_id
        else:
            v_id = list(set(self.f[f_id].flatten()))
        
        points = self.v[v_id]
        
        
        
        
    
        
    