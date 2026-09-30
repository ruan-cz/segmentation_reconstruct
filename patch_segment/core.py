import numpy as np
import igl
from queue import PriorityQueue
import meshplot as mp
import openmesh as om
import trimesh
import os
import json

from utils.mesh_utils import *
from primitive_fitting.fitting import *

class Distance():
    def __init__(self, v, f):
        self.v = v
        self.f = f
        self.v_num = v.shape[0]
        self.f_num = f.shape[0]
        
        self.initialize(v, f)
        
    
    def initialize(self, v, f):
        self.face_area = igl.doublearea(v, f) / 2.
        self.face_center = np.mean(v[f], axis=1)
        self.face_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        self.face_normal = self.face_normal / np.linalg.norm(self.face_normal, axis=1).reshape(-1, 1)

    
    # INPUT
    #   - proxy center, proxy normal, point
    # OUTPUT
    #   - orthogonal distance:
    #       OrthoD(v, plane(center, normal)) =  |dot(v-center, normal)| / len(normal)
    def orthogonal_distance(self, center, normal, v):
        len = np.linalg.norm(normal)
        dist = abs((v - center).dot(normal)) / len
        return dist

    
    # L2 = 1/6 * area * (d1^2 + d2^2 + d3^2 + d1d2 + d2d3 + d3d1)
    def distance_L2(self, f_id, center, normal):
        v1, v2, v3 = self.v[self.f[f_id]]
        area = self.face_area[f_id]
        
        d1 = self.orthogonal_distance(center, normal, v1)
        d2 = self.orthogonal_distance(center, normal, v2)
        d3 = self.orthogonal_distance(center, normal, v3)
        
        distance = 1. / 6 * area * (d1*d1 + d2*d2 + d3*d3 + d1*d2 + d2*d3 + d3*d1)
        return distance
    
    
    # L21: area * ||n(f_id) - normal||^2
    def distance_L21(self, f_id, normal):
        n = np.linalg.norm(self.face_normal[f_id] - normal)
        distance = self.face_area[f_id] * n * n
        return distance
    
    
    def distance(self, f_id, proxy_center, proxy_normal, norme):
        if norme == 0:
            return self.distance_L2(f_id, proxy_center, proxy_normal)
        else:
            return self.distance_L21(f_id, proxy_normal)
    
    
    def distance_error(self, R, proxy_center, proxy_normal, norme):
        error = 0.
        
        for i in range(self.f_num):
            proxy_id = R[i]
            error += self.distance(i, proxy_center[proxy_id], proxy_normal[proxy_id], norme)
        
        return error
    
    def flattern_error(self, R, proxy_center, proxy_normal, norme):
        error = 0.
        for i in range(self.f_num):
            proxy_id = R[i]
            cur_normal = proxy_normal[proxy_id]
            f_area = self.face_area[i]
            f_normal = self.face_normal[i]
            error += f_area * np.linalg.norm(cur_normal - f_normal)
        return error

class Partition():
    def __init__(self, v, f, _distance):
        self.v = v
        self.f = f
        self.v_num = v.shape[0]
        self.f_num = f.shape[0]
        self.distance = _distance
        
        self.compute_f_f_adjacency()
        
        self.n_colors = colors
        self.cur_colors = np.ones((self.f_num, 3), dtype=float)
        # self.mesh = mp.plot(self.v, self.f, c=self.cur_colors)        
        
        
    def draw_proxy_on_face(self, proxy_on_face):
        
        for i in range(proxy_on_face.shape[0]):
            self.cur_colors[proxy_on_face[i]] = self.n_colors[i]

        self.mesh.update_object(colors=self.cur_colors)


    def compute_f_f_adjacency(self):
        v_num, f_num = self.v_num, self.f_num
        self.f_f_adjacency = np.zeros((f_num, 3))
        edge = {}
        
        for i in range(f_num):
            v_id = self.f[i]
            v_id.sort()
            e_id = [v_id[0] + v_num * v_id[1], v_id[0] + v_num * v_id[2], v_id[1] + v_num * v_id[2]]
            for j in range(3):
                if e_id[j] in edge.keys():
                    n_f_id = edge[e_id[j]]
                    self.f_f_adjacency[i][j] = n_f_id

                    n_f_v_id = self.f[n_f_id]
                    n_f_v_id.sort()
                    n_f_e_id = [n_f_v_id[0] + v_num * n_f_v_id[1], n_f_v_id[0] + v_num * n_f_v_id[2], n_f_v_id[1] + v_num * n_f_v_id[2]]
                    k = (e_id[j]==n_f_e_id[1]) + 2 * (e_id[j]==n_f_e_id[2])
                    self.f_f_adjacency[n_f_id][k] = i
                else:
                    edge[e_id[j]] = i   
        
        return self.f_f_adjacency
        
        
    def region_growing(self, proxy_on_face, proxy_center, proxy_normal, norme, find_furthest = False):
        f_num = self.f_num
        R = - np.ones(f_num, dtype=int)
        
        proxy_num = proxy_on_face.shape[0]

        furthest_distance = np.zeros(proxy_num, dtype=float)
        furthest_triangle = np.zeros(proxy_num, dtype=int)
        q = PriorityQueue()
        for proxy_id in range(proxy_num):
            f_id = proxy_on_face[proxy_id]
            R[f_id] = proxy_id
            for i in range(3): 
                n_f_id = int(self.f_f_adjacency[f_id][i])
                d = self.distance.distance(n_f_id, proxy_center[proxy_id], proxy_normal[proxy_id], norme)
                q.put([d, n_f_id, proxy_id])
                if find_furthest & (d > furthest_distance[proxy_id]) & (n_f_id not in proxy_on_face):
                    furthest_distance[proxy_id] = d
                    furthest_triangle[proxy_id] = n_f_id
        
        while not q.empty():
            d, f_id, proxy_id = q.get()
            if R[f_id] == -1:
                R[f_id] = proxy_id
                for i in range(3):
                    n_f_id = int(self.f_f_adjacency[f_id][i])
                    d = self.distance.distance(n_f_id, proxy_center[proxy_id], proxy_normal[proxy_id], norme)
                    q.put([d, n_f_id, proxy_id])
                    if find_furthest & (d > furthest_distance[proxy_id]) & (n_f_id not in proxy_on_face):
                        furthest_distance[proxy_id] = d
                        furthest_triangle[proxy_id] = n_f_id
        max_triangle = furthest_triangle[np.argmax(furthest_distance)]   
   
        return R, max_triangle
    
     
    def initial_proxy_random(self, proxy_num):
        proxy = []
        while len(proxy) < proxy_num:
            f_id = np.random.randint(0, self.f_num)
            if f_id not in proxy:
                proxy.append(f_id)
        proxy = np.array(proxy)
        return proxy
    
    
    def inital_proxy_uniform(self, proxy_num):
        proxy = []
        for i in range(proxy_num):
            proxy.append(int(i * self.f_num / proxy_num))
        return proxy
    
    
    def find_closest_triangle(self, pre_R, proxy_center, proxy_normal, norme):
        proxy_num = proxy_center.shape[0]
        
        # init as max_float distance
        closest_distance = np.array([np.inf] * proxy_num, dtype=float)
        closest_triangle = np.zeros(proxy_num, dtype=int)
        
        for f_id in range(self.f_num):
            pre_proxy_id = pre_R[f_id]
            d = self.distance.distance(f_id, proxy_center[pre_proxy_id], proxy_normal[pre_proxy_id], norme)
            if d < closest_distance[pre_proxy_id]:
                closest_distance[pre_proxy_id] = d
                closest_triangle[pre_proxy_id] = f_id
        
        return closest_triangle
       
        
    def region_growing_proxy_initialized(self, pre_R, proxy_center, proxy_normal, norme):
        proxy_on_face = self.find_closest_triangle(pre_R, proxy_center, proxy_normal, norme)
        proxy_center = self.distance.face_center[proxy_on_face]
        proxy_normal = self.distance.face_normal[proxy_on_face]
        R, _ = self.region_growing(proxy_on_face, proxy_center, proxy_normal, norme)
        return R
    
    
    def region_growing_triangle_initialized(self, proxy_on_face, norme, find_furthest = False):
        proxy_center = self.distance.face_center[proxy_on_face]
        proxy_normal = self.distance.face_normal[proxy_on_face]
        R, max_triangle = self.region_growing(proxy_on_face, proxy_center, proxy_normal, norme, find_furthest)
        
        return R, max_triangle
    
    
    def initial_partition(self, proxy_num, norme):
        proxy_on_face = self.initial_proxy_random(proxy_num)
        # self.draw_proxy_on_face(np.array(proxy_on_face))
        R, max_triangle = self.region_growing_triangle_initialized(proxy_on_face, norme)
        return R, [max_triangle]
    
        
    def initial_partition_furthest(self, proxy_num, norme):
        triangle = []
        tri = np.random.randint(0, self.f_num)
        for i in range(proxy_num):
            # print("[Initial Furthest Partition:{}/{}]".format(i, proxy_num), end="\r")
            triangle.append(tri)
            R, tri = self.region_growing_triangle_initialized(np.array(triangle), norme, find_furthest=True)

            # self.draw_proxy_on_face(np.array(triangle))

        return R, triangle
    
    
    def insert_proxy(self, triangle, norme):
        R, tri = self.region_growing_triangle_initialized(np.array(triangle), norme, find_furthest=True)
        triangle.append(tri)
        R, _ = self.region_growing_triangle_initialized(np.array(triangle), norme)
        return R, triangle
        
    
    
    # INPUT: R, proxies of faces, num_f * 1
    # OUTPUT: cluster of proxies, {num_proxy: []}
    def partition_to_cluster(self, R):
        cluster = {}
        for f_id in range(R.shape[0]):
            proxy_id = R[f_id]
            if proxy_id not in cluster.keys():
                cluster[proxy_id] = [f_id]
            else:
                cluster[proxy_id].append(f_id)
                
        return cluster

class Proxy():
    def __init__(self, v, f, _distance):
        self.v = v
        self.f = f
        self.v_num = v.shape[0]
        self.f_num = f.shape[0]
        self.distance = _distance
        
    
    # area, area-weighted center, area-weighted normal of each proxy
    def weights_of_proxy(self, R, proxy_num):
        # sum of area of faces in each proxy
        weight_sum = np.zeros(proxy_num)
        # sum of area-weighted center in each proxy
        weighted_center_sum = np.zeros((proxy_num, 3))
        # sum of area-weighted normal in each proxy
        weighted_normal_sum = np.zeros((proxy_num, 3))

        # load area, normal from distance
        f_area = self.distance.face_area
        f_normal = self.distance.face_normal
        f_center = self.distance.face_center

        for i in range(self.f_num):
            
            area = f_area[i]
            center = f_center[i]
            normal = f_normal[i]
            
            proxy_id = R[i]
            weight_sum[proxy_id] += area
            weighted_center_sum[proxy_id] += area * center
            weighted_normal_sum[proxy_id] += area * normal
        
        return weight_sum, weighted_center_sum, weighted_normal_sum
        

    # input: proxy of faces, proxy num
    # eigenvector of smallest eigenvalue of MAT
    # MAT =  \sum( 2 * area / 72 * Mi M_symmetric * Mi^T + area * f_center * f_center^T) - area_sum * new_center * new_center.T
    # Mi = (v2 - v1 | v3 - v1 | 0)
    def new_proxy_L2(self, R, proxy_num):
        new_center = self.weighted_center_sum / self.weight_sum.reshape(-1, 1)
        
        MAT = np.zeros((proxy_num, 3, 3))
        M_symmetric = np.matrix([[10, 7, 0], [7, 10, 0], [0, 0, 0]])
        for i in range(self.f_num):
            area = self.distance.face_area[i]
            v1, v2, v3 = self.v[self.f[i]]
            Mi = np.matrix([v2 - v1, v3 - v1, [0, 0, 0]])
            f_center = self.distance.face_center[i]
            
            proxy_id = R[i]
            MAT[proxy_id] += 2 * area / 72 * Mi * M_symmetric * Mi.T + area * np.outer(f_center, f_center)
        
        for proxy_id in range(proxy_num):
            MAT[proxy_id] -= self.weight_sum[proxy_id] * np.outer(new_center[proxy_id], new_center[proxy_id])
        
        # get eigenvector of smallest eigenvalue of MAT
        eigenvalue, eigenvector = np.linalg.eig(MAT)
        new_normal = np.take_along_axis(eigenvector, np.argmin(eigenvalue, axis=1)[:, None, None], axis=1).reshape((200, 3))
        # new_normal = np.linalg.norm(new_normal, axis=1)
        # new_normal = np.linalg.norm(eigenvector[:, np.argmin(eigenvalue, axis=1), :])

        return new_center, new_normal
        
    
    # area-weighted normal
    def new_proxy_L21(self):
        new_center = self.weighted_center_sum / self.weight_sum.reshape(-1, 1)
        new_normal = self.weighted_normal_sum / self.weight_sum.reshape(-1, 1)
        return new_center, new_normal


    # input: proxy of faces, proxy num, norme
    def new_proxy(self, R, proxy_num, norme):
        self.weight_sum, self.weighted_center_sum, self.weighted_normal_sum = self.weights_of_proxy(R, proxy_num)
        
        if norme == 0:
            return self.new_proxy_L2(R, proxy_num)
        else:
            return self.new_proxy_L21()

vsa_obj_id = 0

class VSA():
    def __init__(self, v, f, proxy_num=200, norme=1, max_iters=100):
        self.v = v
        self.f = f
        self.norme = norme
        self.proxy_num = proxy_num
        self.f_num = f.shape[0]
        self.v_num = v.shape[0]
        

        self.max_iters = max_iters
        self.iter_diff_radio_threhold = 1 * 1e-3
        self.iter_diff_threhold = 1 * 1e-2
        # self.threshold_convexity = np.cos(87.5 * np.pi / 180)
        
        self.init_modules()
        
        global vsa_obj_id
        vsa_obj_id += 1
    
    
    def init_modules(self):
        self.distance = Distance(self.v, self.f)
        self.partition = Partition(self.v, self.f, self.distance)
        self.proxy = Proxy(self.v, self.f, self.distance)
    

    # OUTPUT:
    #   min_R: proxies of faces, f_num * 1
    #   proxy_cluster: cluster of proxies, {proxy_id: []}
    def iteration(self, visualize=False):
        all_errors = []
        all_R = []
        cur_iteration = 0
        cur_error = 0.
        precedent_error = np.inf

        n_colors = colors[:self.proxy_num]
        
        cur_proxy_num = 1
        while cur_iteration < self.max_iters:
            # if abs((cur_error - precedent_error) / precedent_error) < self.iter_diff_radio_threhold:
            #     break
            # if abs(cur_error - precedent_error) < self.iter_diff_threhold:
            #     break
            if (abs((cur_error - precedent_error) / precedent_error) < self.iter_diff_radio_threhold) & (abs(cur_error - precedent_error) < self.iter_diff_threhold):
                break
            
            # if cur_iteration == 0:
            #     # cur_R = self.partition.initial_partition(self.proxy_num, self.norme)
            #     cur_R, _ = self.partition.initial_partition_furthest(self.proxy_num, self.norme)
            #     if visualize:
            #         f_colors = np.ones((self.f_num, 3), dtype=float)
            #         for i in range(self.proxy_num):
            #             f_colors[cur_R == i] = n_colors[i]
            #         mesh = mp.plot(self.v, self.f, c=f_colors, shading={"wireframe": True})
            # else:
            #     cur_R = self.partition.region_growing_proxy_initialized(cur_R, cur_proxy_center, cur_proxy_normal, self.norme)
            # cur_proxy_center, cur_proxy_normal = self.proxy.new_proxy(cur_R, self.proxy_num, self.norme)

            if cur_iteration == 0:
                # print(self.norme)
                cur_R, cur_proxy_triangle = self.partition.initial_partition(1, self.norme)
                # cur_R, cur_proxy_triangle = self.partition.initial_partition_furthest(1, self.norme)
                if visualize:
                    f_colors = np.ones((self.f_num, 3), dtype=float)
                    for i in range(self.proxy_num):
                        f_colors[cur_R == i] = n_colors[i%500]
                    mesh = mp.plot(self.v, self.f, c=f_colors, shading={"wireframe": True})
            else:
                cur_R, cur_proxy_triangle = self.partition.insert_proxy(cur_proxy_triangle, self.norme)
                cur_proxy_num += 1
            cur_proxy_center, cur_proxy_normal = self.proxy.new_proxy(cur_R, cur_proxy_num, self.norme)
            
        
            precedent_error = cur_error
            cur_error = self.distance.distance_error(cur_R, cur_proxy_center, cur_proxy_normal, self.norme)
            
            if cur_error in all_errors:
                break
            all_errors.append(cur_error)
            all_R.append(cur_R)
            cur_iteration += 1
            print("[Iteration {}/{}] Error: {}, Delta-Error: {}".format(
                cur_iteration, self.max_iters, cur_error, abs(cur_error - precedent_error)
            ))

            if visualize:        
                proxy_colors = np.array([ n_colors[cur_R[i]%50] for i in range(self.f_num) ])
                # mesh.update_object(colors=proxy_colors)
                mp.plot(self.v, self.f, c=proxy_colors, shading={"wireframe": True})

                # # write R
                # path = 'render/2/vsa/{}/iter_{}.txt'.format(vsa_obj_id, cur_iteration)
                # f = open(path, 'w')
                # for i in range(cur_R.shape[0]):
                #     f.write(str(cur_R[i]) + '\n')
                # f.close()


                
        diff_radio = abs((cur_error - precedent_error) / precedent_error)
        diff = abs(cur_error - precedent_error)
        if diff_radio < self.iter_diff_radio_threhold:
            print("Converged(diff radio): {}".format(diff_radio))
        elif diff < self.iter_diff_threhold:
            print("Converged(diff): {}".format(diff))
        elif cur_iteration == self.max_iters:
            print("Max Iteration Reached")
        else:
            print("Cycle Proxy Detected")
            
        min_R = all_R[np.argmin(all_errors)]
        if visualize:
            proxy_colors = np.array([ n_colors[min_R[i]%50] for i in range(self.f_num) ])
            # mesh.update_object(colors=proxy_colors)
            mp.plot(self.v, self.f, c=proxy_colors, shading={"wireframe": True})
            
        
        proxy_cluster = self.partition.partition_to_cluster(min_R)
        
        return min_R, proxy_cluster


    def write_proxy_cluster(self, proxy_cluster, proxy_cluster_file):
        with open(proxy_cluster_file, "w") as f:
            for key in proxy_cluster.keys():
                for val in proxy_cluster[key]:
                    f.write(str(val) + " ")
                f.write("\n")

def pre_split(v, f, use_vsa=False, visualize=False, visualize_part=False):
    mesh = om.TriMesh(v, f)
    
    f_center = np.mean(v[f], axis=1)
    f_area = igl.doublearea(v, f) / 2
    f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
    f_normal = f_normal / np.linalg.norm(f_normal, axis=1).reshape(-1, 1)
    # threshold_convexity = np.cos(89.5 * np.pi / 180)
    # threshold_convexity = np.cos(87.5 * np.pi / 180)
    threshold_convexity = np.cos(77.5 * np.pi / 180)
    # threshold_convexity = np.cos(67.5 * np.pi / 180)
    # threshold_convexity = np.cos(57.5 * np.pi / 180)
    # threshold_convexity = np.cos(47.5 * np.pi / 180)
    threshold_fitting = 0.9

    # # show normal on mesh
    # plot = mp.plot(v, f, np.ones((f.shape[0], 3)), shading={'wireframe': True})
    # plot.add_lines(
    #     f_center,
    #     f_center + f_normal * 0.01,
    #     shading={'line_width': 1, 'line_color': 'red'},
    # )

    # convex_edge, concave_edge = get_border_edge(mesh, f_center, f_normal, threshold_convexity, get_boundary=False)
    convex_edge, concave_edge, boundary_edge = get_border_edge(mesh, f_center, f_normal, threshold_convexity, get_boundary=True)

    print(f'convex edges: {len(convex_edge)}, concave edges: {len(concave_edge)}')
    
    edge = []
    if len(convex_edge) > 0:
        edge.extend(convex_edge[:, 0])
    if len(concave_edge) > 0:
        edge.extend(concave_edge[:, 0])
    # if len(boundary_edge) > 0:
    #     edge.extend(boundary_edge[:, 0])
    
    # edges = []
    # for i in range(convex_edge.shape[0]):
    #     edges.append([convex_edge[i][1], convex_edge[i][2]])
    # for i in range(concave_edge.shape[0]):
    #     edges.append([concave_edge[i][1], concave_edge[i][2]])
    # edges = np.array(edges)
    
    # if visualize:
    #     # get all edges
    #     edges_v_id = []
    #     edges_color = []
    #     for i in range(len(convex_edge)):
    #         edges_v_id.append([convex_edge[i][1], convex_edge[i][2]])
    #         edges_color.append([0, 0, 1])
    #     for i in range(len(concave_edge)):
    #         edges_v_id.append([concave_edge[i][1], concave_edge[i][2]])
    #         edges_color.append([1, 0, 0])
    #     edges_color = np.array(edges_color)

    #     f_colors = np.ones((f.shape[0], 3))
    #     xml_path = '../meshsegment_paper/camera/140780.xml'
    #     render_mesh_vtk(
    #         v, f, xml_path=xml_path, write_img=True,
    #         output_path='render/140780/vsa_borders.png',
    #         # 'render/140750/raw_borders.png',
    #         # 'render/cad16/raw_borders.png',
    #         # 'render/2/all_patch_borders.png',
    #         # 'render/2/concave_loop.png',
    #         draw_e=True, e = edges_v_id, edge_radius=0.3, _e_color = edges_color, 
    #         draw_f=True, _f_color=f_colors,
    #     )
        
    
    splited_f = split(mesh, edge)
    print(f'{len(splited_f)} regions')
    
    clusters = []
    region_cluster = []

    for i in range(len(splited_f)):
        region_f = np.array(splited_f[i])
        

        cur_v, cur_f = get_submesh(v, f, region_f)
        
        if visualize_part:
            print(f'region {i}')
            mp.plot(cur_v, cur_f, shading={'wireframe': False})
        
        if cur_f.shape[0] == 1:
            print('planar')
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue
        
        f_center = np.mean(cur_v[cur_f], axis=1)
        f_area = igl.doublearea(cur_v, cur_f) / 2
        f_normal = igl.per_face_normals(cur_v, cur_f, np.array([1., 0., 0.]))
        f_normal = f_normal / np.linalg.norm(f_normal, axis=1).reshape(-1, 1)
        primitive_fitting = PrimitiveFitting(cur_v, cur_f, f_normal, f_center, f_area)

        # primitive fitting
        f_id = np.arange(cur_f.shape[0])
        fit_rate, fit_planar_params = primitive_fitting.fit_planar(f_id)
        print(f"planar fit:{fit_rate}")
        if fit_rate > threshold_fitting:
            print(f"planar: {fit_planar_params}")
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue
        fit_rate, fit_cylinder_params = primitive_fitting.fit_cylinder(f_id, visualize=True)
        print(f"cylinder fit:{fit_rate}")
        if fit_rate > threshold_fitting:
            print(f"cylinder: {fit_cylinder_params}")
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue

        fit_rate, fit_sphere_params = primitive_fitting.fit_sphere(f_id)
        print(f"sphere fit:{fit_rate}")
        if fit_rate > threshold_fitting:
            print(f"sphere: {fit_sphere_params}")
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue

        
        fit_rate, fit_cone_params = primitive_fitting.fit_cone(f_id)
        print(f"cone fit:{fit_rate}")
        if fit_rate > threshold_fitting:
            print(f"cone: {fit_cone_params}")
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue
        
        fit_rate, fit_extrusion_params = primitive_fitting.fit_extrusion(f_id)
        print(f"extrusion fit:{fit_rate}")
        if fit_rate > threshold_fitting:
            print(f"extrusion: extrude thourgh {fit_extrusion_params}")
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)
            continue
        
        if use_vsa:
            vsa = VSA(cur_v, cur_f, proxy_num=50)
            _, proxy_cluster = vsa.iteration(visualize=False)
            raw_region_id = len(clusters)
            for proxy_id in proxy_cluster.keys():
                f_id = np.array(proxy_cluster[proxy_id])
                clusters.append(region_f[f_id])
            region_cluster.append(range(raw_region_id, len(clusters)))
        else:
            clusters.append(region_f)
            region_cluster.append(len(clusters) - 1)

    
    face_color = np.ones((f.shape[0], 3))
    for i in range(len(clusters)):
        cur_color = colors[i%500]
        for f_id in clusters[i]:
            face_color[f_id] = cur_color
    mesh = mp.plot(v, f, c=face_color, shading={"wireframe": False})
    return clusters, region_cluster, face_color

def region_only(v, f, visualize=False, visualize_part=False):
    mesh = om.TriMesh(v, f)
    
    f_center = np.mean(v[f], axis=1)
    f_area = igl.doublearea(v, f) / 2
    f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
    f_normal = f_normal / np.linalg.norm(f_normal, axis=1).reshape(-1, 1)
    threshold_convexity = np.cos(82.5 * np.pi / 180)
    threshold_fitting = 0.9

    convex_edge, concave_edge = get_border_edge(mesh, f_center, f_normal, threshold_convexity, get_boundary=False)
    
    edge = []
    if len(convex_edge) > 0:
        edge.extend(convex_edge[:, 0])
    if len(concave_edge) > 0:
        edge.extend(concave_edge[:, 0])
    
    edges = []
    for i in range(convex_edge.shape[0]):
        edges.append([convex_edge[i][1], convex_edge[i][2]])
    for i in range(concave_edge.shape[0]):
        edges.append([concave_edge[i][1], concave_edge[i][2]])
    edges = np.array(edges)
    
    splited_f = split(mesh, edge)
    print(f'{len(splited_f)} regions')
    
    patch2f = {}
    for i in range(len(splited_f)):
        patch2f[i] = np.array(splited_f[i])
    return patch2f
