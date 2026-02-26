import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import igl
from queue import PriorityQueue
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from distinctipy import distinctipy
import meshplot as mp
import timeit
import openmesh as om
from collections import deque
from scipy.interpolate import CubicSpline
# from hole_filling import triangulate_refine_fair

from utils import *
from primitive_fitting import *


class BoundaryVert:
    def __init__(self, v_id, theta):
        self.theta = theta
        self.v_id = v_id
        
    def __lt__(self, other):
        return self.theta < other.theta
    
class Node:
    def __init__(self, v_id, theta):
        self.v_id = v_id
        self.theta = theta
        self.prev = None
        self.next = None

class BoundaryAdvancingFront:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def is_empty(self):
        return self.head is None

    def append(self, v_id, theta):
        new_node = Node(v_id, theta)
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
            new_node.next = new_node
            new_node.prev = new_node
        else:
            new_node.prev = self.tail
            new_node.next = self.head
            self.tail.next = new_node
            self.head.prev = new_node
            self.tail = new_node
        
        self.size += 1

    def insert(self, target_v_id, v_id, theta):
        if self.is_empty():
            raise ValueError("BoundaryAdvancingFront is empty")
        
        current_node = self.head
        while True:
            if current_node.v_id == target_v_id:
                new_node = Node(v_id, theta)
                new_node.prev = current_node
                new_node.next = current_node.next

                current_node.next.prev = new_node
                current_node.next = new_node

                if current_node == self.tail:
                    self.tail = new_node
                
                self.size += 1
                return
            
            current_node = current_node.next
            if current_node.v_id == self.head.v_id:
                break
        
        raise ValueError(f"Vertex ID {target_v_id} not found in BoundaryAdvancingFront")

    def delete(self, current_node):
        if self.is_empty():
            raise ValueError("BoundaryAdvancingFront is empty")
        
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            current_node.prev.next = current_node.next
            current_node.next.prev = current_node.prev

            if current_node == self.head:
                self.head = current_node.next
            if current_node == self.tail:
                self.tail = current_node.prev
        
        self.size -= 1
        

    def delete_by_id(self, v_id):
        if self.is_empty():
            raise ValueError("BoundaryAdvancingFront is empty")
        
        current_node = self.head
        while True:
            if current_node.v_id == v_id:
                if self.size == 1:
                    self.head = None
                    self.tail = None
                else:
                    current_node.prev.next = current_node.next
                    current_node.next.prev = current_node.prev

                    if current_node == self.head:
                        self.head = current_node.next
                    if current_node == self.tail:
                        self.tail = current_node.prev
                
                self.size -= 1
                return
            
            current_node = current_node.next
            if current_node.v_id == self.head.v_id:
                break
        raise ValueError(f"Vertex ID {v_id} not found in BoundaryAdvancingFront")

    def find(self, v_id):
        if self.is_empty():
            return None
        
        current_node = self.head
        while True:
            if current_node.v_id == v_id:
                return current_node
            current_node = current_node.next
            if current_node.v_id == self.head.v_id:
                break
        raise ValueError(f"Vertex ID {v_id} not found in BoundaryAdvancingFront")


    def update_theta(self, v_id, new_theta):
        node = self.find(v_id)
        if node:
            node.theta = new_theta
        else:
            raise ValueError(f"Vertex ID {v_id} not found in BoundaryAdvancingFront")
    
    def update_id(self, v_id, new_id):
        node = self.find(v_id)
        if node:
            node.v_id = new_id
        else:
            raise ValueError(f"Vertex ID {v_id} not found in BoundaryAdvancingFront")


# assume boundary in closed
class HoleFilling:
    def __init__(self, v, f, visualize=False):
        self.v = v
        self.f = f
        self.visualize = visualize
        self.v_num = v.shape[0]
        self.f_num = f.shape[0]
        self.boundary_v = igl.boundary_loop(f)
        
        if self.visualize:
            print('start hole filling:')
            self.plot = mp.plot(self.v, self.f, np.ones((self.f_num, 3)), shading={
                'wireframe': False
            })
            self.plot.add_lines(
                self.v[self.boundary_v],
                self.v[np.roll(self.boundary_v, 1)],
                shading={
                    'line_color': 'black',
                    'line_width': 1
                }
            )
                
            # for i in range(len(self.boundary_v)):
            #     self.plot.add_points(np.array([self.v[self.boundary_v[i]]]), shading={
            #         'point_color': 'red',
            #         'point_size': 5
            #     })
            # self.plot.add_points(self.v[self.boundary_v], shading={
            #         'point_color': 'red',
            #         'point_size': 5
            # })
            

        
        self.initialize()
        if self.boundary_v.shape[0] == 0:
            self.new_v = self.v
            self.new_f = self.f
            self.region_triangles = []
            return
        
        # if fit planar, fill hole with planar
        primitive_fitting = PrimitiveFitting(self.v, self.f, self.f_normal, self.f_center, self.f_area)
        fit_plane_rate, fit_plane_params = primitive_fitting.fit_planar(self.boundary_v, is_v_id=True)
        if fit_plane_rate > 0.9:
            boundary_v = list(self.boundary_v.copy()[::-1])
            
            triangles = self.triangulation(boundary_v)
            region_triangles = [triangles]
            self.new_v = self.v
            self.new_f = np.concatenate((self.f, triangles), axis=0)

            # new_v, new_f = self.lib_planar_triangulate(boundary_v)
            # new_f = self.advancing_front_mesh_triangulation(boundary_v)
            # region_triangles = [new_f]
            # self.new_v = np.concatenate((self.v, new_v), axis=0)
            # self.new_f = np.concatenate((self.f, new_f), axis=0)

            # plot
            if self.visualize:
                # print("region triangles", region_triangles)
                plot = mp.plot(self.v, self.new_f, shading={
                    'wireframe': False
                })
                for triangle in region_triangles:
                    for new_f in triangle:
                        f_center = np.mean(self.v[new_f], axis=0)
                        f_normal = np.cross(self.v[new_f[1]] - self.v[new_f[0]], self.v[new_f[2]] - self.v[new_f[0]]) / np.linalg.norm(np.cross(self.v[new_f[1]] - self.v[new_f[0]], self.v[new_f[2]] - self.v[new_f[0]]))
                        plot.add_lines(np.array([f_center]), np.array([f_center + 5 * f_normal]), shading={
                            'line_color': 'red',
                            'line_width': 5
                        })
                    
            self.region_triangles = region_triangles
            return
            
        
        # pair
        if self.feature_v.shape[0] > 0:
            self.pairs = self.match_feature_sets()
        else:
            self.pairs = self.match_boundary_feature_sets()
        
        for pair in self.pairs:
            v0, v1 = pair[0], pair[1]
            pair[0] = min(v0, v1)
            pair[1] = max(v0, v1)
        
        # connect and triangular
        self.reconstruction()
    

    def get_border_edge(self, mesh, f_center, f_normal, threshold_convexity, get_boundary=False):
        convex_edge = []
        concave_edge = []
        boundary_edge = []
        
        for e_h in mesh.edges():
            
            if get_boundary & mesh.is_boundary(e_h):
                he_h = mesh.halfedge_handle(e_h, 0)
                v0 = mesh.from_vertex_handle(he_h).idx()
                v1 = mesh.to_vertex_handle(he_h).idx()
                boundary_edge.append([e_h.idx(), v0, v1])
                continue
            
            if mesh.is_boundary(e_h):
                continue
            
            he_h0 = mesh.halfedge_handle(e_h, 0)
            he_h1 = mesh.halfedge_handle(e_h, 1)
            f0 = mesh.face_handle(he_h0).idx()
            f1 = mesh.face_handle(he_h1).idx()
            
            # normal-center angle
            c0, c1, n0, n1 = f_center[f0], f_center[f1], f_normal[f0], f_normal[f1]
            angle0 = np.arccos(np.dot(n0, c1 - c0) / np.linalg.norm(c1 - c0))
            angle1 = np.arccos(np.dot(n1, c0 - c1) / np.linalg.norm(c0 - c1))
            
            metric = 0.5 * (np.cos(angle0) + np.cos(angle1))

            # dihedral angle
            metric2 = np.dot(n0, n1)


            vi = mesh.from_vertex_handle(he_h0).idx()
            vj = mesh.to_vertex_handle(he_h0).idx()
            # vi, vj = min(vi, vj), max(vi, vj)
            # if np.abs(metric2) < np.cos(12.5 * np.pi / 180):
            #     if metric2 > 0:
            #         concave_edge.append([e_h.idx(), vi, vj])
            #     elif metric2 < 0:
            #         convex_edge.append([e_h.idx(), vi, vj])
            if metric >= threshold_convexity:
                concave_edge.append([e_h.idx(), vi, vj])
            elif metric <= -threshold_convexity:
                convex_edge.append([e_h.idx(), vi, vj])
            elif abs(metric2) < np.cos(12.5 * np.pi / 180):
            # elif abs(metric2) < threshold_convexity:
                if metric > 0:
                    concave_edge.append([e_h.idx(), vi, vj])
                elif metric < 0:
                    convex_edge.append([e_h.idx(), vi, vj])
                    
        
        convex_edge = np.array(convex_edge)
        concave_edge = np.array(concave_edge)
        boundary_edge = np.array(boundary_edge)
        
        if get_boundary:
            return convex_edge, concave_edge, boundary_edge
        else:
            return convex_edge, concave_edge
    
    def initialize(self):
        self.f_area = igl.doublearea(self.v, self.f) / 2.
        self.f_center = np.mean(self.v[self.f], axis=1)
        self.f_normal = igl.per_face_normals(self.v, self.f, np.array([1., 0., 0.]))
        self.f_normal = self.f_normal / np.linalg.norm(self.f_normal, axis=1).reshape(-1, 1)
        
        self.om_mesh = om.TriMesh(self.v, self.f)
        # self.interest_vertices = self.get_interest_vertices()
        self.threshold_convexity = np.cos(87.5 * np.pi / 180)
        
        convex_edge, concave_edge = self.get_border_edge(self.om_mesh, self.f_center, self.f_normal, self.threshold_convexity, get_boundary=False)
        self.convex_edge = convex_edge[:, 0] if convex_edge.shape[0] > 0 else []
        self.concave_edge = concave_edge[:, 0] if concave_edge.shape[0] > 0 else []
        self.feature_edges = np.concatenate((self.convex_edge, self.concave_edge)).astype(int)
        
        self.feature_sets, self.feature_v = self.get_feature_sets()
        self.boundary_feature_sets, self.boundary_feature_v = self.get_feature_sets_boundary_based()

    def lib_planar_triangulate(self, v_boundary_id):
        primitive_fitting = PrimitiveFitting(self.v, self.f, self.f_normal, self.f_center, self.f_area)
        fit_plane_rate, fit_plane_params = primitive_fitting.fit_planar(v_boundary_id, is_v_id=True)


        raw_v_len, raw_f_len = self.v.shape[0], self.f.shape[0]
        new_v, new_f = triangulate_refine_fair(self.v, self.f, close_hole_fast=True)
        
        new_v = new_v[raw_v_len:]
        new_f = new_f[raw_f_len:]

        normal = fit_plane_params[:3]
        normal = normal / np.linalg.norm(normal)
        D = fit_plane_params[3] /  np.linalg.norm(normal)
        
        distances = np.dot(new_v, normal) + D
        
        # Project points onto the plane
        # p_proj = p - distance * normal
        new_v = new_v - distances[:, np.newaxis] * normal
        return new_v, new_f


    def get_interest_vertices(self):
        interest_vertices = []
        dist_max = 5
    
        for v_id in self.boundary_v:
            v_h = self.om_mesh.vertex_handle(v_id)
            queue = deque([v_h])
            visited = {v_h: 0}
            
            while queue:
                cur_v = queue.popleft()
                cur_dist = visited[cur_v]
                
                if cur_dist == dist_max:
                    interest_vertices.append(cur_v.idx())
                    continue
                
                for n_v in self.om_mesh.vv(cur_v):
                    if n_v not in visited:
                        interest_vertices.append(n_v.idx())
                        queue.append(n_v)
                        visited[n_v] = cur_dist + 1
        
        results = ()
        for v in interest_vertices:
            if v not in self.boundary_v:
                results.add(v)
        results = np.array(list(results))  
        return results
    
    
    def get_feature_sets(self):
        # v = []
        # for v_id in self.boundary_v:
        #     v.append(self.v[v_id])
        # v = np.array(v)
        # print(v.shape)
        # self.plot.add_points(v, shading={
        #     'point_color': 'red',
        #     'point_size': 5
        # })
        

        boundary_e = set()
        for i in range(len(self.boundary_v)):
            pre_v_id, v_id, next_v_id = self.boundary_v[(i-1) % len(self.boundary_v)], self.boundary_v[i], self.boundary_v[(i+1) % len(self.boundary_v)]
            v = self.om_mesh.vertex_handle(v_id)
            for he in self.om_mesh.voh(v):
                cur_next_v_id = self.om_mesh.to_vertex_handle(he).idx()
                if cur_next_v_id == pre_v_id or cur_next_v_id == next_v_id:
                    e = self.om_mesh.edge_handle(he)
                    boundary_e.add(e.idx())
            
        feature_sets = {}
        for v_id in self.boundary_v:
            
            feature_sets[v_id] = []
            v = self.om_mesh.vertex_handle(v_id)
            while True:
                # only one next-edge in feature edge and not in boundary edge
                num = 0
                he_id = -1
                for he in self.om_mesh.voh(v):
                    e = self.om_mesh.edge_handle(he)
                    n_v = self.om_mesh.to_vertex_handle(he)
                    if (e.idx() in self.feature_edges) & (e.idx() not in boundary_e) & (e.idx() not in feature_sets[v_id]) & (n_v.idx() not in self.boundary_v):
                        num += 1
                        he_id = he.idx()
                if num != 1:
                    break
                
                e_id = self.om_mesh.edge_handle(self.om_mesh.halfedge_handle(he_id)).idx()
                feature_sets[v_id].append(e_id)
                
                v = self.om_mesh.to_vertex_handle(self.om_mesh.halfedge_handle(he_id))

        
        feature_sets = {v_id: feature_sets[v_id] for v_id in feature_sets if len(feature_sets[v_id]) > 0}
        feature_v_sets = np.array([v_id for v_id in feature_sets], dtype=int)
        
        if self.visualize:
            print(feature_v_sets)
            print(feature_sets)
            lines = []
            for feature_set in feature_sets:
                for e_id in feature_sets[feature_set]:
                    he = self.om_mesh.halfedge_handle(self.om_mesh.edge_handle(e_id), 0)
                    v0 = self.v[self.om_mesh.from_vertex_handle(he).idx()]
                    v1 = self.v[self.om_mesh.to_vertex_handle(he).idx()]
                    lines.append([v0, v1])
            lines = np.array(lines)
            if feature_v_sets.shape[0] == 0:
                print("No feature vertices found.")
            else:
                print("Find {} feature vertices.".format(feature_v_sets.shape[0]))
                # self.plot.add_lines(lines[:, 0], lines[:, 1], shading={
                #     'line_color': 'red'
                # })
            pass
        
        return feature_sets, feature_v_sets
    
    
    def get_feature_sets_boundary_based(self):
        boundary_feature_sets = {}
        boundary_feature_v = []
        for i in range(len(self.boundary_v)): 
            pre_v_id, v_id, next_v_id = self.boundary_v[(i-1) % len(self.boundary_v)], self.boundary_v[i], self.boundary_v[(i+1) % len(self.boundary_v)]
            pre_vector = (self.v[v_id] - self.v[pre_v_id]) / np.linalg.norm(self.v[v_id] - self.v[pre_v_id])
            next_vector = (self.v[v_id] - self.v[next_v_id]) / np.linalg.norm(self.v[v_id] - self.v[next_v_id])
            if (np.arccos(np.dot(pre_vector, next_vector)) - np.pi) < np.pi * 30 / 180:
                cur_feature = np.zeros((3, 3))
                cur_feature[0] = pre_vector
                cur_feature[1] = next_vector
                # cross product
                cur_feature[2] = np.cross(pre_vector, next_vector) / np.linalg.norm(np.cross(pre_vector, next_vector)) 
                boundary_feature_sets[v_id] = cur_feature
                boundary_feature_v.append(v_id)
        boundary_feature_v = np.array(boundary_feature_v).astype(int)

        
        return boundary_feature_sets, boundary_feature_v
    
    
    def neighbor_normals(self, e_id):
        he = self.om_mesh.halfedge_handle(self.om_mesh.edge_handle(e_id), 0)
        n1 = self.f_normal[self.om_mesh.face_handle(he).idx()]
        n2 = self.f_normal[self.om_mesh.face_handle(self.om_mesh.opposite_halfedge_handle(he)).idx()]
        return n1, n2
    
    
    def normal_diff(self, n1, n2):
        return np.linalg.norm(n1 - n2)
    
    
    def same_edge_type(self, e1, e2):
        if e1 in self.convex_edge and e2 in self.convex_edge:
            return True
        if e1 in self.concave_edge and e2 in self.concave_edge:
            return True
        return False
    
    
    def match_feature_sets(self):
        print('matching feature sets:')
        feature_v_num = len(self.feature_v)
        
        E = np.zeros(feature_v_num // 2)
        for i in range(feature_v_num // 2):
            j = i + feature_v_num // 2

            v_i = self.feature_v[i]
            v_ni = self.feature_v[i+1]
            v_j = self.feature_v[j]
            v_nj = self.feature_v[(j+1) % feature_v_num]
            
            
            if len(self.feature_sets[v_i]) * len(self.feature_sets[v_ni]) * len(self.feature_sets[v_j]) * len(self.feature_sets[v_nj]) == 0:
                E[i] = np.inf
                continue
            
            e_i = self.feature_sets[v_i][0]
            e_ni = self.feature_sets[v_ni][0]
            e_j = self.feature_sets[v_j][0]
            e_nj = self.feature_sets[v_nj][0]
            
            if not (self.same_edge_type(e_i, e_ni) & self.same_edge_type(e_j, e_nj)):
                E[i] = np.inf
                continue
            
            n_i1, n_i2 = self.neighbor_normals(e_i)
            n_j1, n_j2 = self.neighbor_normals(e_j)
            
            n_ni1, n_ni2 = self.neighbor_normals(e_ni)
            n_nj1, n_nj2 = self.neighbor_normals(e_nj)
        
            E[i] = self.normal_diff(n_i1, n_ni2) + self.normal_diff(n_i2, n_ni1) +\
                self.normal_diff(n_j1, n_nj2) + self.normal_diff(n_j2, n_nj1)
        
        i = np.argmin(E)
        
        pairs = np.zeros((feature_v_num // 2, 2), dtype=int)
        for k in range(feature_v_num // 2):
            v0, v1 = (i - k) % feature_v_num, (i + k + 1) % feature_v_num
            pairs[k][0] = v0
            pairs[k][1] = v1
        
        if self.visualize:
            for i in range(pairs.shape[0]):
                v = np.zeros((2, 3))
                v[0] = self.v[self.feature_v[pairs[i][0]]]
                v[1] = self.v[self.feature_v[pairs[i][1]]]
                v_color = np.zeros_like(v)
                v_color[0] = colors[i]
                v_color[1] = colors[i]
                # self.plot.add_points(v, c=v_color, shading={
                #     'point_size': 5
                # })
            
        pairs = self.feature_v[pairs]
        return pairs
        
        
    def match_boundary_feature_sets(self):
        print('matching boundary feature sets:')
        feature_v_num = len(self.boundary_feature_v)
        
        E = np.zeros(feature_v_num)
        min_v = -np.ones(feature_v_num)
        for i in range(feature_v_num):
            v_i = self.boundary_feature_v[i]
            E[i] = np.inf
            for j in range(1, feature_v_num):
                v_j = self.boundary_feature_v[(i+j) % feature_v_num]

                e_i_pre = self.boundary_feature_sets[v_i][0]
                e_i_next = self.boundary_feature_sets[v_i][1]
                e_i_cross = self.boundary_feature_sets[v_i][2]
                e_ij = (self.boundary_v[v_j] - self.boundary_v[v_i]) / np.linalg.norm(self.boundary_v[v_j] - self.boundary_v[v_i])

                e_j_pre = self.boundary_feature_sets[v_j][0]
                e_j_next = self.boundary_feature_sets[v_j][1]
                e_j_cross = self.boundary_feature_sets[v_j][2]
                e_ji = (self.boundary_v[v_i] - self.boundary_v[v_j]) / np.linalg.norm(self.boundary_v[v_i] - self.boundary_v[v_j])
                
                cur_E = self.normal_diff(e_i_pre, e_j_next) + self.normal_diff(e_i_next, e_j_pre) + self.normal_diff(e_i_cross, e_ij) + self.normal_diff(e_j_cross, e_ji)
                
                if cur_E < E[i]:
                    E[i] = cur_E
                    min_v[i] = j
    
        i = np.argmin(E)
        j = min_v[i]

        pairs = -np.ones((feature_v_num // 2, 2), dtype=int)
        pairs[0] = [i, j]
        pair_id = 1
        cur_i, cur_j = i, j
        while (cur_i - cur_j) % feature_v_num != 1:
            cur_i = (cur_i - 1) % feature_v_num
            cur_j = (cur_j + 1) % feature_v_num
            pairs[pair_id] = [cur_i, cur_j]
            pair_id += 1
        cur_i, cur_j = i, j
        while (cur_j - cur_i) % feature_v_num!= 1:
            cur_i = (cur_i + 1) % feature_v_num
            cur_j = (cur_j - 1) % feature_v_num
            pairs[pair_id] = [cur_i, cur_j]
            pair_id += 1
        
        
        if self.visualize:
            for i in range(pairs.shape[0]):
                v = np.zeros((2, 3))
                v[0] = self.v[self.boundary_feature_v[pairs[i][0]]]
                v[1] = self.v[self.boundary_feature_v[pairs[i][1]]]
                v_color = np.zeros_like(v)
                v_color[0] = colors[i]
                v_color[1] = colors[i]
                # self.plot.add_points(v, c=v_color, shading={
                #     'point_size': 5
                # })
        
        pairs = self.boundary_feature_v[pairs]
        
        return pairs
    

    def boundary_theta(self, prev_v, v, next_v, normal=None):
        vec1 = prev_v - v
        vec2 = next_v - v
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        dot = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
        theta = np.arccos(dot)


        # judge convex or concave
        cross = np.cross(vec1, vec2)
        cross_norm = np.linalg.norm(cross)

        if cross_norm < 1e-6:
            return np.pi
        
        if normal is None:
            return theta

        if np.dot(cross, normal) > 0:
            return 2 * np.pi - theta
        else:
            return theta



    def bisector(self, vector1, vector2):
        norm1, norm2 = np.linalg.norm(vector1), np.linalg.norm(vector2)
        bisector = vector1 / norm1 + vector2 / norm2
        bisector /= np.linalg.norm(bisector)
        return bisector


    def trisector(self, vector1, vector2):
        dot = np.clip(np.dot(vector1, vector2), -1.0, 1.0)
        angle = np.arccos(dot)

        # if angle < 1e-3 or abs(angle - np.pi) < 1e-3:
        #     raise ValueError("Vectors are parallel or anti-parallel; trisector is undefined.")
        
        # else:
        t1 = 1.0 / 3.0
        sin = np.sin(angle)
        a = np.sin((1 - t1) * angle) / sin
        b = np.sin(t1 * angle) / sin
        trisector1 = a * vector1 + b * vector2
        trisector1 /= np.linalg.norm(trisector1)

        t2 = 2.0 / 3.0
        a = np.sin((1 - t2) * angle) / sin
        b = np.sin(t2 * angle) / sin
        trisector2 = a * vector1 + b * vector2
        trisector2 /= np.linalg.norm(trisector2)
        return trisector1, trisector2


    def advancing_front_mesh_triangulation(self, v_boundary_id):
        plot = mp.plot(self.v, self.f, np.ones((self.f_num, 3)), shading={
            'wireframe': False
        })
        
        v_boundary = self.v[v_boundary_id]
        v_boundary_num = len(v_boundary_id)

        # front = PriorityQueue()
        all_boundary_theta = {}
        boundary_front = BoundaryAdvancingFront()

        for i in range(v_boundary_num):
            v_pre = v_boundary[(i-1) % v_boundary_num]
            v_cur = v_boundary[i]
            v_next = v_boundary[(i+1) % v_boundary_num]
            theta_i = self.boundary_theta(v_pre, v_cur, v_next)

            # front.put(BoundaryVert(v_boundary_id[i], theta_i))
            all_boundary_theta[v_boundary_id[i]] = theta_i
            boundary_front.append(v_boundary_id[i], theta_i)
        
        # get surface normal
        v_id = min(all_boundary_theta, key=all_boundary_theta.get)
        v_pos = self.v[v_id]
        prev_v_pos = self.v[boundary_front.find(v_id).prev.v_id]
        next_v_pos = self.v[boundary_front.find(v_id).next.v_id]
        surface_normal = np.cross(v_pos - prev_v_pos, next_v_pos - v_pos)
        surface_normal /= np.linalg.norm(surface_normal)

        # raw_v_len = self.v.shape[0]
        # new_vertices = []
        new_faces = []

        threshold_theta_one = 85  * np.pi / 180
        threshold_theta_two = 135 * np.pi / 180
        alpha_new_edge = 1
        cur_new_f_plot_id = 0

        # def find_min_theta_vert(front):
        #     min_theta = float('inf')
        #     min_v_id = -1
        #     node = front.head
        #     while True:
        #         if node.theta < min_theta:
        #             min_theta = node.theta
        #             min_v_id = node.v_id
        #         node = node.next
        #         if node.v_id == front.head.v_id:
        #             break
        #     return min_v_id, min_theta

        # while front.qsize() > 3:
        #     v = front.get()
            # v_id, theta = v.v_id, v.theta
        # while len(new_faces) <= 1000:
        while boundary_front.size > 3:
            # v_id, theta = find_min_theta_vert(boundary_front)
            # all_theta = np.array(list(all_boundary_theta.keys()))
            # v_id = np.argmin(all_theta)
            # theta = all_theta[v_id]
            v_id = min(all_boundary_theta, key=all_boundary_theta.get)
            theta = all_boundary_theta[v_id]


            # case 1:
            # 1. update prev and next vert theta
            # 2. remove current vert
            # 3. add new face
            # 4. update all thetas
            if theta <= threshold_theta_one:
                current_vert = boundary_front.find(v_id)

                prev_prev_v_id = current_vert.prev.prev.v_id
                prev_v_id      = current_vert.prev.v_id
                next_v_id      = current_vert.next.v_id
                next_next_v_id = current_vert.next.next.v_id
                
                prev_prev_v = self.v[prev_prev_v_id]
                prev_v = self.v[prev_v_id]
                next_v = self.v[next_v_id]
                next_next_v = self.v[next_next_v_id]

                new_prev_theta = self.boundary_theta(prev_prev_v, prev_v, next_v, surface_normal)
                boundary_front.update_theta(prev_v_id, new_prev_theta)
                new_next_theta = self.boundary_theta(prev_v, next_v, next_next_v, surface_normal)
                boundary_front.update_theta(next_v_id, new_next_theta)

                boundary_front.delete_by_id(v_id)

                new_faces.append([prev_v_id, v_id, next_v_id])

                all_boundary_theta[prev_v_id] = new_prev_theta
                all_boundary_theta[next_v_id] = new_next_theta
                del all_boundary_theta[v_id]

            # case 2: 
            # 1. update current node v_id and theta
            # 2. update prev and next vert theta
            # 3. add new vertex
            # 4. add new faces
            # 5. update all thetas
            elif theta <= threshold_theta_two:
                current_vert = boundary_front.find(v_id)
                v_pos = self.v[v_id]

                prev_prev_v_id = current_vert.prev.prev.v_id
                prev_v_id      = current_vert.prev.v_id
                next_v_id      = current_vert.next.v_id
                next_next_v_id = current_vert.next.next.v_id
                
                prev_prev_v = self.v[prev_prev_v_id]
                prev_v = self.v[prev_v_id]
                next_v = self.v[next_v_id]
                next_next_v = self.v[next_next_v_id]

                # new_v_id = raw_v_len + len(new_vertices)
                new_v_id = self.v.shape[0]
                bisector = self.bisector(prev_v - self.v[v_id], next_v - self.v[v_id])
                # new_v_pos = v_pos + alpha_new_edge * (np.linalg.norm(prev_v - v_pos) + np.linalg.norm(next_v - v_pos)) * bisector
                new_v_pos = v_pos + alpha_new_edge * np.linalg.norm(prev_v + next_v - 2 * v_pos) * bisector
                # new_vertices.append(new_v_pos)
                self.v = np.vstack((self.v, new_v_pos))
                new_v_theta = self.boundary_theta(prev_v, new_v_pos, next_v, surface_normal)

                boundary_front.update_theta(v_id, new_v_theta)
                boundary_front.update_id(v_id, new_v_id)

                new_prev_theta = self.boundary_theta(prev_prev_v, prev_v, new_v_pos, surface_normal)
                boundary_front.update_theta(prev_v_id, new_prev_theta)

                new_next_theta = self.boundary_theta(prev_v, new_v_pos, next_next_v, surface_normal)
                boundary_front.update_theta(next_v_id, new_next_theta)

                new_faces.append([prev_v_id, v_id, new_v_id])
                new_faces.append([v_id, next_v_id, new_v_id])

                all_boundary_theta[new_v_id] = new_v_theta
                all_boundary_theta[prev_v_id] = new_prev_theta
                all_boundary_theta[next_v_id] = new_next_theta
                del all_boundary_theta[v_id]

            # case 2: 
            # 1. update current node v_id1 and theta1
            # 2. insert node v_id2 and theta2
            # 3. update prev and next vert theta
            # 4. add new vertices
            # 5. add new faces
            # 6. update all thetas
            else:
                current_vert = boundary_front.find(v_id)
                v_pos = self.v[v_id]

                prev_prev_v_id = current_vert.prev.prev.v_id
                prev_v_id      = current_vert.prev.v_id
                next_v_id      = current_vert.next.v_id
                next_next_v_id = current_vert.next.next.v_id
                
                prev_prev_v = self.v[prev_prev_v_id]
                prev_v = self.v[prev_v_id]
                next_v = self.v[next_v_id]
                next_next_v = self.v[next_next_v_id]

                # 4
                # new_v_id0 = raw_v_len + len(new_vertices)
                new_v_id0 = self.v.shape[0]
                trisector0, trisector1 = self.trisector(prev_v - self.v[v_id], next_v - self.v[v_id])
                # new_v_pos0 = v_pos + alpha_new_edge * (np.linalg.norm(prev_v - v_pos) + np.linalg.norm(next_v - v_pos)) * trisector0
                new_v_pos0 = v_pos + alpha_new_edge * np.linalg.norm(prev_v + next_v - 2 * v_pos) * trisector0
                # new_vertices.append(new_v_pos0)
                self.v = np.vstack((self.v, new_v_pos0))
                # new_v_id1 = raw_v_len + len(new_vertices)
                new_v_id1 = self.v.shape[0]
                # new_v_pos1 = v_pos + alpha_new_edge * (np.linalg.norm(prev_v - v_pos) + np.linalg.norm(next_v - v_pos)) * trisector1
                new_v_pos1 = v_pos + alpha_new_edge * np.linalg.norm(prev_v + next_v - 2 * v_pos) * trisector1
                # new_vertices.append(new_v_pos1)
                self.v = np.vstack((self.v, new_v_pos1))

                new_v_theta0 = self.boundary_theta(prev_v, new_v_pos0, new_v_pos1, surface_normal)
                new_v_theta1 = self.boundary_theta(new_v_pos0, new_v_pos1, next_v, surface_normal)

                # 1, 2                boundary_front.update_theta(v_id, new_v_theta0)
                boundary_front.update_theta(v_id, new_v_theta0)
                boundary_front.update_id(v_id, new_v_id0)
                boundary_front.insert(new_v_id0, new_v_id1, new_v_theta1)

                # 3
                new_prev_theta = self.boundary_theta(prev_prev_v, prev_v, new_v_pos0, surface_normal)
                new_next_theta = self.boundary_theta(new_v_pos1, next_v, next_next_v, surface_normal)
                boundary_front.update_theta(prev_v_id, new_prev_theta)
                boundary_front.update_theta(next_v_id, new_next_theta)

                # 5
                new_faces.append([prev_v_id, v_id, new_v_id0])
                new_faces.append([new_v_id0, v_id, new_v_id1])
                new_faces.append([v_id, next_v_id, new_v_id1])

                # 6
                all_boundary_theta[prev_v_id] = new_prev_theta
                all_boundary_theta[next_v_id] = new_next_theta
                all_boundary_theta[new_v_id0] = new_v_theta0
                all_boundary_theta[new_v_id1] = new_v_theta1
                del all_boundary_theta[v_id]


            # print(boundary_front.size)
            # for i in range(len(new_faces)):
            #     plot.add_lines(
            #         self.v[new_faces[i][0]],
            #         self.v[new_faces[i][1]],
            #         shading={
            #             "line_color": "red",
            #             "line_width": 1
            #         }
            #     )
            #     plot.add_lines(
            #         self.v[new_faces[i][1]],
            #         self.v[new_faces[i][2]],
            #         shading={
            #             "line_color": "red",
            #             "line_width": 1
            #         }
            #     )
            #     plot.add_lines(
            #         self.v[new_faces[i][2]],
            #         self.v[new_faces[i][0]],
            #         shading={
            #             "line_color": "red",
            #             "line_width": 1
            #         }
            #     )
            # cur_new_f_plot_id = len(new_faces)
            tmp_f = np.concatenate((self.f, np.array(new_faces)), axis=0)
            mp.plot(self.v, tmp_f, np.ones((tmp_f.shape[0], 3)), shading={
                'wireframe': False
            })
            a = 1

        # v = front.get()
        # v_id, theta = v.v_id, v.theta
        # current_vert = boundary_front.find(v_id)
        current_vert = boundary_front.head
        prev_v_id      = current_vert.prev.v_id
        next_v_id      = current_vert.next.v_id
        new_faces.append([prev_v_id, v_id, next_v_id])

        return np.array(new_faces, dtype=int)
        # return np.array(new_vertices), np.array(new_faces, dtype=int)

        

    def triangulation(self, v_region):
        points = v_region.copy()
        # find project direction
        def convex_angle(v1, v2):
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / norm))
            theta = np.rad2deg(np.arccos(np.dot(v1, v2) / norm))
            if np.any(rho < 0):
                theta = - theta
            return theta > 0
        
        
        project_normal = np.cross(self.v[points[1]] - self.v[points[0]], self.v[points[2]] - self.v[points[0]])
        project_normal /= np.linalg.norm(project_normal)
        # project_normal = - project_normal
        # # 选择非共线轴
        # if abs(project_normal[2]) > 0.9:
        #     basis_x = np.array([1.0, 0.0, 0.0])
        # else:
        #     basis_x = np.array([0.0, 0.0, 1.0])
        # # 构造正交基
        # basis_x = np.cross(project_normal, basis_x)
        # basis_x /= np.linalg.norm(basis_x)
        # basis_y = np.cross(project_normal, basis_x)
        # origin, basis = self.v[points[0]], (basis_x, basis_y)
        
        
        # # project to 2d
        # points_pos = self.v[points]
        # vecs = points_pos - origin
        # projected_points = np.array([(np.dot(v, basis[0]), np.dot(v, basis[1])) for v in vecs])
        


        # triangulation
        triangles = []
        i = 0
        while len(points) > 3:
            while True:
            # for i in range(len(points)):
                pre_v_id = (i - 1) % len(points)
                cur_v_id = i
                next_v_id = (i + 1) % len(points)
                pre_v = self.v[points[pre_v_id]]
                cur_v = self.v[points[cur_v_id]]
                next_v = self.v[points[next_v_id]]
                # pre_v = projected_points[pre_v_id]
                # cur_v = projected_points[cur_v_id]
                # next_v = projected_points[next_v_id]
                # if convex_angle(pre_v - cur_v, next_v - cur_v):
                dot = np.dot(pre_v - cur_v, next_v - cur_v)
                norm1 = np.linalg.norm(pre_v - cur_v)
                norm2 = np.linalg.norm(next_v - cur_v)
                cos_theta = dot / (norm1 * norm2)

                if cos_theta >= -1 + 1e-2:
                    triangles.append([points[pre_v_id], points[cur_v_id], points[next_v_id]])
                    points.pop(i)
                    i = i % len(points)
                    break
                i = (i + 1) % len(points)
                

                    
        if len(points) == 3:
            triangles.append(points)
        
        triangles = np.array(triangles)
        if self.visualize:
            print("triangles", triangles)     
        
        return triangles



    def reconstruction(self):
        # split by pairs
        
        # get ordered boundary edges
        boundary_e = []
        for i in range(len(self.boundary_v)):
            v = self.om_mesh.vertex_handle(self.boundary_v[i])
            v_next = self.om_mesh.vertex_handle(self.boundary_v[(i+1) % len(self.boundary_v)])
            for he in self.om_mesh.voh(v):
                cur_next_v = self.om_mesh.to_vertex_handle(he)
                if cur_next_v.idx() == v_next.idx():
                    boundary_e.append(he.idx())
                    break
        
        # delete pair in pairs which is boundary e
        boundary_edge_pair = np.zeros((len(boundary_e), 2), dtype=int)
        for i in range(len(boundary_e)):
            he = self.om_mesh.halfedge_handle(boundary_e[i])
            v0_id = self.om_mesh.from_vertex_handle(he).idx()
            v1_id = self.om_mesh.to_vertex_handle(he).idx()
            v0_id, v1_id = min(v0_id, v1_id), max(v0_id, v1_id)
            boundary_edge_pair[i][0] = v0_id
            boundary_edge_pair[i][1] = v1_id
        set_boundary = {tuple(row) for row in boundary_edge_pair}
        mask = [tuple(row) in set_boundary for row in self.pairs]
        self.pairs = self.pairs[~np.array(mask)]
                
        # search on edge
        visit_boundary_e = np.zeros(len(boundary_e), dtype=bool)
        v_regions = []
        while not (visit_boundary_e.all() == True):
            cur_he_boundary_id = np.where(visit_boundary_e == False)[0][0]
            cur_he_id = boundary_e[cur_he_boundary_id]
            cur_he = self.om_mesh.halfedge_handle(cur_he_id)
            visit_boundary_e[cur_he_boundary_id] = True
            v_start = self.om_mesh.from_vertex_handle(cur_he)
            v_region = [v_start.idx()]
            while True:
                n_v = self.om_mesh.to_vertex_handle(cur_he)
                if n_v.idx() == v_start.idx():
                    break
                v_region.append(n_v.idx())
                in_feature_index = np.where(self.pairs == n_v.idx())
                # continue if not in pair
                if in_feature_index[0].shape[0] == 0:
                    
                    cur_he_boundary_id += 1
                    cur_he_id = boundary_e[cur_he_boundary_id]
                    cur_he = self.om_mesh.halfedge_handle(cur_he_id)
                    visit_boundary_e[cur_he_boundary_id] = True
                # jump if in pair
                else:
                    i, j = in_feature_index[0][0], in_feature_index[1][0]
                    j = 0 if j == 1 else 1
                    # append opposite v
                    n_v_id = self.pairs[i][j]
                    if n_v_id == v_start.idx():
                        break
                    v_region.append(n_v_id)
                    # get next he
                    for i in range(len(boundary_e)):
                        he_id = boundary_e[i]
                        he = self.om_mesh.halfedge_handle(he_id)
                        if self.om_mesh.from_vertex_handle(he).idx() == n_v_id:
                            cur_he_boundary_id = i
                            cur_he = self.om_mesh.halfedge_handle(he_id)
                            visit_boundary_e[cur_he_boundary_id] = True
                            break
            v_regions.append(v_region)
        if self.visualize:
            print("v_regions", v_regions)
        
        

        region_triangles = []
        self.new_v = self.v.copy()
        self.new_f = self.f.copy()
        for v_region in v_regions:
            v_region.reverse()

            # triangles = self.advancing_front_mesh_triangulation(v_region)
            triangles = self.triangulation(v_region)
            region_triangles.append(triangles)
            self.new_f = np.concatenate((self.new_f, triangles), axis=0)

            # new_v, new_f = self.lib_planar_triangulate(v_region)
            # region_triangles.append(new_f)
            # self.new_v = np.concatenate((self.new_v, new_v), axis=0)
            # self.new_f = np.concatenate((self.new_f, new_f), axis=0)
        
        if self.visualize:
            print("region triangles", region_triangles)

        # plot
        if self.visualize:
            plot = mp.plot(self.v, self.new_f, shading={
                'wireframe': False
            })
            for triangle in region_triangles:
                for f in triangle:
                    f_center = np.mean(self.v[f], axis=0)
                    f_normal = np.cross(self.v[f[1]] - self.v[f[0]], self.v[f[2]] - self.v[f[0]]) / np.linalg.norm(np.cross(self.v[f[1]] - self.v[f[0]], self.v[f[2]] - self.v[f[0]]))
                    plot.add_lines(np.array([f_center]), np.array([f_center + 5 * f_normal]), shading={
                        'line_color': 'red',
                        'line_width': 5
                    })
                
        self.region_triangles = region_triangles
            
    
        