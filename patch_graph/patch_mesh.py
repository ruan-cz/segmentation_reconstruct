import numpy as np
import igl
import meshplot as mp
import openmesh as om
import networkx as nx
import trimesh
from collections import defaultdict
from collections import Counter
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from utils.mesh_utils import *
from primitive_fitting.fitting import *
from hole_filling.hole_filling import *
from cutting.intersection import *
from geometry.geometry_process import *
from geometry.geometry_computation import *
from primitive_fitting.primitives_2d import *
from primitive_fitting.geometry_primitive import *
from hole_filling.triangulate import *
from hole_filling.cylinder_hole_filling import *
from cutting.cut import *
from cutting.graph_cut import *

# read colors from file
file = open('utils/colors_500.txt', 'r')
lines = file.readlines()
colors = np.zeros((len(lines),3))
for i in range(len(lines)):
    colors[i] = np.array(lines[i].split())


# def cluster_to_partition(proxy_cluster):
#     f_num = -1
#     for proxy2f_id in proxy2f:
#         f_num = max(f_num, max(proxy_cluster[cluster_id]))
#     f_num += 1
#     face_proxy = np.zeros(f_num, dtype=int)
#     for cluster_id in proxy_cluster:
#         for f_id in proxy_cluster[cluster_id]:
#             face_proxy[f_id] = cluster_id
#     return face_proxy

class InitPatchMesh:
    '''
        INPUT
            - v, f, patch_cluster_file
        OUTPUT
            - mesh attributes: f normal, area, center
            - patch topology: patch2v, patch2f, f2patch
    '''
    def __init__(self, v, f, patch_cluster_file=None, init_region_file=None):
        self.v = v
        self.f = f
        self.patch2f = self.read_cluster(patch_cluster_file)
        
        self.initialize()
        self.construct_graph()
    
    
    '''
        COMPUTE
        - v num, f num
        - f normal, f center, f area
        # - vv_idx, vf_idx
        - patch_num, patch2v: {[]}, patch2f: {[]}, f2patch: []
    '''
    def initialize(self):
        self.v_num = self.v.shape[0]
        self.f_num = self.f.shape[0]

        # mesh geometry
        self.f_normal = igl.per_face_normals(self.v, self.f, np.array([1., 0., 0.]))
        self.f_normal = self.f_normal / np.linalg.norm(self.f_normal, axis=1).reshape(-1, 1)
        self.f_center = np.mean(self.v[self.f], axis=1)
        self.f_area = igl.doublearea(self.v, self.f) / 2.
        
        
        # mesh topology
        self.om_mesh = om.TriMesh(self.v, self.f)
        # self.vv_idx = self.om_mesh.vv_indices()
        # self.vf_idx = self.om_mesh.vf_indices()
        
        # patch-mesh topology
        self.patch_num = len(self.patch2f)
        self.patch2v = defaultdict(list)
        for p_id in range(self.patch_num):
            self.patch2v[p_id] = list(np.unique(np.array(self.f[self.patch2f[p_id]]).flatten()))
        self.f2patch = np.zeros(self.f_num, dtype=int)
        for p_id in range(self.patch_num):
            self.f2patch[self.patch2f[p_id]] = p_id
                

    '''
        INPUT
            - patch2v
        OUTPUT:
            - p_p_adj: adjacency between patch, list(list(p0, p1))
                - only consider common-edge 
    '''
    def construct_adjacency(self):
        p_edges = []
        for p_id in range(self.patch_num):
            pf = self.patch2f[p_id]
            p_edge_set = set()
            for f in pf:
                for j in range(3):
                    v0, v1 = self.f[f][j], self.f[f][(j+1)%3]
                    p_edge_set.add((min(v0, v1), max(v0, v1)))
            p_edges.append(p_edge_set)
        
        self.p_p_adj = []
        for p0 in range(self.patch_num - 1):
            for p1 in range(p0 + 1, self.patch_num):
                if len(p_edges[p0] & p_edges[p1]) > 0:
                    self.p_p_adj.append([p0, p1])
    

    '''
        INPUT
            - patch2v
        OUTPUT
            - graph: patch graph with
                - nodes with v, f
                - edges
    '''
    def construct_graph(self):
        self.construct_adjacency()

        graph = nx.Graph()
        for p_id in range(self.patch_num):
            graph.add_node(p_id)
            graph.nodes[p_id]['v'] = self.patch2v[p_id]
            graph.nodes[p_id]['f'] = self.patch2f[p_id]
        
        for pair in self.p_p_adj:
            p0, p1 = pair
            graph.add_edge(p0, p1)
        self.graph = graph


    def read_region_cluster(self, region_file):
        file = open(region_file, 'r')
        lines = file.readlines()
        region2p = {}
        for i in range(len(lines)):
            region2p[i] = lines[i].split(' ')[:-1]
            region2p[i] = [int(pid) for pid in region2p[i]]
        return region2p
    

    def read_cluster(self, patch_cluster_file):
        file = open(patch_cluster_file, 'r')
        lines = file.readlines()
        cluster = {}
        for i in range(len(lines)):
            cluster[i] = []
            for j in lines[i].split():
                cluster[i].append(int(j))
        return cluster

class PatchMesh(InitPatchMesh):
    def __init__(self, v, f, patch2f):
        self.v = v
        self.f = f
        self.patch2f = {i: list(patch2f[i]) for i in range(len(patch2f))}
        
        self.initialize()
        self.construct_graph()

class BorderType:
    '''
        INPUT
            - patch mesh with:
                - patch<->f, patch<->v
                - connectivity of patches
                - connectivity of mesh
    '''
    def __init__(self, patch_mesh):
        self.patch_mesh = patch_mesh
        self.threshold_convexity = np.cos(87.5 * np.pi / 180)
        # self.threshold_convexity = np.cos(82.5 * np.pi / 180)
        
        
    
    '''
        fit vertices into to line, by eigen vector of max eigen value
        INPUT
            - indices of vertices
        OUTPUT
            - direction, center
    '''
    def get_line_fitting(self, v_idx):
        v = self.patch_mesh.v[v_idx]
        centroid = np.mean(v, axis=0)
        v_centered = v - centroid
        cov = np.cov(v_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        line_direction = eigenvectors[:, np.argmax(eigenvalues)]
        line_direction /= np.linalg.norm(line_direction)
        return line_direction, centroid


    '''
        get neighbor vertices of vertices in patch
        INPUT
            - indices of vertices, patch id
            - patch2v
            - vv neighbor
        OUTPUT
            - indices of vertices
    '''
    def get_neighbor_v(self, v_border, patch_id):
        # all_n_v = np.unique(np.array([self.patch_mesh.vv_idx[v] for v in v_border]).flatten())
        all_n_v = []
        for v in v_border:
            v_h = self.patch_mesh.om_mesh.vertex_handle(v)
            for n_v_h in self.patch_mesh.om_mesh.vv(v_h):
                all_n_v.append(n_v_h.idx())
        all_n_v = list(set(all_n_v))
        n_v = []
        for v in all_n_v:
            if (v not in v_border) & (v in self.patch_mesh.patch2v[patch_id]):
                n_v.append(v)
        n_v = np.array(n_v).astype(int)
        return n_v

    
    '''
        get neighbor faces of vertices in patch
        INPUT
            - indices of vertices, patch id
            - f2patch
            - vf neighbor
        OUTPUT
            - indices of faces
    '''
    def get_neighbor_f(self, v_border, patch_id):
        # all_n_f = np.unique(np.array([self.patch_mesh.vf_idx[v] for v in v_border]).flatten())
        all_n_f = []
        for v in v_border:
            v_h = self.patch_mesh.om_mesh.vertex_handle(v)
            for n_v_h in self.patch_mesh.om_mesh.vf(v_h):
                all_n_f.append(n_v_h.idx())
        all_n_f = list(set(all_n_f))
        n_f = []
        for f in all_n_f:
            if self.patch_mesh.f2patch[f] == patch_id:
                n_f.append(f)
        n_f = np.array(n_f)
        # if n_f.shape[0] == 0:
        #     a = 1
        return n_f


    ''' 
        project vertices into plane
        INPUT
            - plane's normal, center
            - indices of vertices
        OUTPUT
            - projected vertices
    '''
    def projected_v(self, normal, center, v_id):
        v = self.patch_mesh.v[v_id]
        
        line_proj = np.dot(v - center, normal) / np.dot(normal, normal)
        line_proj = line_proj[:, np.newaxis]
        v_proj = v - line_proj * normal
        return v_proj
    
    
    ''' 
        project vector into plane
        INPUT
            - plane's normal
            - vector
        OUTPUT
            - projected vector
    '''
    def projected_vec(self, normal, vec):
        line_proj = np.dot(vec, normal) / np.dot(normal, normal)
        line_proj = line_proj[:, np.newaxis]
        vec_proj = vec - line_proj * normal
        return vec_proj
    

    '''
        get types of all borders
        NEED
            - v, f, type of all patches
            - connectivities of patches
        COMPUTE
            - edge_type: border type of each edge
    '''
    def get_all_borders(self, visualize=False):        
        for edge in self.patch_mesh.graph.edges:
            p1, p2 = edge
            border_type, convexity = self.get_border_type(p1, p2)

            self.patch_mesh.graph.edges[p1, p2]["border_type"] = border_type
            self.patch_mesh.graph.edges[p1, p2]["convexity"] = convexity
            self.patch_mesh.graph.edges[p1, p2]['border_v'] = self.get_patch_border_v(p1, p2)


    '''
        get border type between two neighbor patches
        INPUT:
            - patch id p1, p2
            - patch<->f, patch->v
            - type of patch
            - vf, vv neighbor
        OUTPUT:
            - border type of p1 and p2
                1: concave
                2: convex
                3: flat
    '''
    def get_border_type(self, p1, p2, G=None):
        # if p1 == 2 and p2 == 1:
        #     print(1)

        if G:
            self.patch_mesh.graph = G
        # if same planar, combine
        # if (self.patch_mesh.patch_num == 18) & (p1 == 0) & (p2 == 17):
            # print('at')
        # if (self.patch_mesh.graph.nodes[p1]["type"] == "Plane") & (self.patch_mesh.graph.nodes[p2]["type"] == "Plane"):
        #     params1 = self.patch_mesh.graph.nodes[p1]["params"]
        #     params2 = self.patch_mesh.graph.nodes[p2]["params"]
        #     if np.linalg.norm(params1 - params2) < 1e-2:
        #         return 3

        def get_single_border_type(_tmp_border):
            
            # fitted line of border
            line_direction, line_center = self.get_line_fitting(_tmp_border)

            # neighbor v & f
            n_v_1, n_v_2 = self.get_neighbor_v(_tmp_border, p1), self.get_neighbor_v(_tmp_border, p2)
            n_f_1, n_f_2 = self.get_neighbor_f(_tmp_border, p1), self.get_neighbor_f(_tmp_border, p2)
            # print(p1, p2, len(_tmp_border), n_v_1.shape, n_v_2.shape, n_f_1.shape, n_f_2.shape)
            n_f_n_1, n_f_n_2 = self.patch_mesh.f_normal[n_f_1], self.patch_mesh.f_normal[n_f_2]

            # projected center of n_v in p1, p2
            c1 = self.projected_v(line_direction, line_center, n_v_1)
            c1 = np.mean(c1, axis=0)
            c2 = self.projected_v(line_direction, line_center, n_v_2)
            c2 = np.mean(c2, axis=0)
            
            # area-weighted projected normal of n_f in p1, p2
            n1 = self.projected_vec(line_direction, n_f_n_1)
            n1 *= self.patch_mesh.f_area[n_f_1][:, np.newaxis]
            n1 = np.sum(n1, axis=0)
            n1 /= np.linalg.norm(n1)
            n2 = self.projected_vec(line_direction, n_f_n_2)
            n2 *= self.patch_mesh.f_area[n_f_2][:, np.newaxis]
            n2 = np.sum(n2, axis=0)
            n2 /= np.linalg.norm(n2)
            
            
            # metric
            convexity = 0.5 * (np.dot(n1, (c2 - c1) / np.linalg.norm(c2 - c1)) + np.dot(n2, (c1 - c2) / np.linalg.norm(c1 - c2)))
            

            
            # first classify
            type1 = -1
            if convexity >= self.threshold_convexity:
                type1 = 1
            elif convexity <= -self.threshold_convexity:
                type1 = 2
            else:
                type1 = 3
            return type1, convexity
        # print(type1)
        # return type1    

        # print(f'{p1} and {p2} have {v_border.shape[0]}')

        # v_border = self.get_patch_border_v(p1, p2)

        # if closed border, detect radio of types of all edges
        has_circle, edges = self.detect_patch_border_v_circle(p1, p2)
        if has_circle:
            e_type = []
            for edge in edges:
                e_h = v_pair2e_h(self.patch_mesh.om_mesh, edge[0], edge[1])
                e_type.append(get_single_edge_convexity(
                    self.patch_mesh.om_mesh,
                    self.patch_mesh.f_center,
                    self.patch_mesh.f_normal,
                    e_h
                ))
            most_type = Counter(e_type).most_common()[0][0]
            all_convexity = []
            all_types = []
            for _tmp in self.get_patch_border_v(p1, p2):
                # length of border
                length = np.linalg.norm(self.patch_mesh.v[_tmp[-1]] - self.patch_mesh.v[_tmp[0]])
                _type, _convexity = get_single_border_type(_tmp)
                all_types.append(_type)
                all_convexity.append(abs(_convexity) * length)
            
            return most_type, np.sum(all_convexity)
        

        v1 = self.patch_mesh.patch2v[p1]
        v2 = self.patch_mesh.patch2v[p2]
        # v_border = np.intersect1d(v1, v2)
        v_border = self.get_patch_border_v(p1, p2)

        

        all_convexity = []
        all_types = []
        for _tmp in v_border:
            # length of border
            length = np.linalg.norm(self.patch_mesh.v[_tmp[-1]] - self.patch_mesh.v[_tmp[0]])
            _type, _convexity = get_single_border_type(_tmp)
            all_types.append(_type)
            all_convexity.append(abs(_convexity) * length)
        # all_types = [get_single_border_type(_tmp) for _tmp in v_border]
        most_type = Counter(all_types).most_common()[0][0]
        # most_type = get_single_border_type(v_border)
        # get most common type and correspond convexity

        # print(most_type)
        # print(all_convexity)
        return most_type, np.sum(all_convexity)


    def show_all_borders(self):
        # show all borders
        


        # get all edges
        edges_v_id = []
        edges_color = []
        G = self.patch_mesh.graph
        all_endpoints = []
        # for edge in G.edges:
        #     p1, p2 = edge
        #     border_type = G.edges[p1,p2]['border_type']
        #     if (border_type == 1) | (border_type == 2):
        #         v_borders = self.get_patch_border_v(p1, p2)
        #         for single_border in v_borders:
        #             for i in range(len(single_border) - 1):
        #                 edges_v_id.append([single_border[i], single_border[i+1]])
        #                 edges_color.append([1, 0, 0] if border_type == 1 else [0, 0, 1])
        for edge in G.edges:
            p1, p2 = edge
            # if p1 == 8 and p2 == 9:
            v_borders = self.get_patch_border_v(p1, p2)
            for single_border in v_borders:
                all_endpoints.append(self.patch_mesh.v[single_border[0]])
                all_endpoints.append(self.patch_mesh.v[single_border[-1]])

                for i in range(len(single_border) - 1):
                    edges_v_id.append([single_border[i], single_border[i+1]])
                    edges_color.append(colors[92])
        edges_color = np.array(edges_color)
        p_color = colors[92]

        f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        # f_colors[self.patch_mesh.patch2f[9]] = colors[41]
        # xml_path = '../meshsegment_paper/camera/2/concave_loop.xml'
        xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # xml_path = '../meshsegment_paper/camera/cad16.xml'
        # xml_path = '../meshsegment_paper/camera/140750.xml'
        render_mesh_vtk(
            self.patch_mesh.v, self.patch_mesh.f, xml_path=xml_path,
            # 'render/140750/reference.png',
            # 'render/140750/raw_borders.png',
            # 'render/cad16/raw_borders.png',
            # 'render/2/all_patch_borders.png',
            # 'render/2/concave_loop.png',
            write_img=False, 
            draw_e=True, e = edges_v_id, edge_radius=0.3, _e_color = edges_color, 
            draw_f=True, _f_color=f_colors,
            draw_p=True, _p=all_endpoints,  _p_color=p_color, _p_size=0.7
        )


        # f_merged_5 = []
        # f_colors = []
        # for p_id in [5, 6, 7]:
        #     f_merged_5.extend(self.patch_mesh.patch2f[p_id])
        #     f_colors.extend([colors[p_id]] * len(self.patch_mesh.patch2f[p_id]))
        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # render_mesh_vtk(
        #     self.patch_mesh.v, self.patch_mesh.f[f_merged_5],
        #     'render/2/merged_patch_5.png',
        #     draw_f=True, _f_color=f_colors, xml_path=xml_path
        # )

         
        # f_merged_7 = []
        # f_colors = []
        # for p_id in [9, 10, 11, 12, 13]:
        #     f_merged_7.extend(self.patch_mesh.patch2f[p_id])
        #     f_colors.extend([colors[p_id]] * len(self.patch_mesh.patch2f[p_id]))
        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # render_mesh_vtk(
        #     self.patch_mesh.v, self.patch_mesh.f[f_merged_7],
        #     'render/2/merged_patch_7.png',
        #     draw_f=True, _f_color=f_colors, xml_path=xml_path
        # )

        # f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        # for p_id in range(len(self.patch_mesh.patch2f)):
        #     f_colors[self.patch_mesh.patch2f[p_id]] = colors[p_id]
        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # render_mesh_vtk(
        #     self.patch_mesh.v, self.patch_mesh.f,
        #     'render/2/all_merged_patch.png',
        #     draw_f=True, _f_color=f_colors, xml_path=xml_path
        # )


        # xml_path = '../meshsegment_paper/camera/2/non_primitive_5.xml'
        # f_5 = self.patch_mesh.patch2f[5]
        # raw_f5_index = []
        # raw_f5_path = 'render/2/vsa/region_0.txt'
        # raw_f5_file = open(raw_f5_path, 'r')
        # lines = raw_f5_file.readlines()
        # for line in lines:
        #     raw_f5_index.append(int(line.strip()))
        # raw_f5_file.close()
        # folder_5 = 'render/2/vsa/5'
        # import os
        # txt_files = [f for f in os.listdir(folder_5) if f.endswith('.txt')]
        # for txt_file in txt_files:
        #     print(txt_file)
        #     f_colors = np.ones((self.patch_mesh.f[f_5].shape[0], 3))
        #     path = os.path.join(folder_5, txt_file)
        #     R = []
        #     with open(path, 'r') as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             R.append(int(line.strip()))
        #     for i in range(self.patch_mesh.f[f_5].shape[0]):
        #         idx = [index for index, value in enumerate(f_5) if value == raw_f5_index[i]][0]
        #         f_colors[idx] = colors[5 + R[i]]
            
        #     render_mesh_vtk(
        #         self.patch_mesh.v, self.patch_mesh.f[f_5],
        #         f'render/2/non_primitive_5/{txt_file[:-4]}.png',
        #         draw_f=True, _f_color=f_colors, xml_path=xml_path
        #     )


        # xml_path = '../meshsegment_paper/camera/2/non_primitive_7.xml'
        # f_7 = self.patch_mesh.patch2f[7]
        # raw_f7_index = []
        # raw_f7_path = 'render/2/vsa/region_1.txt'
        # raw_f7_file = open(raw_f7_path, 'r')
        # lines = raw_f7_file.readlines()
        # for line in lines:
        #     raw_f7_index.append(int(line.strip()))
        # raw_f7_file.close()
        # folder_7 = 'render/2/vsa/7'
        # import os
        # txt_files = [f for f in os.listdir(folder_7) if f.endswith('.txt')]
        # for txt_file in txt_files:
        #     print(txt_file)
        #     f_colors = np.ones((self.patch_mesh.f[f_7].shape[0], 3))
        #     path = os.path.join(folder_7, txt_file)
        #     R = []
        #     with open(path, 'r') as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             R.append(int(line.strip()))
        #     for i in range(self.patch_mesh.f[f_7].shape[0]):
        #         idx = [index for index, value in enumerate(f_7) if value == raw_f7_index[i]][0]
        #         f_colors[idx] = colors[7 + R[i]]

        #     render_mesh_vtk(
        #         self.patch_mesh.v, self.patch_mesh.f[f_7],
        #         f'render/2/non_primitive_7/{txt_file[:-4]}.png',
        #         draw_f=True, _f_color=f_colors, xml_path=xml_path
        #     )


        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # for p_id in [6, 8, 9, 15, 16]:
        #     f_cylinder = self.patch_mesh.patch2f[p_id]
        #     f_colors_cylinder = np.tile(colors[41], (self.patch_mesh.f[f_cylinder].shape[0], 1))

        #     render_mesh_vtk(
        #         self.patch_mesh.v, self.patch_mesh.f[f_cylinder],
        #         f'render/2/cylinder_{p_id}.png',
        #         draw_f=True, _f_color=f_colors_cylinder, xml_path=xml_path
        #     )
        
        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # for p_id in [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]:
        #     f_plane = self.patch_mesh.patch2f[p_id]
        #     f_colors_plane = np.tile(colors[29], (self.patch_mesh.f[f_plane].shape[0], 1))

        #     render_mesh_vtk(
        #         self.patch_mesh.v, self.patch_mesh.f[f_plane],
        #         f'render/2/plane_{p_id}.png',
        #         draw_f=True, _f_color=f_colors_plane, xml_path=xml_path
        #     )
        
        

        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # f_7 = self.patch_mesh.patch2f[7]
        # f_colors_7 = np.tile(colors[7], (self.patch_mesh.f[f_7].shape[0], 1))
        # render_mesh_vtk(
        #     self.patch_mesh.v, self.patch_mesh.f[f_7],
        #     # 'render/2/non_primitive_7.png',
        #     'render/2/non_primitive_7_raw.png',
        #     draw_f=True, _f_color=f_colors_7, xml_path=xml_path
        # )

        # # xml_path = '../meshsegment_paper/camera/2/non_primitive_5.xml'
        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # f_5 = self.patch_mesh.patch2f[5]
        # f_colors_5 = np.tile(colors[5], (self.patch_mesh.f[f_5].shape[0], 1))
        # render_mesh_vtk(
        #     self.patch_mesh.v, self.patch_mesh.f[f_5],
        #     # 'render/2/non_primitive_5.png',
        #     'render/2/non_primitive_5_raw.png',
        #     draw_f=True, _f_color=f_colors_5, xml_path=xml_path
        # )
        
        
        # f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        # # for p_id in range(len(self.patch_mesh.patch2f)):
        # # # for p_id in [6, 8, 9, 15, 16]:
        # # # for p_id in [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]:
        # #     f_colors[self.patch_mesh.patch2f[p_id]] = colors[p_id]
        # #     # f_colors[self.patch_mesh.patch2f[p_id]] = colors[41]
        # #     # f_colors[self.patch_mesh.patch2f[p_id]] = colors[29]
        # # # xml_path = '../meshsegment_paper/camera/2/non_primitive_7.xml'
        # # # xml_path = '../meshsegment_paper/camera/2/non_primitive_5.xml'
        # xml_path = '../meshsegment_paper/camera/2/raw_borders.xml'
        # render_mesh_vtk(
        #     self.patch_mesh.v, self.patch_mesh.f,
        #     # 'render/2/pre_patch.png',
        #     # 'render/2/cylinder_primitives.png',
        #     # 'render/2/plane_primitives.png',
        #     'render/2/raw_borders.png',
        #     e = edges_v_id, draw_f=True, draw_e=True, _f_color=f_colors, _e_color = edges_color, xml_path=xml_path
        # )


    def cut_by_convex_concave_border(self):
        # area
        for node in self.patch_mesh.graph.nodes:
            f_ids = self.patch_mesh.graph.nodes[node]['f']
            area = np.sum(self.patch_mesh.f_area[f_ids])
            self.patch_mesh.graph.nodes[node]['area'] = area
        

        seeds = set()
        for edge in self.patch_mesh.graph.edges:
            p1, p2 = edge
            border_type = self.patch_mesh.graph.edges[p1, p2]['border_type']
            if border_type == 1:
                seeds.add(p1)
                seeds.add(p2)

        convex_edges = []
        for edge in self.patch_mesh.graph.edges:
            p1, p2 = edge
            border_type = self.patch_mesh.graph.edges[p1, p2]['border_type']
            if border_type == 2:
                convex_edges.append((p1, p2))
        
        
        results = graph_min_cut(
            self.patch_mesh.graph,
            seeds,
            convex_edges
        )

        # get all edges
        edges_v_id = []
        edges_color = []
        G = self.patch_mesh.graph
        all_endpoints = []
        for edge in G.edges:
            p1, p2 = edge
            border_type = G.edges[p1,p2]['border_type']
            if (border_type == 1) | (border_type == 2):
                v_borders = self.get_patch_border_v(p1, p2)
                for single_border in v_borders:
                    all_endpoints.append(self.patch_mesh.v[single_border[0]])
                    if single_border[0] == single_border[-1]:
                        all_endpoints.append(self.patch_mesh.v[single_border[-1]])
                    else:
                        all_endpoints.append(self.patch_mesh.v[single_border[-2]])

                    for i in range(len(single_border) - 1):
                        edges_v_id.append([single_border[i], single_border[i+1]])
                        edges_color.append([1, 0, 0] if border_type == 1 else [0, 0, 1])
        edges_color = np.array(edges_color)
        

        for seed_id, sub_nodes in results.items():
            print(f'Seed patch {seed_id}:')
            
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            for node in sub_nodes:
                f_ids = self.patch_mesh.graph.nodes[node]['f']
                f_colors[f_ids] = [0, 0, 1]
            f_ids = self.patch_mesh.graph.nodes[seed_id]['f']
            f_colors[f_ids] = [1., 0., 0.]
            mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={
                'wireframe': False
            })
        
            # xml_path = '../meshsegment_paper/camera/cad16.xml'
            # render_mesh_vtk(
            #     self.patch_mesh.v, self.patch_mesh.f,
            #     f'render/cad16/seed{seed_id}.png',
            #     # 'render/2/all_patch_borders.png',
            #     e = edges_v_id, draw_f=True, draw_e=True, _f_color=f_colors, _e_color = edges_color, xml_path=xml_path,
            #     edge_radius=0.007
            #     # draw_p=True, _p=all_endpoints, edge_radius=0.3, _p_size=0.7
            # )

        f_0 = []
        patch2f_0 = []
        for n in results[16]:
            f_0.extend(self.patch_mesh.graph.nodes[n]['f'])
        f_0 = np.array(f_0)
        for n in results[16]:
            cur_f = self.patch_mesh.graph.nodes[n]['f']
            patch2f_0.append(np.array([np.where(f_0 == f_id)[0][0] for f_id in cur_f]))
        
        # rest of self.patch_mesh.f
        f_1 = []
        patch2f_1 = []
        for p_id in self.patch_mesh.graph.nodes():
            if p_id not in results[16]:
                f_1.extend(self.patch_mesh.graph.nodes[p_id]['f'])
        f_1 = np.array(f_1)
        for p_id in self.patch_mesh.graph.nodes():
            if p_id not in results[16]:
                cur_f = self.patch_mesh.graph.nodes[p_id]['f']
                patch2f_1.append(np.array([np.where(f_1 == f_id)[0][0] for f_id in cur_f]))
        
        v0, f0 = get_submesh(self.patch_mesh.v, self.patch_mesh.f, f_0)
        v1, f1 = get_submesh(self.patch_mesh.v, self.patch_mesh.f, f_1)
        
        f_colors_0 = np.ones((f0.shape[0], 3))
        for p_id in range(len(patch2f_0)):
            f_colors_0[patch2f_0[p_id]] = colors[p_id % 500]
        f_colors_1 = np.ones((f1.shape[0], 3))
        for p_id in range(len(patch2f_1)):
            f_colors_1[patch2f_1[p_id]] = colors[p_id % 500]
            
        mp.plot(v0, f0, f_colors_0, shading={'wireframe': False})
        mp.plot(v1, f1, f_colors_1, shading={'wireframe': False})


        return [
            {
                'v': v0,
                'f': f0,
                'patch': patch2f_0
            },
            {
                'v': v1,
                'f': f1,
                'patch': patch2f_1
            }
        ]


    '''
        try cut patches by closed and seperable concave border between patches 
        INPUT
            - border type of all edges
            - patch2f
        OUTPUT
            - is seperable
            - sub objects, if seperable 
    '''
    def cut_by_concave_border(self):
        graph_cut = GraphCut(copy.deepcopy(self.patch_mesh))
        return graph_cut.result(viz=False)
     

    def show_merged_patch(self, proxy2f_file, merged_file=None):
        proxy2f = {}
        with open(proxy2f_file, 'r') as f0:
            lines = f0.readlines()
            for i in range(len(lines)):
                proxy2f[i] = []
                for j in lines[i].split():
                    proxy2f[i].append(int(j))
        merged_proxy = {}
        with open(merged_file, 'r') as f1:
            lines = f1.readlines()
            for i in range(len(lines)):
                merged_proxy[i] = []
                for j in lines[i].split():
                    merged_proxy[i].append(int(j))
        
        f_color = np.ones((self.patch_mesh.f.shape[0], 3))
        for region in merged_proxy:
            for p_id in merged_proxy[region]:
                f_id = proxy2f[p_id]
                f_color[f_id] = colors[region % 500]
        mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
            'wireframe': False
        })

        # # raw proxy color
        # raw_f_color = np.ones((self.patch_mesh.f.shape[0], 3))
        # for p in proxy2f:
        #     f_id = proxy2f[p]
        #     raw_f_color[f_id] = colors[p % 500]
        # raw_file_name = proxy2f_file.split('/')[-1].split('.')[0]
        # raw_ply_filename = f"{raw_file_name}.ply"
        # mesh = trimesh.Trimesh(self.patch_mesh.v, self.patch_mesh.f)
        # mesh.visual.face_colors = (raw_f_color * 255).astype(np.uint8)
        # mesh.export(raw_ply_filename)

        # # save into ply file
        # file_name = merged_file.split('/')[-1].split('.')[0]
        # ply_filename = f"{file_name}_patch.ply"
        # mesh = trimesh.Trimesh(self.patch_mesh.v, self.patch_mesh.f)
        # mesh.visual.face_colors = (f_color * 255).astype(np.uint8)
        # mesh.export(ply_filename)

    
    def show_patch_type(self):
        patch2f = self.patch_mesh.patch2f
        f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        for p_id in patch2f:
            p_type = self.patch_mesh.graph.nodes[p_id]['type']
            if p_type == 'Plane':
                c = [1., 0., 0.]
            elif p_type == 'Cylinder':
                c = [0., 1., 0.]
            elif p_type == 'Sphere':
                c = [0., 0., 1.]
            elif p_type == 'Cone':
                c = [1., 1., 0]
            else:
                c = [1., 0., 1.]
            f_colors[patch2f[p_id]] = c
        mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors)

    
    def show_patch(self):
        patch2f = self.patch_mesh.patch2f
        f_color = np.ones((self.patch_mesh.f.shape[0], 3))
        for p_id in patch2f:
            f_color[patch2f[p_id]] = colors[p_id % 500]
        mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
            'wireframe': False
        })


    def show_each_patch(self):
        # patch2f = self.patch_mesh.patch2f
        # f_color = np.ones((self.patch_mesh.f.shape[0], 3))
        # plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
        #     'wireframe': False
        # })
        # for p_id in patch2f:
        #     print(f"patch {p_id}")
        #     f_color[patch2f[p_id]] = colors[p_id % 500]
        #     plot.update_object(colors=f_color)
        #     print(1)

        patch2f = self.patch_mesh.patch2f
        for p_id in patch2f:
            print(f"patch {p_id}")
            f_color = np.ones((self.patch_mesh.f.shape[0], 3))
            f_color[patch2f[p_id]] = colors[p_id % 500]
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
            'wireframe': False
            })


    def show_single_patch(self, patch_id):
        f_id = self.patch_mesh.graph.nodes[patch_id]['f']
        f_color = np.ones((self.patch_mesh.f.shape[0], 3))
        f_color[f_id] = colors[0]
        mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
            'wireframe': False
        })


    def show_multi_patch(self, patch_list):
        f_color = np.ones((self.patch_mesh.f.shape[0], 3))
        for p_id in patch_list:
            f_id = self.patch_mesh.graph.nodes[p_id]['f']
            f_color[f_id] = colors[p_id % 500]
        mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_color, shading={
            'wirefrace': False
        })

    
    '''
        compute normal attribute of all patches
    '''
    def compute_all_patch_normal_attribute(self, patch_list, visualize=False):
        
        om_mesh = self.patch_mesh.om_mesh
        f_normal = igl.per_face_normals(self.patch_mesh.v, self.patch_mesh.f, np.array([1., 0., 0.]))
        def get_n_ring_neighbot_f(f_id):
            nf_1_ring = set()
            nf_2_ring = set()
            nf_3_ring = set()
            nf_4_ring = set()
            f_h = om_mesh.face_handle(f_id)
            for n_f_h in om_mesh.ff(f_h):
                nf_1_ring.add(n_f_h.idx())
                for nn_f_h in om_mesh.ff(n_f_h):
                    nf_2_ring.add(nn_f_h.idx())
                    for nnn_f_h in om_mesh.ff(nn_f_h):
                        nf_3_ring.add(nnn_f_h.idx())
                        for nnnn_f_h in om_mesh.ff(nnn_f_h):
                            nf_4_ring.add(nnnn_f_h.idx())
            nf_1_ring = list(nf_1_ring)
            nf_2_ring = list(nf_2_ring)
            nf_3_ring = list(nf_3_ring)
            nf_4_ring = list(nf_4_ring)
            return nf_1_ring, nf_2_ring, nf_3_ring, nf_4_ring
        
        def pca(vectors):
            centroid = np.mean(vectors, axis=0)
            centroid_vectors = vectors - centroid
            cov = np.cov(centroid_vectors.T)
            eigen_value, eigen_vector = np.linalg.eigh(cov)
            
            return eigen_value

        # face_1_ring_pca = np.zeros((self.patch_mesh.f.shape[0], 3))
        
        pca_1_ring = []
        pca_2_ring = []
        pca_3_ring = []
        pca_4_ring = []
        
        
        
        pca_1_ring_on_patch_list = {}
        pca_2_ring_on_patch_list = {}
        pca_3_ring_on_patch_list = {}
        pca_4_ring_on_patch_list = {}
        
        for p_id in patch_list:
            pca_1_ring = []
            pca_2_ring = []
            pca_3_ring = []
            pca_4_ring = []
            p_faces = self.patch_mesh.graph.nodes[p_id]['f']
            
            # for f_id in range(self.patch_mesh.f.shape[0]):
            for f_id in p_faces:
                nf_1_ring, nf_2_ring, nf_3_ring, nf_4_ring = get_n_ring_neighbot_f(f_id)
                nf_1_ring_normal = f_normal[nf_1_ring]
                nf_1_ring_normal = np.array(nf_1_ring_normal).reshape((-1, 3))
                nf_1_ring_normal_pca = pca(nf_1_ring_normal)
                pca_1_ring.append(nf_1_ring_normal_pca)
                
                nf_2_ring_normal = f_normal[nf_2_ring]
                nf_2_ring_normal = np.array(nf_2_ring_normal).reshape((-1, 3))
                nf_2_ring_normal_pca = pca(nf_2_ring_normal)
                pca_2_ring.append(nf_2_ring_normal_pca)
                
                nf_3_ring_normal = f_normal[nf_3_ring]
                nf_3_ring_normal = np.array(nf_3_ring_normal).reshape((-1, 3))
                nf_3_ring_normal_pca = pca(nf_3_ring_normal)
                pca_3_ring.append(nf_3_ring_normal_pca)
                
                nf_4_ring_normal = f_normal[nf_4_ring]
                nf_4_ring_normal = np.array(nf_4_ring_normal).reshape((-1, 3))
                nf_4_ring_normal_pca = pca(nf_4_ring_normal)
                pca_4_ring.append(nf_4_ring_normal_pca)
                
            pca_1_ring = np.array(pca_1_ring)
            pca_2_ring = np.array(pca_2_ring)
            pca_3_ring = np.array(pca_3_ring)
            pca_4_ring = np.array(pca_4_ring)
            
            pca_1_ring_on_patch_list[p_id] = pca_1_ring
            pca_2_ring_on_patch_list[p_id] = pca_2_ring
            pca_3_ring_on_patch_list[p_id] = pca_3_ring
            pca_4_ring_on_patch_list[p_id] = pca_4_ring

            
        # show value
        if visualize:
            # normalize over all faces
            def pca_n_ring_all_normalized(pca_n_ring_on_patch_list):
                # concat all np array in dictory to one
                pca_n_ring_all = np.concatenate(list(pca_n_ring_on_patch_list.values()), axis=0)
                min_val, max_val = np.min(pca_n_ring_all, axis=0), np.max(pca_n_ring_all, axis=0)
                return (pca_n_ring_all - min_val) / (max_val - min_val)

            def show_all_normalized_mesh(pca_n_ring):
                f_color = np.ones((self.patch_mesh.f.shape[0], 3))
                cur_row_id = 0
                for i in range(len(patch_list)):
                    p_id = patch_list[i]
                    f_id = self.patch_mesh.graph.nodes[p_id]['f']
                    for cur_p_f_id in f_id:
                        f_color[cur_p_f_id] = pca_n_ring[cur_row_id]
                        cur_row_id += 1                
                mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_color, shading={
                    'wireframe': False
                })

            pca_1_ring_all = pca_n_ring_all_normalized(pca_1_ring_on_patch_list)
            show_all_normalized_mesh(pca_1_ring_all)
            pca_2_ring_all = pca_n_ring_all_normalized(pca_2_ring_on_patch_list)
            show_all_normalized_mesh(pca_2_ring_all)
            pca_3_ring_all = pca_n_ring_all_normalized(pca_3_ring_on_patch_list)
            show_all_normalized_mesh(pca_3_ring_all)
            pca_4_ring_all = pca_n_ring_all_normalized(pca_4_ring_on_patch_list)
            show_all_normalized_mesh(pca_4_ring_all)



            # mean in patch + normalized over patch
            def show_patch_normalized_mesh(pca_n_ring_on_patch_list):
                pca_attr = []
                for p_id in patch_list:
                    f_id = self.patch_mesh.graph.nodes[p_id]['f']
                    pca_attribute = pca_n_ring_on_patch_list[p_id]
                    pca_attribute = np.mean(pca_attribute, axis=0)
                    pca_attr.append(pca_attribute)
                pca_attr = np.array(pca_attr)
                min_val, max_val = np.min(pca_attr, axis=0), np.max(pca_attr, axis=0)
                pca_attr = (pca_attr - min_val) / (max_val - min_val)
                f_color = np.ones((self.patch_mesh.f.shape[0], 3))
                for i in range(len(patch_list)):
                    p_id = patch_list[i]
                    f_id = self.patch_mesh.graph.nodes[p_id]['f']
                    f_color[f_id] = pca_attr[i]                
                mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_color, shading={
                    'wireframe': False
                })
            show_patch_normalized_mesh(pca_1_ring_on_patch_list)
            show_patch_normalized_mesh(pca_2_ring_on_patch_list)
            show_patch_normalized_mesh(pca_3_ring_on_patch_list)
            show_patch_normalized_mesh(pca_4_ring_on_patch_list)

            
            a = 1
            
            # for j in range(3):
                # min_val, max_val = np.min(pca_1_ring[:, j]), np.max(pca_1_ring[:, j])
                # min_val, max_val = np.min(pca_2_ring[:, j]), np.max(pca_2_ring[:, j])
                # if max_val != min_val:
                #     for i in range(num_p_faces):
                #         f_id = p_faces[i]
                #         f_color[f_id, j] = (pca_1_ring[i, j] - min_val) / (max_val - min_val)
                # else:
                #     for i in range(num_p_faces):
                #         f_color[p_faces[i], j] = 0.5
                # if max_val != min_val:
                #     f_color[:, j] = (pca_1_ring[i, j] - min_val) / (max_val - min_val)
                # else:
                #     f_color[:, j] = 0.5
            # mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
            #     'wireframe': False
            # })

        
        def show_cluster_by_val(labels):
            f_color = np.ones((self.patch_mesh.f.shape[0], 3))
            for i in range(len(patch_list)):
                p_id = patch_list[i]
                f_id = self.patch_mesh.graph.nodes[p_id]['f']
                f_color[f_id] = colors[labels[i]]
            mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_color, shading={
                'wireframe': False
            })
        
        for pca_n_ring_on_patch_list in [pca_1_ring_on_patch_list, pca_2_ring_on_patch_list, pca_3_ring_on_patch_list, pca_4_ring_on_patch_list]:
            pca_val = []
            for p_id in patch_list:
                f_id = self.patch_mesh.graph.nodes[p_id]['f']
                pca_attribute = pca_n_ring_on_patch_list[p_id]
                pca_attribute = np.mean(pca_attribute, axis=0)
                pca_val.append(pca_attribute)
            
            pca_val = np.array(pca_val)
            ms = sklearn.cluster.MeanShift()
            ms.fit(pca_val)
            print(ms.cluster_centers_)
            pca_val_labels = ms.labels_
            show_cluster_by_val(pca_val_labels)
            
            min_val, max_val = np.min(pca_val, axis=0), np.max(pca_val, axis=0)
            pca_val_normalized = (pca_val - min_val) / (max_val - min_val)            
            ms = sklearn.cluster.MeanShift()
            ms.fit(pca_val_normalized)
            print(ms.cluster_centers_)
            pca_normalized_val_labels = ms.labels_
            show_cluster_by_val(pca_normalized_val_labels)
            
            # pca_val = np.array(pca_val)
            # ms = sklearn.cluster.MeanShift()
            # ms.fit(pca_val[:, 2])
            # print(ms.cluster_centers_)
            # pca_val_labels = ms.labels_
            # show_cluster_by_val(pca_val_labels)
            
            # min_val, max_val = np.min(pca_val, axis=0), np.max(pca_val, axis=0)
            # pca_val_normalized = (pca_val - min_val) / (max_val - min_val)
            # ms = sklearn.cluster.MeanShift()
            # ms.fit(pca_val_normalized[:, 2])
            # print(ms.cluster_centers_)
            # pca_normalized_val_labels = ms.labels_
            # show_cluster_by_val(pca_normalized_val_labels)
            
            ms = sklearn.cluster.MeanShift(bandwidth=0.5)
            ms.fit(pca_val_normalized)
            print(ms.cluster_centers_)
            pca_band_labels = ms.labels_
            show_cluster_by_val(pca_band_labels)

            a = 1


    '''
        compute dihedral attribute of all patches
    '''
    def compute_patch_dihedral_attribute(self, patch_list, visualize=False):
        
        om_mesh = self.patch_mesh.om_mesh
        f_normal = igl.per_face_normals(self.patch_mesh.v, self.patch_mesh.f, np.array([1., 0., 0.]))
        def get_n_ring_neighbot_f(f_id):
            nf_1_ring = set()
            f_h = om_mesh.face_handle(f_id)
            for n_f_h in om_mesh.ff(f_h):
                nf_1_ring.add(n_f_h.idx())
            nf_1_ring = list(nf_1_ring)
            return nf_1_ring
        

    ''' 
        compute attribute of patch
    '''
    def compute_patch_attribute(self, p, visualize=False):
        om_mesh = self.patch_mesh.om_mesh
        p_faces = self.patch_mesh.graph.nodes[p]['f']
        num_p_faces = len(p_faces)
        
        def get_n_ring_neighbot_f(f_id):
            nf_1_ring = set()
            nf_2_ring = set()
            nf_3_ring = set()
            f_h = om_mesh.face_handle(f_id)
            for n_f_h in om_mesh.ff(f_h):
                nf_1_ring.add(n_f_h.idx())
                for nn_f_h in om_mesh.ff(n_f_h):
                    nf_2_ring.add(nn_f_h.idx())
                    for nnn_f_h in om_mesh.ff(nn_f_h):
                        nf_3_ring.add(nnn_f_h.idx())
            nf_1_ring = list(nf_1_ring)
            nf_2_ring = list(nf_2_ring)
            nf_3_ring = list(nf_3_ring)
            return nf_1_ring, nf_2_ring, nf_3_ring
        
        def pca(vectors):
            centroid = np.mean(vectors, axis=0)
            centroid_vectors = vectors - centroid
            cov = np.cov(centroid_vectors.T)
            eigen_value, eigen_vector = np.linalg.eig(cov)
            
            return eigen_value
        
        # pca
        f_normal = igl.per_face_normals(self.patch_mesh.v, self.patch_mesh.f, np.array([1., 0., 0.]))
        
        pca_1_ring = []
        pca_2_ring = []
        pca_3_ring = []
        
        for f_id in p_faces:
            nf_1_ring, nf_2_ring, nf_3_ring = get_n_ring_neighbot_f(f_id)
            nf_1_ring_normal = f_normal[nf_1_ring]
            nf_1_ring_normal = np.array(nf_1_ring_normal).reshape((-1, 3))
            
            nf_1_ring_normal_pca = pca(nf_1_ring_normal)
            pca_1_ring.append(nf_1_ring_normal_pca)
        pca_1_ring = np.array(pca_1_ring)
        # pca_2_ring = np.array(pca_2_ring)

        
        # show value
        if visualize:
            f_color = np.ones((self.patch_mesh.f.shape[0], 3))
            for j in range(3):
                min_val, max_val = np.min(pca_1_ring[:, j]), np.max(pca_1_ring[:, j])
                if max_val != min_val:
                    for i in range(num_p_faces):
                        f_id = p_faces[i]
                        f_color[f_id, j] = (pca_1_ring[i, j] - min_val) / (max_val - min_val)
                else:
                    for i in range(num_p_faces):
                        f_color[p_faces[i], j] = 0.5
            mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
                'wireframe': False
            })
        
        return pca_1_ring

    
    '''
        get type of all patches
        INPUT
            - patch2f
        OUTPUT
            - type of all patches
    '''
    def get_patch_type(self):
        patch_type = {}
        fitting = PrimitiveFitting(self.patch_mesh.v, self.patch_mesh.f, self.patch_mesh.f_normal, self.patch_mesh.f_center, self.patch_mesh.f_area)
        for patch_id in self.patch_mesh.graph.nodes:
            f_id = np.array(self.patch_mesh.patch2f[patch_id])

            # if patch_id == 86:
            #     print(' ')

            f_id = np.array(self.patch_mesh.graph.nodes[patch_id]['f'])
            # mp.plot(self.patch_mesh.v, self.patch_mesh.f[f_id], shading={'wireframe': True})

            planar_rate, planar_params = fitting.fit_planar(f_id)
            if (planar_rate > 0.9):
                # print('PLANAR')
                patch_type[patch_id] = {
                    "type": "Plane",
                    "params": planar_params
                }
                continue

            cylinder_rate, cylinder_params = fitting.fit_cylinder(f_id)
            if (cylinder_rate > 0.8):
                # print('CYLINDER')
                patch_type[patch_id] = {
                    "type": "Cylinder",
                    "params": cylinder_params
                }
                continue

            sphere_rate, sphere_params = fitting.fit_sphere(f_id)
            if (sphere_rate > 0.9):
                print('SPHERE')
                patch_type[patch_id] = {
                    "type": "Sphere",
                    "params": sphere_params
                }
                continue

            cone_rate, cone_params = fitting.fit_cone(f_id)
            if cone_rate > 0.9:
                patch_type[patch_id] = {
                    'type': 'Cone',
                    'params': cone_params
                }
                continue

            # print('OTHERS')
            patch_type[patch_id] = {
                "type": "Others",
                "params": []
            }

        nx.set_node_attributes(self.patch_mesh.graph, patch_type)


    def show_patch_type(self):
        f_color = np.ones((len(self.patch_mesh.f), 3))
        # plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f.shape[0], 3)), shading={
        #     'wireframe': False
        # })
        for p_id in self.patch_mesh.graph.nodes:
            p_type = self.patch_mesh.graph.nodes[p_id]["type"]
            f_id = f_id = self.patch_mesh.patch2f[p_id]
            if p_type == "Plane":
                f_color[f_id] = [1., 0., 0.]
            elif p_type == 'Cylinder':
                f_color[f_id] = [0., 1., 0.]
            elif p_type == 'Sphere':
                f_color[f_id] = [1., 1., 0.]
            else:
                f_color[f_id] = [0., 0., 1.]

            # print(p_id)
            # plot.update_object(colors=f_color)    
    
        mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
            'wireframe': False
        })



    '''
        Merge Condition:
            1. same type and nearly same param
            2. diff type: try fitting
    '''
    def combine_patches_by_fitting(self, visualize=False):
        if visualize:
            print('before merge:')
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            for p in self.patch_mesh.graph.nodes:
                f_colors[self.patch_mesh.patch2f[p]] = colors[p%500]
            mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={'wireframe':False})

        G = self.patch_mesh.graph
        p_plane, p_cylinder, p_others = set(), set(), set()
        for p in G.nodes:
            p_node = G.nodes[p]
            if p_node['type'] == "Plane":
                p_plane.add(p)
            elif p_node['type'] == "Cylinder":
                p_cylinder.add(p)
            elif p_node['type'] == "Others":
                p_others.add(p)
        fitting = PrimitiveFitting(self.patch_mesh.v, self.patch_mesh.f, self.patch_mesh.f_normal, self.patch_mesh.f_center, self.patch_mesh.f_area)
        
        merging = True
        while merging:
            merging = False
            for p_type in [p_plane, p_cylinder]:
                for p in p_type:
                    for n_p in list(G.neighbors(p)):
                        # if not patch_in_same_region(self.patch_mesh.region2patch, p, n_p):
                        #     continue

                        if G[p][n_p]['border_type'] != 3:
                            continue

                        merged_f_set = np.array(list(set(G.nodes[p]['f']).union(set(G.nodes[n_p]['f']))))
                        planar_rate, planar_params = fitting.fit_planar(merged_f_set)
                        cylinder_rate, cylinder_params = fitting.fit_cylinder(merged_f_set)

                        if (planar_rate > 0.9):
                            G.nodes[p]['type'] = 'Plane'
                            G.nodes[p]['params'] = planar_params
                            G = self.combine_patches([[p, n_p]], G)
                            # for key in G.nodes[p]:
                            #     if (key == 'f') | (key == 'v'):
                            #         G.nodes[p][key].extend(G.nodes[n_p][key])
                            # for n_n_p in list(G.neighbors(n_p)):
                            #     if n_n_p == p:
                            #         continue
                            #     G.add_edge(p, n_n_p)
                            # G.remove_node(n_p)

                            
                            p_type.remove(p)
                            for n_p_type in [p_plane, p_cylinder, p_others]:
                                if n_p in n_p_type:
                                    n_p_type.remove(n_p)
                                    break
                            p_plane.add(p)
                            
                            merging = True
                            break

                        if (cylinder_rate > 0.9):
                            G.nodes[p]['type'] = 'Cylinder'
                            G.nodes[p]['params'] = cylinder_params
                            G = self.combine_patches([[p, n_p]], G)
                            # for key in G.nodes[p]:
                            #     if (key == 'f') | (key == 'v'):
                            #         G.nodes[p][key].extend(G.nodes[n_p][key])
                            # for n_n_p in list(G.neighbors(n_p)):
                            #     if n_n_p == p:
                            #         continue
                            #     G.add_edge(p, n_n_p)
                            # G.remove_node(n_p)

                            
                            p_type.remove(p)
                            for n_p_type in [p_plane, p_cylinder, p_others]:
                                if n_p in n_p_type:
                                    n_p_type.remove(n_p)
                                    break
                            p_cylinder.add(p)
                            
                            merging = True
                            break
                        
                    if merging:
                        break
                if merging:
                    break
              
        self.patch_mesh.graph = G
        # update v2patch
        self.patch_mesh.v2patch = defaultdict(list)
        for p_id in self.patch_mesh.patch2f:
            f_id = self.patch_mesh.patch2f[p_id]
            v_id = np.unique(self.patch_mesh.f[f_id].flatten())
            for tmp_v_id in v_id:
                self.patch_mesh.v2patch[tmp_v_id].append(p_id)

        
        # update patch2f
        # self.patch_mesh.patch2f = {}
        # for p in G.nodes:
        #     self.patch_mesh.patch2f[p] = G.nodes[p]['f']
        
            
        # # update patch-edge v
        # update_function = getattr(self.patch_mesh, 'update_graph_edge', None)
        # if callable(update_function):
        #     self.patch_mesh.update_graph_edge()

        if visualize:
            print('after merge:')
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            for p in G.nodes:
                f_colors[self.patch_mesh.patch2f[p]] = colors[p%500]
            mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={'wireframe':False})

    
    def combine_patches_by_border(self, visualize=False):
        combine_pairs = []
        G = self.patch_mesh.graph
        for edge in G.edges:
            p1, p2 = edge
            border_type = G[p1][p2]["border_type"]
            # if (border_type == 1) | (border_type == 2):
            if (border_type == 1):
                continue
            patch_type1, patch_type2 = G.nodes[p1]["type"], G.nodes[p2]["type"]
            
            # if (patch_type1 == "Plane") & (patch_type2 == "Plane"):
            #     if (np.linalg.norm(G.nodes[p1]["params"] - G.nodes[p2]["params"]) < 1e-2) or (np.linalg.norm(G.nodes[p1]["params"] + G.nodes[p2]["params"]) < 1e-2):
            #         if p2 < p1:
            #             p1, p2 = p2, p1
            #         combine_pairs.append([p1, p2])
            #         continue
                
            if (patch_type1 == "Plane") | (patch_type2 == "Plane"):
                continue
            if p2 < p1:
                p1, p2 = p2, p1
            combine_pairs.append([p1, p2])
        combine_pairs = np.array(combine_pairs)
        # print(combine_pairs.shape)

        for i in range(combine_pairs.shape[0]):
            edge = combine_pairs[i]
            p1, p2 = edge
            if p1 == p2:
                continue
            if p2 < p1:
                p1, p2 = p2, p1
            
            # combine p2 into p1
            # update graph
            for key in G.nodes[p1]:
                if (key == 'f') | (key == 'v'):
                    G.nodes[p1][key].extend(G.nodes[p2][key])
            # update patch2f, f2patch
            
            for f_id in G.nodes[p2]['f']:
                self.patch_mesh.f2patch[f_id] = p1
            # for n_p in list(G.neighbors(p2)):
            #     self.patch_mesh.p_p_adj[p1][n_p].extend(self.patch_mesh.p_p_adj[p2][n_p])
            
            for n_p in list(G.neighbors(p2)):
                if n_p == p1:
                    continue
                G.add_edge(p1, n_p)
                
            G.remove_node(p2)
            combine_pairs[combine_pairs == p2] = p1
        
        # update patch2f
        self.patch_mesh.patch2f = {}
        for p in G.nodes:
            self.patch_mesh.patch2f[p] = G.nodes[p]['f']
        
            
        # update patch-edge v
        update_function = getattr(self.patch_mesh, 'update_graph_edge', None)
        if callable(update_function):
            self.patch_mesh.update_graph_edge()

        if visualize:
            print('after merge:')
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            for p in G.nodes:
                f_colors[self.patch_mesh.patch2f[p]] = colors[p%500]
            mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={'wireframe':False})


    def combine_plane_patches(self, visualize=False):
        G = self.patch_mesh.graph
        combine_pairs = []
        for edge in self.patch_mesh.graph.edges:
            p1, p2 = edge
            if (self.patch_mesh.graph.nodes[p1]["type"] == "Plane") & (self.patch_mesh.graph.nodes[p2]["type"] == "Plane"):
                fitting_primitive = PrimitiveFitting(self.patch_mesh.v, self.patch_mesh.f, self.patch_mesh.f_normal, self.patch_mesh.f_center, self.patch_mesh.f_area)
                f1 = self.patch_mesh.graph.nodes[p1]["f"]
                f2 = self.patch_mesh.graph.nodes[p2]["f"]
                merged_f_set = np.array(list(set(f1).union(set(f2))))
                planar_rate, _ = fitting_primitive.fit_planar(merged_f_set)
                if planar_rate > 0.9:
                    if p2 < p1:
                        p1, p2 = p2, p1
                    combine_pairs.append([p1, p2])

        for pair in combine_pairs:
            p1, p2 = pair
            if p1 == p2:
                continue

            # combine p2 into p1
            # update graph
            for key in G.nodes[p1]:
                if (key == 'f') | (key == 'v'):
                    G.nodes[p1][key].extend(G.nodes[p2][key])
            # update patch2f, f2patch
            for f_id in G.nodes[p2]['f']:
                self.patch_mesh.f2patch[f_id] = p1
            # for n_p in list(G.neighbors(p2)):
            #     self.patch_mesh.p_p_adj[p1][n_p].extend(self.patch_mesh.p_p_adj[p2][n_p])
            
            for n_p in list(G.neighbors(p2)):
                if n_p == p1:
                    continue
                G.add_edge(p1, n_p)
                
            G.remove_node(p2)
        
        # update patch2f
        self.patch_mesh.patch2f = {}
        for p in G.nodes:
            self.patch_mesh.patch2f[p] = G.nodes[p]['f']
        
            
        # update patch-edge v
        update_function = getattr(self.patch_mesh, 'update_graph_edge', None)
        if callable(update_function):
            self.patch_mesh.update_graph_edge()

        if visualize:
            print('after merge:')
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            for p in G.nodes:
                f_colors[self.patch_mesh.patch2f[p]] = colors[p%500]
            mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={'wireframe':False})


    '''
        Combine two patches p1, p2, p2 -> p1
        INPUT
            - (p1, p2)
            - node attributes
                - v, f
            - border type
                - patch<->f, patch->v
                - type of patch
                - vf, vv neighbor
        UPDATE
            - node attributes
                - v, f
                - type, params
            - neighbor attributes
                - connectivity: new neighbor = raw neighbor U (N(p2) \ p1)
                - border_type between p1, N(p2)\p1
            - patch <-> f, patch -> v
    '''
    def combine_patches(self, pairs, G, visualize=False):
        # G = self.patch_mesh.graph
        for pair in pairs:
            p0, p1 = pair

            # if visualize:
            #     print('before merge')
            #     patch2f = self.patch_mesh.patch2f
            #     f_color = np.ones((self.patch_mesh.f.shape[0], 3))
            #     for p_id in [p0, p1]:
            #         f_color[patch2f[p_id]] = colors[p_id % 500]
            #     mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
            #         'wireframe': False
            #     })
            # if p0 > p1:
            #     p0, p1 = p1, p0

            # combine p1 into p0
            # merge v, f
            for key in G.nodes[p1]:
                if (key == 'f') | (key == 'v'):
                    G.nodes[p0][key].extend(G.nodes[p1][key])

            # update f -> patch, patch -> v
            self.patch_mesh.f2patch[G.nodes[p1]['f']] = p0
            self.patch_mesh.patch2v[p0].extend(G.nodes[p1]['v'])

    
            # update neighbor
            for n_p in list(G.neighbors(p1)):
                if n_p == p0:
                    continue
                G.add_edge(p0, n_p)
                G.edges[p0, n_p]["border_type"] = self.get_border_type(p0, n_p, G)


            G.remove_node(p1)


        # update self.patch_mesh.patch2f
            self.patch_mesh.patch2f[p0] = G.nodes[p0]['f']
        # for p in G.nodes:
        #     self.patch_mesh.patch2f[p] = G.nodes[p]['f']
        # self.patch_mesh.graph = G
            if visualize:
                print('after merge')
                patch2f = self.patch_mesh.patch2f
                f_color = np.ones((self.patch_mesh.f.shape[0], 3))
                f_color[patch2f[p0]] = colors[p0 % 500]
                mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_color, shading={
                    'wireframe': False
                })

        return G


    def detect_patch_border_v_circle(self, p1, p2):
        p1_f = self.patch_mesh.patch2f[p1]
        p2_f = self.patch_mesh.patch2f[p2]

        # # 1. edge to neighbor face
        # edge2face = defaultdict(list)
        # for f_id in p1_f:
        #     f = self.patch_mesh.f[f_id]
        #     for i in range(3):
        #         v0, v1 = f[i], f[(i + 1) % 3]
        #         if v0 > v1:
        #             v0, v1 = v1, v0
        #         edge2face[(v0, v1)].append(f_id)
        # for f_id in p2_f:
        #     f = self.patch_mesh.f[f_id]
        #     for i in range(3):
        #         v0, v1 = f[i], f[(i + 1) % 3]
        #         if v0 > v1:
        #             v0, v1 = v1, v0
        #         edge2face[(v0, v1)].append(f_id)

        # # 2. get border edges: num(n_f) == 2
        # border_edges = []
        # for edge, f_id in edge2face.items():
        #     if len(f_id) == 2:
        #         f1, f2 = f_id
        #         if (f1 in p1_f and f2 in p2_f) or (f1 in p2_f and f2 in p1_f):
        #             border_edges.append([edge[0], edge[1]])

        def face_edges(face):
            return {tuple(sorted((face[i], face[(i+1) % len(face)]))) for i in range(len(face))}

        edges1 = set()
        for fi in p1_f:
            edges1 |= face_edges(self.patch_mesh.f[fi])
        edges2 = set()
        for fi in p2_f:
            edges2 |= face_edges(self.patch_mesh.f[fi])

        edges = list(edges1 & edges2)

        return detect_circle_in_edge(edges), edges
        

    def get_patch_border_v(self, p1, p2):
        p1_f = self.patch_mesh.patch2f[p1]
        p2_f = self.patch_mesh.patch2f[p2]

        def face_edges(face):
            return {tuple(sorted((face[i], face[(i+1) % len(face)]))) for i in range(len(face))}
        edges1 = set()
        for fi in p1_f:
            edges1 |= face_edges(self.patch_mesh.f[fi])
        edges2 = set()
        for fi in p2_f:
            edges2 |= face_edges(self.patch_mesh.f[fi])
        border_edges = np.array(list(edges1 & edges2))

        # 3. order border vertices
        edge_ordered, vert_ordered = connect_edge_to_line(border_edges)

        return vert_ordered
        # return vert_ordered[0]



    def cut_mesh_by_plane_tracing(self, plane_params, v_from, v_to, v_border):
        mesh = self.patch_mesh.om_mesh

        '''
            do
                target visit faces
                if v on vertex:
                    get all 'vv, ve' in neighbor faces
                if v on edge:
                    get all 'ev, ee' in neighbor face
                remove duplicate until one 
            until
                next v == v_to
        ''' 
        v_from_id = v_from
        v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
        v_from_type = 'V'
        f_visit = set()
        # is_find_next = True
        prev_v_pos = v_from_pos
        prev_v_id = v_from_id

        insert_v = []
        insert_v_num = -1
        raw_v_num = self.patch_mesh.v.shape[0]
        insert_f = defaultdict(list)
        # remove_f = []
        
        f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={
            'wireframe': True
        })
        cutting_border_vert = []
        for i in range(len(v_border) - 1):
            cutting_border_vert.append([v_border[i], v_border[i + 1]])
        
        # first direction: vertex to edge

        while True:
            if v_from_id == v_to:
                print('Reach v_border')
                break
            plot.add_lines(
                np.array([prev_v_pos]),
                np.array([v_from_pos]),
                shading={
                    "line_color": "red",
                    "line_width": 1
                }
            )
            plot.update_object()

            if v_from_type == 'V':
                v_from_h = mesh.vertex_handle(v_from_id)
            else:
                v_from_h = mesh.edge_handle(v_from_id)

            insert_type = None
            if v_from_type == 'V':
                neighbor_f = []
                for f_h in mesh.vf(v_from_h):
                    f_id = f_h.idx()
                    if f_id not in f_visit:
                        neighbor_f.append(f_id)
                        f_visit.add(f_id)
                
                next_v_info = []
                for f_id in neighbor_f:
                    result = insection_plane_f(mesh, plane_params, mesh.face_handle(f_id), v_from_pos)
                    # f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
                    # f_colors[f_id] = [1, 0, 0]
                    # plot.update_object(colors=f_colors)
                    if (result['type'] == 'vv') or (result['type'] == 've'):
                        next_v_info.append(result)
                if len(next_v_info) == 0:
                    print('Cannot find next intersect face')
                    break

                # get right next vert
                # 1. ve > vv
                # 2. remove vert in v_border
                # 3. choose the duplicate one
                next_v_info_id = -1
                next_v_set = defaultdict(list)
                for i in range(len(next_v_info)):
                    tmp_next_v_info = next_v_info[i]
                    if tmp_next_v_info['type'] == 've':
                        next_v_info_id = i
                        break
                    elif tmp_next_v_info['next_v_id'] == v_to:
                        next_v_info_id = i
                        break                       
                    elif tmp_next_v_info['next_v_id'] in v_border:
                        continue
                    else:
                        tmp_next_v_id = tmp_next_v_info['next_v_id']
                        if tmp_next_v_id not in next_v_set:
                            next_v_info_id = i
                            next_v_set[tmp_next_v_id].append(tmp_next_v_info)
                        else:
                            next_v_info_id = i
                            break
                if next_v_info_id == -1:
                    raise('[Trace Cut]: cannot find next v id')
                next_v_info = next_v_info[next_v_info_id]
                # next_v_info = next_v_info[0]

                if next_v_info['type'] == 'vv':
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = next_v_info['next_v_id']
                    v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                    v_from_type = 'V'
                    f_id = next_v_info['f_id']

                    insert_type = 'vv'
                    cutting_border_vert.append([prev_v_id, v_from_id])
                else:
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = next_v_info['next_e_id']
                    v_from_pos = next_v_info['v_insert']
                    v_from_type = 'E'
                    f_id = next_v_info['f_id']
                    
                    insert_type = 've'
                    insert_v.append(v_from_pos)
                    insert_v_num += 1
                    insert_v_id = raw_v_num + insert_v_num
                    cutting_border_vert.append([prev_v_id, insert_v_id])
                    

            elif v_from_type == 'E':
                neighbor_f = -1
                for f_id in e_h2f(mesh, v_from_h):
                # for f_h in mesh.ef(v_from_h):
                    # f_id = f_h.idx()
                    if f_id not in f_visit:
                        neighbor_f = f_id
                        f_visit.add(f_id)
                if neighbor_f == -1:
                    print('Cannot find next not visited face')
                    break
                
                result = insection_plane_f(mesh, plane_params, mesh.face_handle(neighbor_f), v_from_pos)

                if result['type'] == 'ev':
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = result['next_v_id']
                    v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                    v_from_type = 'V'
                    f_id = result['f_id']

                    insert_type = 'ev'
                    prev_insert_v_id = raw_v_num + insert_v_num
                    cutting_border_vert.append([prev_insert_v_id, v_from_id])

                elif result['type'] == 'ee':
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = result['to_e_id']
                    v_from_pos = result['v_insert']
                    v_from_type = 'E'
                    f_id = result['f_id']

                    insert_type = 'ee'
                    prev_insert_v_id = raw_v_num + insert_v_num
                    insert_v.append(v_from_pos)
                    insert_v_num += 1
                    insert_v_id = raw_v_num + insert_v_num
                    cutting_border_vert.append([prev_insert_v_id, insert_v_id])
            

            # insert faces
            if insert_type == 'vv':
                pass
            elif insert_type == 've':
                raw_face = self.patch_mesh.f[f_id]
                pre_v_index = np.where(raw_face == prev_v_id)[0][0]

                insert_f[f_id].append([
                    raw_face[pre_v_index],
                    raw_face[(pre_v_index + 1) % 3],
                    insert_v_id
                ])
                insert_f[f_id].append([
                    raw_face[(pre_v_index - 1) % 3],
                    raw_face[pre_v_index],
                    insert_v_id
                ])
            elif insert_type == 'ev':
                raw_face = self.patch_mesh.f[f_id]
                pre_v_index = np.where(raw_face == v_from_id)[0][0]

                insert_f[f_id].append([
                    raw_face[pre_v_index],
                    raw_face[(pre_v_index + 1) % 3],
                    prev_insert_v_id
                ])
                insert_f[f_id].append([
                    raw_face[(pre_v_index - 1) % 3],
                    raw_face[pre_v_index],
                    prev_insert_v_id
                ])
            elif insert_type == 'ee':
                raw_face = self.patch_mesh.f[f_id]
                
                prev_e_id = prev_v_id
                cur_e_id = v_from_id
                op_v_id0 = e_h2ov_in_face(mesh, mesh.edge_handle(prev_e_id), raw_face)
                op_v_index0 = np.where(raw_face == op_v_id0)[0][0]
                op_v_id1 = e_h2ov_in_face(mesh, mesh.edge_handle(cur_e_id), raw_face)
                op_v_index1 = np.where(raw_face == op_v_id1)[0][0]

        
                if op_v_index1 == (op_v_index0 + 1) % 3:
                    insert_f[f_id].append([
                        raw_face[op_v_index0],
                        raw_face[(op_v_index0 + 1) % 3],
                        prev_insert_v_id
                    ])
                    insert_f[f_id].append([
                        prev_insert_v_id,
                        raw_face[(op_v_index0 - 1) % 3],
                        insert_v_id
                    ])
                    insert_f[f_id].append([
                        raw_face[op_v_index0],
                        prev_insert_v_id,
                        insert_v_id
                    ])
                else:
                    insert_f[f_id].append([
                        raw_face[(op_v_index0 - 1) % 3],
                        raw_face[op_v_index0],
                        prev_insert_v_id
                    ])
                    insert_f[f_id].append([
                        raw_face[(op_v_index0 + 1) % 3],
                        prev_insert_v_id,
                        insert_v_id
                    ])
                    insert_f[f_id].append([
                        raw_face[op_v_index0],
                        insert_v_id,
                        prev_insert_v_id
                    ])
            
        for f_id in insert_f.keys():
            insert_f[f_id] = np.array(insert_f[f_id])

        # update vertices
        insert_v = np.array(insert_v)
        new_mesh_vertices = np.concatenate((self.patch_mesh.v, insert_v), axis=0)       

        # remove old faces & insert new faces with patch id
        remove_f = list(insert_f.keys())
        new_mesh_faces = np.delete(self.patch_mesh.f, remove_f, axis = 0)
        new_f_id2raw_f_id = [i for i in range(len(self.patch_mesh.f)) if i not in remove_f]
        raw_f_len = new_mesh_faces.shape[0]
        
        new_f_p_id = {}
        new_f_id = raw_f_len
        for raw_f_id in insert_f.keys():
            for _ in range(insert_f[raw_f_id].shape[0]):
                new_f_p_id[new_f_id] = self.patch_mesh.f2patch[raw_f_id]
                new_f_id += 1
            new_mesh_faces = np.concatenate((new_mesh_faces, insert_f[raw_f_id]))
        
        # check
        plot = mp.plot(new_mesh_vertices, new_mesh_faces, np.ones((new_mesh_faces.shape[0], 3)), shading={'wireframe': True})
        cutting_border_vert_array = np.array(cutting_border_vert)
        plot.add_lines(
            new_mesh_vertices[cutting_border_vert_array[:, 0]],
            new_mesh_vertices[cutting_border_vert_array[:, 1]],
            shading={
                "line_color": "red",
                "line_width": 1
            }
        )

        # 



        # split mesh
        # noting the insert edge, and split by connectivity of face Graph
        cutting_border_vert_ordered = []
        for v_pair in cutting_border_vert:
            v0_id, v1_id = v_pair
            if v0_id > v1_id:
                cutting_border_vert_ordered.append([v1_id, v0_id])
            else:
                cutting_border_vert_ordered.append([v0_id, v1_id])
        # cutting_border_vert = np.array(cutting_border_vert)
        # cutting_mesh = om.TriMesh(new_mesh_vertices, new_mesh_faces)
        # cutting_border_edge = []
        # for i in range(cutting_border_vert.shape[0]):
        #     v0_id, v1_id = cutting_border_edge[i]
        #     cutting_border_edge.append(v_pair2e(cutting_mesh, v0_id, v1_id))
        
        cutting_graph = nx.Graph()
        # add node
        for i in range(new_mesh_faces.shape[0]):
            cutting_graph.add_node(i)
        # edge to face
        edge2face = defaultdict(list)
        for f_id in range(new_mesh_faces.shape[0]):
            for i in range(3):
                v0, v1 = new_mesh_faces[f_id][i], new_mesh_faces[f_id][(i + 1) % 3]
                if v0 > v1:
                    v0, v1 = v1, v0
                edge2face[(v0, v1)].append(f_id)
        # add edges
        for edge, n_f in edge2face.items():
            if len(n_f) == 2:
                if [edge[0], edge[1]] in cutting_border_vert_ordered:
                    continue
                cutting_graph.add_edge(n_f[0], n_f[1])
        components = list(nx.connected_components(cutting_graph))
        if len(components) != 2:
            raise ValueError("SPLIT ERROR]: tracing cut graph components != 2")
        f_in, f_out = list(components[0]), list(components[1])


        # # show cutting plane
        # self.plot_mesh_cut_with_plane(
        #     new_mesh_vertices, new_mesh_faces, f_out, f_in, cutting_border_vert_array[0, :].flatten(),
        #     plane_params,
        #     plane_alpha=0.1
        # )
        a=1
        self.plot_mesh_with_cutting_plane(
            new_mesh_vertices,
            # new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            new_mesh_faces[f_in],
            plane_params,
            color_A=(0.55, 0.78, 0.98),  # 近似浅蓝
            color_B=(0.15, 0.68, 0.2),   # 近似绿色
            plane_color=(0.98, 0.92, 0.70),  # 淡黄
            plane_alpha=0.5,
            edge_alpha=0.15,
            linewidth=0.2,
            figsize=(10, 8),
            margin_ratio=0.10,
            show_reference_arrow=False
        )

        # convert patch
        patch2f_in = defaultdict(list)
        for f_id in f_in:
            if f_id >= raw_f_len:
                p_id = new_f_p_id[f_id]
            else:
                raw_f_id = new_f_id2raw_f_id[f_id]
                p_id = self.patch_mesh.f2patch[raw_f_id]
            patch2f_in[p_id].append(f_id)
        patch2f_in = [patch2f_in[p_id] for p_id in patch2f_in]

        for i in range(len(patch2f_in)):
            for j in range(len(patch2f_in[i])):
                if patch2f_in[i][j] in f_in:
                    patch2f_in[i][j] = f_in.index(patch2f_in[i][j])


        patch2f_out = defaultdict(list)
        for f_id in f_out:
            if f_id >= raw_f_len:
                p_id = new_f_p_id[f_id]
            else:
                raw_f_id = new_f_id2raw_f_id[f_id]
                p_id = self.patch_mesh.f2patch[raw_f_id]
            patch2f_out[p_id].append(f_id)
        patch2f_out = [patch2f_out[p_id] for p_id in patch2f_out]

        for i in range(len(patch2f_out)):
            for j in range(len(patch2f_out[i])):
                if patch2f_out[i][j] in f_out:
                    patch2f_out[i][j] = f_out.index(patch2f_out[i][j])
        

        # final split result
        v0, f0 = get_submesh(new_mesh_vertices, new_mesh_faces, f_in)
        # hole_filling0 = HoleFilling(v0, f0, visualize=False)
        # new_v0, new_f0, filled_regions = hole_filling0.new_v, hole_filling0.new_f, hole_filling0.region_triangles
        # if len(filled_regions) != 0:
        #     cur_id = f0.shape[0]
        #     for region in filled_regions:
        #         patch2f_in.append(list(np.arange(cur_id, cur_id + region.shape[0])))
        #         cur_id += len(region)
        # v0, f0 = new_v0, new_f0
        # v0, f0 = triangulate_refine_fair(v0, f0)
        
    

        v1, f1 = get_submesh(new_mesh_vertices, new_mesh_faces, f_out)
        # hole_filling1 = HoleFilling(v1, f1, visualize=False)
        # new_v1, new_f1, filled_regions =  hole_filling1.new_v, hole_filling1.new_f, hole_filling1.region_triangles
        # if len(filled_regions) != 0:
        #     cur_id = f1.shape[0]
        #     for region in filled_regions:
        #         patch2f_out.append(list(np.arange(cur_id, cur_id + region.shape[0])))
        #         cur_id += len(region)
        # v1, f1 = new_v1, new_f1
        # v1, f1 = triangulate_refine_fair(v1, f1)
        

        mp.plot(v0, f0, np.ones((f0.shape[0], 3)), shading={'wireframe': True})
        mp.plot(v1, f1, np.ones((f1.shape[0], 3)), shading={'wireframe': True})

        return True, [
            {
                'v': v0, 'f': f0, 'patch': patch2f_in
            },
            {
                'v': v1, 'f': f1, 'patch': patch2f_out
            }
        ]
    

    def plot_mesh_with_cutting_plane(self,
        V, F_A, F_B, plane,
        color_A=(0.55, 0.78, 0.98),  # 近似浅蓝
        color_B=(0.15, 0.68, 0.2),   # 近似绿色
        plane_color=(0.98, 0.92, 0.70),  # 淡黄
        plane_alpha=0.5,
        edge_alpha=0.15,
        linewidth=0.2,
        figsize=(10, 8),
        margin_ratio=0.10,
        show_reference_arrow=False
    ):  
        # ---------- helpers ----------
        def _plane_from_input(plane):
            """
            支持两种输入：
            1) (A, B, C, D) -> Ax + By + Cz + D = 0
            2) {'point': p0, 'normal': n}
            返回 (p0, n_unit)
            """
            if isinstance(plane, (list, tuple, np.ndarray)) and len(plane) == 4:
                A, B, C, D = map(float, plane)
                n = np.array([A, B, C], dtype=float)
                n_norm = np.linalg.norm(n)
                if n_norm == 0:
                    raise ValueError("Plane normal is zero in (A,B,C,D).")
                n = n / n_norm
                # 任取平面上一点：选择使得 Ax+By+Cz = -D
                # 把某两维设为0，第三维解出（优先选法向量最大分量对应坐标）
                idx = np.argmax(np.abs(n))
                p0 = np.zeros(3, dtype=float)
                p0[idx] = -D / n[idx]
                return p0, n
            elif isinstance(plane, dict) and 'point' in plane and 'normal' in plane:
                p0 = np.asarray(plane['point'], dtype=float)
                n = np.asarray(plane['normal'], dtype=float)
                n_norm = np.linalg.norm(n)
                if n_norm == 0:
                    raise ValueError("Plane normal is zero in {'point','normal'}.")
                n = n / n_norm
                return p0, n
            else:
                raise ValueError("Unsupported plane format. Use (A,B,C,D) or {'point':p0,'normal':n}.")

        def _orthonormal_basis_from_normal(n):
            """给定单位法向 n，返回平面内一组正交单位基 (u, v)。"""
            # 取与 n 不平行的向量做叉乘
            a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(n, a)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            v /= np.linalg.norm(v)
            return u, v

        def _plane_polygon_covering_mesh(V, p0, n, margin_ratio=0.10):
            """
            在给定平面上构造一个覆盖模型包围盒的矩形。
            做法：把所有顶点投影到平面基 (u,v)，取 min/max 并外扩 margin。
            返回 4 个顶点的 3D 坐标（按环顺序）。
            """
            u, v = _orthonormal_basis_from_normal(n)
            # 顶点到平面基的坐标（平移到p0后再投影）
            rel = V - p0
            U = rel @ u
            Vv = rel @ v
            umin, umax = U.min(), U.max()
            vmin, vmax = Vv.min(), Vv.max()
            du, dv = umax - umin, vmax - vmin
            # 外扩 margin
            umin -= du * margin_ratio
            umax += du * margin_ratio
            vmin -= dv * margin_ratio
            vmax += dv * margin_ratio
            # 矩形四角（u,v 坐标系）
            corners_uv = np.array([
                [umin, vmin],
                [umax, vmin],
                [umax, vmax],
                [umin, vmax],
            ], dtype=float)
            # 还原到三维：p = p0 + u*u_ + v*v_
            corners_xyz = p0[None, :] + corners_uv[:, 0:1] * u[None, :] + corners_uv[:, 1:2] * v[None, :]
            return corners_xyz

        def _tri_faces_to_vertices(V, F):
            """把 (N,3) 顶点和 (M,3) 三角面索引转成 Poly3DCollection 需要的顶点列表"""
            return [V[tri] for tri in F]

        def _set_equal_aspect(ax, V):
            """3D等比例显示"""
            mins = V.min(axis=0)
            maxs = V.max(axis=0)
            ctr = (mins + maxs) / 2.0
            size = (maxs - mins).max()
            ax.set_xlim(ctr[0]-size/2, ctr[0]+size/2)
            ax.set_ylim(ctr[1]-size/2, ctr[1]+size/2)
            ax.set_zlim(ctr[2]-size/2, ctr[2]+size/2)
            ax.set_box_aspect([1, 1, 1])
        """
        渲染两部分网格 + 半透明切割平面。
        - plane: (A,B,C,D) 或 {'point':p0,'normal':n}
        """
        V = np.asarray(V, dtype=float)
        F_A = np.asarray(F_A, dtype=int)
        F_B = np.asarray(F_B, dtype=int)

        p0, n = _plane_from_input(plane)
        plane_quad = _plane_polygon_covering_mesh(V, p0, n, margin_ratio=margin_ratio)

        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        # Part A
        poly_A = Poly3DCollection(_tri_faces_to_vertices(V, F_A))
        poly_A.set_facecolor((*color_A, 1.0))
        poly_A.set_edgecolor((0, 0, 0, edge_alpha))
        poly_A.set_linewidth(linewidth)
        ax.add_collection3d(poly_A)
        # Cutting plane (大矩形)
        plane_poly = Poly3DCollection([plane_quad])
        plane_poly.set_facecolor((*plane_color, plane_alpha))
        plane_poly.set_edgecolor((0.6, 0.6, 0.5, 0.6))
        plane_poly.set_linewidth(0.6)
        ax.add_collection3d(plane_poly)
        # Part B
        poly_B = Poly3DCollection(_tri_faces_to_vertices(V, F_B))
        poly_B.set_facecolor((*color_B, 1.0))
        poly_B.set_edgecolor((0, 0, 0, edge_alpha))
        poly_B.set_linewidth(linewidth)
        ax.add_collection3d(poly_B)



        _set_equal_aspect(ax, V)

        # 视觉风格：去坐标轴、边框淡化
        ax.set_axis_off()
        # 轻微视角
        ax.view_init(elev=22, azim=-55)

        # （可选）加一个“reference plane”箭头文字
        if show_reference_arrow:
            c = plane_quad.mean(axis=0)
            ax.text(c[0], c[1], c[2], "reference plane", zdir=None, fontsize=10)

        plt.tight_layout(pad=0)
        plt.show()

    
    def plot_mesh_with_cutting_cylinder(self,
        V, F_A, F_B,
        cylinder,                 # {'center': c, 'axis': a, 'radius': r, 'length': L(optional)}
        color_A=(0.55, 0.78, 0.98),
        color_B=(0.15, 0.68, 0.20),
        cyl_color=(0.98, 0.92, 0.70),
        cyl_alpha=0.45,
        edge_alpha=0,
        linewidth=0.2,
        seg=96,
        figsize=(10, 8),
        margin_ratio=0.10,
        add_caps=False,

        elev=20,
        azim=-60
    ):
        def _unit(v):
            v = np.asarray(v, dtype=float)
            n = np.linalg.norm(v)
            if n == 0: raise ValueError("zero-length vector")
            return v / n

        def _orthonormal_from_axis(a):
            a = _unit(a)
            t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(a, t); u /= np.linalg.norm(u)
            v = np.cross(a, u); v /= np.linalg.norm(v)
            return a, u, v

        def _auto_cyl_span_along_axis(V, c, a, margin_ratio=0.10):
            """
            Project all vertices to axis direction 'a' and get [smin, smax] as cylinder span.
            Returns endpoints p0, p1 (center line ends).
            """
            rel = V - c[None, :]
            s = rel @ a
            smin, smax = s.min(), s.max()
            L = smax - smin
            smin -= L * margin_ratio
            smax += L * margin_ratio
            p0 = c + smin * a
            p1 = c + smax * a
            return p0, p1

        def _cylinder_surface_mesh(V, center, axis, radius, length=None, seg=72, margin_ratio=0.10, add_caps=False):
            """
            Build a triangle mesh for a (finite) cylinder that visually covers the mesh.
            If `length` is None, span is auto-chosen from V along axis with margin.
            center: any point on cylinder axis (3,)
            axis: cylinder axis direction (3,)
            radius: scalar
            length: scalar total length along axis, optional
            """
            a, u, v = _orthonormal_from_axis(axis)
            c = np.asarray(center, dtype=float)

            if length is None:
                p0, p1 = _auto_cyl_span_along_axis(V, c, a, margin_ratio)
            else:
                p0 = c - 0.5 * length * a
                p1 = c + 0.5 * length * a

            angles = np.linspace(0, 2*np.pi, seg, endpoint=False)
            ring0 = p0[None,:] + radius*np.cos(angles)[:,None]*u[None,:] + radius*np.sin(angles)[:,None]*v[None,:]
            ring1 = p1[None,:] + radius*np.cos(angles)[:,None]*u[None,:] + radius*np.sin(angles)[:,None]*v[None,:]

            Vc = np.vstack([ring0, ring1])
            F = []

            # lateral surface (as two triangles per quad)
            for i in range(seg):
                a0 = i
                a1 = (i+1) % seg
                b0 = seg + i
                b1 = seg + (i+1) % seg
                F += [[a0, a1, b1], [a0, b1, b0]]

            if add_caps:
                # optional discs for visual emphasis (kept very light)
                Vc = np.vstack([Vc, p0[None,:], p1[None,:]])
                id0, id1 = 2*seg, 2*seg+1
                for i in range(seg):
                    j = (i+1) % seg
                    F += [[id0, j, i]]
                    F += [[id1, seg+i, seg+j]]

            return Vc, np.asarray(F, dtype=int)
        V = np.asarray(V, dtype=float)
        F_A = np.asarray(F_A, dtype=int)
        F_B = np.asarray(F_B, dtype=int)

        # c = np.asarray(cylinder['center'], dtype=float)
        # a = np.asarray(cylinder['axis'], dtype=float)
        # r = float(cylinder['radius'])
        # L = float(cylinder['length']) if 'length' in cylinder and cylinder['length'] is not None else None
        c = cylinder[:3].reshape(-1)
        a = cylinder[3:6].reshape(-1)
        r = float(cylinder[6])
        L = None

        cyl_V, cyl_F = _cylinder_surface_mesh(
            V, c, a, r, length=L, seg=seg, margin_ratio=margin_ratio, add_caps=add_caps
        )

        fig = plt.figure(figsize=figsize, dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        # Part A
        ax.add_collection3d(Poly3DCollection([V[f] for f in F_A],
            facecolors=(*color_A, 1.0), edgecolors=(0,0,0,edge_alpha), linewidths=linewidth))
        
        # Cutting cylinder (semi-transparent)
        ax.add_collection3d(Poly3DCollection([cyl_V[f] for f in cyl_F],
            facecolors=(*cyl_color, cyl_alpha), edgecolors=(0.,0.,0.,0.), linewidths=0.6))
        
        # Part B
        ax.add_collection3d(Poly3DCollection([V[f] for f in F_B],
            facecolors=(*color_B, 1.0), edgecolors=(0,0,0,edge_alpha), linewidths=linewidth))

        

        # equal aspect + clean look
        mins = V.min(axis=0); maxs = V.max(axis=0)
        ctr = (mins + maxs) / 2.0; size = (maxs - mins).max()
        ax.set_xlim(ctr[0]-size/2, ctr[0]+size/2)
        ax.set_ylim(ctr[1]-size/2, ctr[1]+size/2)
        ax.set_zlim(ctr[2]-size/2, ctr[2]+size/2)
        ax.set_box_aspect([1,1,1])
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        # ax.view_init(elev=42, azim=-150)
        plt.tight_layout(pad=0)
        plt.show()


    '''
        Cut the mesh by plane for rotation symmetry items
            - for p_id, next_p_id: cut from common_patch_border[p_id][v_to] to common_patch_border[next_p_id][v_from]
        INPUT
            - common_patch: ordered p_id list
            - common_patch_border: [dict{'v_border', 'v_from', 'v_to'}]
    '''
    def cut_mesh_by_rotation_symmetric_plane_tracing(self, plane_param, common_patch, common_patch_border):
        mesh = self.patch_mesh.om_mesh

        cutting_border_vert = []
        for p_id in common_patch_border:
            v_border = common_patch_border[p_id]['v_border']
            for i in range(len(v_border) - 1):
                cutting_border_vert.append([v_border[i], v_border[i + 1]])
        insert_v = []
        insert_v_num = -1
        raw_v_num = self.patch_mesh.v.shape[0]
        insert_f = defaultdict(list)
        f_visit = set()


        f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={
            'wireframe': False
        })
        for i in range(len(common_patch)):
            p_id = common_patch[i]
            next_p_id = common_patch[(i + 1) % len(common_patch)]

            # initial
            v_from_id = common_patch_border[p_id]['v_to']
            v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
            v_from_type = 'V'
            v_to = common_patch_border[next_p_id]['v_from']

            prev_v_pos = v_from_pos
            prev_v_id = v_from_id


            # v_from_id = v_from
            # v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
            # v_from_type = 'V'
            # prev_v_pos = v_from_pos
            # prev_v_id = v_from_id

            # cutting_border_vert = []
            # for i in range(len(v_border) - 1):
            #     cutting_border_vert.append([v_border[i], v_border[i + 1]])
            
            # first direction: vertex to edge

            while True:
                if v_from_id == v_to:
                    print('Reach v_border')
                    break
                plot.add_lines(
                    np.array([prev_v_pos]),
                    np.array([v_from_pos]),
                    shading={
                        "line_color": "red",
                        "line_width": 1
                    }
                )
                plot.update_object()

                if v_from_type == 'V':
                    v_from_h = mesh.vertex_handle(v_from_id)
                else:
                    v_from_h = mesh.edge_handle(v_from_id)

                insert_type = None
                if v_from_type == 'V':
                    neighbor_f = []
                    for f_h in mesh.vf(v_from_h):
                        f_id = f_h.idx()
                        if f_id not in f_visit:
                            neighbor_f.append(f_id)
                            f_visit.add(f_id)
                    
                    next_v_info = []
                    for f_id in neighbor_f:
                        result = insection_plane_f(mesh, plane_param, mesh.face_handle(f_id), v_from_pos)
                        # f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
                        # f_colors[f_id] = [1, 0, 0]
                        # plot.update_object(colors=f_colors)
                        if (result['type'] == 'vv') or (result['type'] == 've'):
                            next_v_info.append(result)
                    if len(next_v_info) == 0:
                        print('Cannot find next intersect face')
                        break



                    # get right next vert
                    # 1. ve > vv
                    # 2. remove vert in v_border
                    # 3. choose the duplicate one
                    next_v_info_id = -1
                    next_v_set = defaultdict(list)
                    for i in range(len(next_v_info)):
                        tmp_next_v_info = next_v_info[i]
                        if tmp_next_v_info['type'] == 've':
                            next_v_info_id = i
                            break
                        elif tmp_next_v_info['next_v_id'] == v_to:
                            next_v_info_id = i
                            break                       
                        elif tmp_next_v_info['next_v_id'] in v_border:
                            continue
                        else:
                            tmp_next_v_id = tmp_next_v_info['next_v_id']
                            if tmp_next_v_id not in next_v_set:
                                next_v_info_id = i
                                next_v_set[tmp_next_v_id].append(tmp_next_v_info)
                            else:
                                next_v_info_id = i
                                break
                    if next_v_info_id == -1:
                        raise('[Trace Cut]: cannot find next v id')
                    next_v_info = next_v_info[next_v_info_id]
                    # next_v_info = next_v_info[0]

                    if next_v_info['type'] == 'vv':
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = next_v_info['next_v_id']
                        v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                        v_from_type = 'V'
                        f_id = next_v_info['f_id']

                        insert_type = 'vv'
                        cutting_border_vert.append([prev_v_id, v_from_id])
                    else:
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = next_v_info['next_e_id']
                        v_from_pos = next_v_info['v_insert']
                        v_from_type = 'E'
                        f_id = next_v_info['f_id']
                        
                        insert_type = 've'
                        insert_v.append(v_from_pos)
                        insert_v_num += 1
                        insert_v_id = raw_v_num + insert_v_num
                        cutting_border_vert.append([prev_v_id, insert_v_id])
                        

                elif v_from_type == 'E':
                    neighbor_f = -1
                    for f_id in e_h2f(mesh, v_from_h):
                    # for f_h in mesh.ef(v_from_h):
                        # f_id = f_h.idx()
                        if f_id not in f_visit:
                            neighbor_f = f_id
                            f_visit.add(f_id)
                    if neighbor_f == -1:
                        print('Cannot find next not visited face')
                        break
                    
                    result = insection_plane_f(mesh, plane_param, mesh.face_handle(neighbor_f), v_from_pos)

                    if result['type'] == 'ev':
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = result['next_v_id']
                        v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                        v_from_type = 'V'
                        f_id = result['f_id']

                        insert_type = 'ev'
                        prev_insert_v_id = raw_v_num + insert_v_num
                        cutting_border_vert.append([prev_insert_v_id, v_from_id])

                    elif result['type'] == 'ee':
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = result['to_e_id']
                        v_from_pos = result['v_insert']
                        v_from_type = 'E'
                        f_id = result['f_id']

                        insert_type = 'ee'
                        prev_insert_v_id = raw_v_num + insert_v_num
                        insert_v.append(v_from_pos)
                        insert_v_num += 1
                        insert_v_id = raw_v_num + insert_v_num
                        cutting_border_vert.append([prev_insert_v_id, insert_v_id])
                

                # insert faces
                if insert_type == 'vv':
                    pass
                elif insert_type == 've':
                    raw_face = self.patch_mesh.f[f_id]
                    pre_v_index = np.where(raw_face == prev_v_id)[0][0]

                    insert_f[f_id].append([
                        raw_face[pre_v_index],
                        raw_face[(pre_v_index + 1) % 3],
                        insert_v_id
                    ])
                    insert_f[f_id].append([
                        raw_face[(pre_v_index - 1) % 3],
                        raw_face[pre_v_index],
                        insert_v_id
                    ])
                elif insert_type == 'ev':
                    raw_face = self.patch_mesh.f[f_id]
                    pre_v_index = np.where(raw_face == v_from_id)[0][0]

                    insert_f[f_id].append([
                        raw_face[pre_v_index],
                        raw_face[(pre_v_index + 1) % 3],
                        prev_insert_v_id
                    ])
                    insert_f[f_id].append([
                        raw_face[(pre_v_index - 1) % 3],
                        raw_face[pre_v_index],
                        prev_insert_v_id
                    ])
                elif insert_type == 'ee':
                    raw_face = self.patch_mesh.f[f_id]
                    
                    prev_e_id = prev_v_id
                    cur_e_id = v_from_id
                    op_v_id0 = e_h2ov_in_face(mesh, mesh.edge_handle(prev_e_id), raw_face)
                    op_v_index0 = np.where(raw_face == op_v_id0)[0][0]
                    op_v_id1 = e_h2ov_in_face(mesh, mesh.edge_handle(cur_e_id), raw_face)
                    op_v_index1 = np.where(raw_face == op_v_id1)[0][0]

            
                    if op_v_index1 == (op_v_index0 + 1) % 3:
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            raw_face[(op_v_index0 + 1) % 3],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            prev_insert_v_id,
                            raw_face[(op_v_index0 - 1) % 3],
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                    else:
                        insert_f[f_id].append([
                            raw_face[(op_v_index0 - 1) % 3],
                            raw_face[op_v_index0],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[(op_v_index0 + 1) % 3],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            insert_v_id,
                            prev_insert_v_id
                        ])
            
        for f_id in insert_f.keys():
            insert_f[f_id] = np.array(insert_f[f_id])

        # update vertices
        if len(insert_v) > 0:
            insert_v = np.array(insert_v)
            new_mesh_vertices = np.concatenate((self.patch_mesh.v, insert_v), axis=0)
        else:
            new_mesh_vertices = self.patch_mesh.v

        # remove old faces & insert new faces with patch id
        remove_f = list(insert_f.keys())
        new_mesh_faces = np.delete(self.patch_mesh.f, remove_f, axis = 0)
        new_f_id2raw_f_id = [i for i in range(len(self.patch_mesh.f)) if i not in remove_f]
        raw_f_len = new_mesh_faces.shape[0]
        
        new_f_p_id = {}
        new_f_id = raw_f_len
        for raw_f_id in insert_f.keys():
            for _ in range(insert_f[raw_f_id].shape[0]):
                new_f_p_id[new_f_id] = self.patch_mesh.f2patch[raw_f_id]
                new_f_id += 1
            new_mesh_faces = np.concatenate((new_mesh_faces, insert_f[raw_f_id]))
        
        # check
        f_colors = np.ones((new_mesh_faces.shape[0], 3))
        # color = colors[41]
        # f_colors = np.tile(color, (new_mesh_faces.shape[0], 1))
        plot = mp.plot(new_mesh_vertices, new_mesh_faces, f_colors, shading={'wireframe': False, 'point_size': 0})
        cutting_border_vert_array = np.array(cutting_border_vert)
        plot.add_lines(
            new_mesh_vertices[cutting_border_vert_array[:, 0]],
            new_mesh_vertices[cutting_border_vert_array[:, 1]],
            shading={
                "line_color": "red",
                "line_width": 5
            }
        )

        # split mesh
        # noting the insert edge, and split by connectivity of face Graph
        cutting_border_vert_ordered = []
        for v_pair in cutting_border_vert:
            v0_id, v1_id = v_pair
            if v0_id > v1_id:
                cutting_border_vert_ordered.append([v1_id, v0_id])
            else:
                cutting_border_vert_ordered.append([v0_id, v1_id])
        # cutting_border_vert = np.array(cutting_border_vert)
        # cutting_mesh = om.TriMesh(new_mesh_vertices, new_mesh_faces)
        # cutting_border_edge = []
        # for i in range(cutting_border_vert.shape[0]):
        #     v0_id, v1_id = cutting_border_edge[i]
        #     cutting_border_edge.append(v_pair2e(cutting_mesh, v0_id, v1_id))
        
        cutting_graph = nx.Graph()
        # add node
        for i in range(new_mesh_faces.shape[0]):
            cutting_graph.add_node(i)
        # edge to face
        edge2face = defaultdict(list)
        for f_id in range(new_mesh_faces.shape[0]):
            for i in range(3):
                v0, v1 = new_mesh_faces[f_id][i], new_mesh_faces[f_id][(i + 1) % 3]
                if v0 > v1:
                    v0, v1 = v1, v0
                edge2face[(v0, v1)].append(f_id)
        # add edges
        for edge, n_f in edge2face.items():
            if len(n_f) == 2:
                if [edge[0], edge[1]] in cutting_border_vert_ordered:
                    continue
                cutting_graph.add_edge(n_f[0], n_f[1])
        components = list(nx.connected_components(cutting_graph))

        f_splited = [list(components[i]) for i in range(len(components))]
        new_patch2f = []
        for i in range(len(f_splited)):
            patch2f_i = defaultdict(list)
            for f_id in f_splited[i]:
                if f_id >= raw_f_len:
                    p_id = new_f_p_id[f_id]
                else:
                    raw_f_id = new_f_id2raw_f_id[f_id]
                    p_id = self.patch_mesh.f2patch[raw_f_id]
                patch2f_i[p_id].append(f_id)
            patch2f_i = [patch2f_i[p_id] for p_id in patch2f_i]

            for j in range(len(patch2f_i)):
                for k in range(len(patch2f_i[j])):
                    if patch2f_i[j][k] in f_splited[i]:
                        patch2f_i[j][k] = f_splited[i].index(patch2f_i[j][k])
            new_patch2f.append(patch2f_i)
        
        objs = []
        for i in range(len(f_splited)):
            vi, fi = get_submesh(new_mesh_vertices, new_mesh_faces, f_splited[i])
            # hole filling
            objs.append({
                'v': vi,
                'f': fi,
                'patch': new_patch2f[i]
            })
            mp.plot(vi, fi, np.ones((fi.shape[0], 3)), shading={'wireframe': False})
        return True, objs


    '''
        Cut the mesh by cylinder for rotation symmetry items
            - for p_id, next_p_id: cut from common_patch_border[p_id][v_to] to common_patch_border[next_p_id][v_from]
        INPUT
            - common_patch: ordered p_id list
            - common_patch_border: [dict{'v_border', 'v_from', 'v_to'}]
    '''
    def cut_mesh_by_rotation_symmetric_cylinder_tracing(self, cylinder_param, common_patch, common_patch_border):
        mesh = self.patch_mesh.om_mesh

        cutting_border_vert = []
        v_border = []
        for p_id in common_patch_border:
            for i in range(2):
                v_border.extend(common_patch_border[p_id][i]['v_border'])
                cutting_border_vert.append([common_patch_border[p_id][i]['v_border'][0], common_patch_border[p_id][i]['v_border'][-1]])
            # for i in range(2):
            #     v_border.extend(common_patch_border[p_id][i]['v_border'])
            #     for j in range(len(v_border) - 1):
            #         cutting_border_vert.append([v_border[j], v_border[j + 1]])
        
        insert_v = []
        insert_v_num = -1
        raw_v_num = self.patch_mesh.v.shape[0]
        insert_f = defaultdict(list)
        f_visit = set()


        f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={
            'wireframe': False
        })
        for i in range(len(common_patch)):
            p_id = common_patch[i]
            next_p_id = common_patch[(i + 1) % len(common_patch)]

            # initial
            # v_from_id = common_patch_border[p_id][0]['v_from']
            v_from_id = common_patch_border[p_id][0]['v_to']
            v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
            v_from_type = 'V'
            # v_to = common_patch_border[next_p_id][1]['v_from']
            v_to = common_patch_border[next_p_id][1]['v_to']

            plot.add_points(
                np.array([v_from_pos]),
                shading={
                    'point_color': 'red',
                    'point_size': 1
                }
            )
            plot.add_points(
                np.array([mesh.point(mesh.vertex_handle(v_to))]),
                shading={
                    'point_color': 'blue',
                    'point_size': 1
                }
            )

            prev_v_pos = v_from_pos
            prev_v_id = v_from_id


            # v_from_id = v_from
            # v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
            # v_from_type = 'V'
            # prev_v_pos = v_from_pos
            # prev_v_id = v_from_id

            # cutting_border_vert = []
            # for i in range(len(v_border) - 1):
            #     cutting_border_vert.append([v_border[i], v_border[i + 1]])
            
            # first direction: vertex to edge

            while True:
                if v_from_id == v_to:
                    print('Reach v_border')
                    break
                plot.add_lines(
                    np.array([prev_v_pos]),
                    np.array([v_from_pos]),
                    shading={
                        "line_color": "red",
                        "line_width": 1
                    }
                )
                plot.update_object()

                if v_from_type == 'V':
                    v_from_h = mesh.vertex_handle(v_from_id)
                else:
                    v_from_h = mesh.edge_handle(v_from_id)

                insert_type = None
                if v_from_type == 'V':
                    neighbor_f = []
                    for f_h in mesh.vf(v_from_h):
                        f_id = f_h.idx()
                        if f_id not in f_visit:
                            neighbor_f.append(f_id)
                            f_visit.add(f_id)
                    
                    next_v_info = []
                    for f_id in neighbor_f:
                        result = insection_cylinder_f_by_vert(mesh, cylinder_param, mesh.face_handle(f_id), v_from_pos, v_from_type, v_from_id)
                        f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
                        f_colors[f_id] = [1, 0, 0]
                        plot.update_object(colors=f_colors)
                        if (result['type'] == 've'):
                            next_v_info.append(result)
                            break
                        if (result['type'] == 'vv'):
                            next_v_info.append(result)
                    if len(next_v_info) == 0:
                        print('Cannot find next intersect face')
                        break



                    # get right next vert
                    # 1. ve > vv
                    # 2. remove vert in v_border
                    # 3. choose the duplicate one
                    next_v_info_id = -1
                    next_v_set = defaultdict(list)
                    for i in range(len(next_v_info)):
                        tmp_next_v_info = next_v_info[i]
                        if tmp_next_v_info['type'] == 've':
                            next_v_info_id = i
                            break
                        elif tmp_next_v_info['next_v_id'] == v_to:
                            next_v_info_id = i
                            break                       
                        elif tmp_next_v_info['next_v_id'] in v_border:
                            continue
                        else:
                            tmp_next_v_id = tmp_next_v_info['next_v_id']
                            if tmp_next_v_id not in next_v_set:
                                next_v_info_id = i
                                next_v_set[tmp_next_v_id].append(tmp_next_v_info)
                            else:
                                next_v_info_id = i
                                break
                    if next_v_info_id == -1:
                        raise('[Trace Cut]: cannot find next v id')
                    next_v_info = next_v_info[next_v_info_id]
                    # next_v_info = next_v_info[0]

                    if next_v_info['type'] == 'vv':
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = next_v_info['next_v_id']
                        v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                        v_from_type = 'V'
                        f_id = next_v_info['f_id']

                        insert_type = 'vv'
                        cutting_border_vert.append([prev_v_id, v_from_id])
                    else:
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = next_v_info['next_e_id']
                        v_from_pos = next_v_info['v_insert']
                        v_from_type = 'E'
                        f_id = next_v_info['f_id']
                        
                        insert_type = 've'
                        insert_v.append(v_from_pos)
                        insert_v_num += 1
                        insert_v_id = raw_v_num + insert_v_num
                        cutting_border_vert.append([prev_v_id, insert_v_id])
                        

                elif v_from_type == 'E':
                    neighbor_f = -1
                    for f_id in e_h2f(mesh, v_from_h):
                    # for f_h in mesh.ef(v_from_h):
                        # f_id = f_h.idx()
                        if f_id not in f_visit:
                            neighbor_f = f_id
                            f_visit.add(f_id)
                    if neighbor_f == -1:
                        print('Cannot find next not visited face')
                        break
                    
                    result = insection_cylinder_f_by_vert(mesh, cylinder_param, mesh.face_handle(neighbor_f), v_from_pos, v_from_type, v_from_id)

                    if result['type'] == 'ev':
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = result['next_v_id']
                        v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                        v_from_type = 'V'
                        f_id = result['f_id']

                        insert_type = 'ev'
                        prev_insert_v_id = raw_v_num + insert_v_num
                        cutting_border_vert.append([prev_insert_v_id, v_from_id])

                    elif result['type'] == 'ee':
                        prev_v_id = v_from_id
                        prev_v_pos = v_from_pos
                        v_from_id = result['to_e_id']
                        v_from_pos = result['v_insert']
                        v_from_type = 'E'
                        f_id = result['f_id']

                        insert_type = 'ee'
                        prev_insert_v_id = raw_v_num + insert_v_num
                        insert_v.append(v_from_pos)
                        insert_v_num += 1
                        insert_v_id = raw_v_num + insert_v_num
                        cutting_border_vert.append([prev_insert_v_id, insert_v_id])
                

                # insert faces
                if insert_type == 'vv':
                    pass
                elif insert_type == 've':
                    raw_face = self.patch_mesh.f[f_id]
                    pre_v_index = np.where(raw_face == prev_v_id)[0][0]

                    insert_f[f_id].append([
                        raw_face[pre_v_index],
                        raw_face[(pre_v_index + 1) % 3],
                        insert_v_id
                    ])
                    insert_f[f_id].append([
                        raw_face[(pre_v_index - 1) % 3],
                        raw_face[pre_v_index],
                        insert_v_id
                    ])
                elif insert_type == 'ev':
                    raw_face = self.patch_mesh.f[f_id]
                    pre_v_index = np.where(raw_face == v_from_id)[0][0]

                    insert_f[f_id].append([
                        raw_face[pre_v_index],
                        raw_face[(pre_v_index + 1) % 3],
                        prev_insert_v_id
                    ])
                    insert_f[f_id].append([
                        raw_face[(pre_v_index - 1) % 3],
                        raw_face[pre_v_index],
                        prev_insert_v_id
                    ])
                elif insert_type == 'ee':
                    raw_face = self.patch_mesh.f[f_id]
                    
                    prev_e_id = prev_v_id
                    cur_e_id = v_from_id
                    op_v_id0 = e_h2ov_in_face(mesh, mesh.edge_handle(prev_e_id), raw_face)
                    op_v_index0 = np.where(raw_face == op_v_id0)[0][0]
                    op_v_id1 = e_h2ov_in_face(mesh, mesh.edge_handle(cur_e_id), raw_face)
                    op_v_index1 = np.where(raw_face == op_v_id1)[0][0]

            
                    if op_v_index1 == (op_v_index0 + 1) % 3:
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            raw_face[(op_v_index0 + 1) % 3],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            prev_insert_v_id,
                            raw_face[(op_v_index0 - 1) % 3],
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                    else:
                        insert_f[f_id].append([
                            raw_face[(op_v_index0 - 1) % 3],
                            raw_face[op_v_index0],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[(op_v_index0 + 1) % 3],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            insert_v_id,
                            prev_insert_v_id
                        ])
            
        for f_id in insert_f.keys():
            insert_f[f_id] = np.array(insert_f[f_id])

        # update vertices
        if len(insert_v) > 0:
            insert_v = np.array(insert_v)
            new_mesh_vertices = np.concatenate((self.patch_mesh.v, insert_v), axis=0)
        else:
            new_mesh_vertices = self.patch_mesh.v

        # remove old faces & insert new faces with patch id
        remove_f = list(insert_f.keys())
        new_mesh_faces = np.delete(self.patch_mesh.f, remove_f, axis = 0)
        new_f_id2raw_f_id = [i for i in range(len(self.patch_mesh.f)) if i not in remove_f]
        raw_f_len = new_mesh_faces.shape[0]
        
        new_f_p_id = {}
        new_f_id = raw_f_len
        for raw_f_id in insert_f.keys():
            for _ in range(insert_f[raw_f_id].shape[0]):
                new_f_p_id[new_f_id] = self.patch_mesh.f2patch[raw_f_id]
                new_f_id += 1
            new_mesh_faces = np.concatenate((new_mesh_faces, insert_f[raw_f_id]))
        
        # check
        f_colors = np.tile(colors[61], (new_mesh_faces.shape[0], 1))
        plot = mp.plot(new_mesh_vertices, new_mesh_faces, f_colors, shading={'wireframe': False, 'point_size': 0})
        cutting_border_vert_array = np.array(cutting_border_vert)
        plot.add_lines(
            new_mesh_vertices[cutting_border_vert_array[:, 0]],
            new_mesh_vertices[cutting_border_vert_array[:, 1]],
            shading={
                "line_color": "red",
                "line_width": 3
            }
        )

        # split mesh
        # noting the insert edge, and split by connectivity of face Graph
        cutting_border_vert_ordered = []
        for v_pair in cutting_border_vert:
            v0_id, v1_id = v_pair
            if v0_id > v1_id:
                cutting_border_vert_ordered.append([v1_id, v0_id])
            else:
                cutting_border_vert_ordered.append([v0_id, v1_id])
        # cutting_border_vert = np.array(cutting_border_vert)
        # cutting_mesh = om.TriMesh(new_mesh_vertices, new_mesh_faces)
        # cutting_border_edge = []
        # for i in range(cutting_border_vert.shape[0]):
        #     v0_id, v1_id = cutting_border_edge[i]
        #     cutting_border_edge.append(v_pair2e(cutting_mesh, v0_id, v1_id))
        
        cutting_graph = nx.Graph()
        # add node
        for i in range(new_mesh_faces.shape[0]):
            cutting_graph.add_node(i)
        # edge to face
        edge2face = defaultdict(list)
        for f_id in range(new_mesh_faces.shape[0]):
            for i in range(3):
                v0, v1 = new_mesh_faces[f_id][i], new_mesh_faces[f_id][(i + 1) % 3]
                if v0 > v1:
                    v0, v1 = v1, v0
                edge2face[(v0, v1)].append(f_id)
        # add edges
        for edge, n_f in edge2face.items():
            if len(n_f) == 2:
                if [edge[0], edge[1]] in cutting_border_vert_ordered:
                    continue
                cutting_graph.add_edge(n_f[0], n_f[1])
        components = list(nx.connected_components(cutting_graph))

        f_splited = [list(components[i]) for i in range(len(components))]
        new_patch2f = []
        for i in range(len(f_splited)):
            patch2f_i = defaultdict(list)
            for f_id in f_splited[i]:
                if f_id >= raw_f_len:
                    p_id = new_f_p_id[f_id]
                else:
                    raw_f_id = new_f_id2raw_f_id[f_id]
                    p_id = self.patch_mesh.f2patch[raw_f_id]
                patch2f_i[p_id].append(f_id)
            patch2f_i = [patch2f_i[p_id] for p_id in patch2f_i]

            for j in range(len(patch2f_i)):
                for k in range(len(patch2f_i[j])):
                    if patch2f_i[j][k] in f_splited[i]:
                        patch2f_i[j][k] = f_splited[i].index(patch2f_i[j][k])
            new_patch2f.append(patch2f_i)
        
        objs = []
        for i in range(len(f_splited)):
            vi, fi = get_submesh(new_mesh_vertices, new_mesh_faces, f_splited[i])
            objs.append({
                'v': vi,
                'f': fi,
                'patch': new_patch2f[i]
            })
            mp.plot(vi, fi, np.ones((fi.shape[0], 3)), shading={'wireframe': False})
        return True, objs
    


    def cut_mesh_by_cylinder_tracing(self, cylinder_params, v_from, v_to, v_border):
        # v_border = self.get_patch_border_v(p1, p2)
        # v_from, v_to = v_border[0], v_border[-1]
        # v_to_pos = self.patch_mesh.v[v_to]
        mesh = self.patch_mesh.om_mesh

        c, axis, r = cylinder_params[:3].squeeze(), cylinder_params[3:6].squeeze(), cylinder_params[6].item()
        threshold_sample_len = r * np.pi / 18. / 2.

        v_from_id = v_from
        v_from_pos = self.patch_mesh.v[v_from_id]
        # v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
        v_from_type = 'V'
        f_visit = set()
        prev_v_pos = v_from_pos
        prev_v_id = v_from_id
        prev_insert_v_id = -1

        insert_v = []
        insert_v_num = -1
        raw_insert_v_num = 0
        raw_v_num = self.patch_mesh.v.shape[0]
        insert_f = defaultdict(list)
        del_f = []
        
        f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        # f_colors[self.patch_mesh.patch2f[7]]= [0, 1, 1]
        plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={
            'wireframe': True
        })
        plot.add_points(
            self.patch_mesh.v[v_border],
            shading={
                'point_color': 'blue',
                'point_size': 5
            }
        )
        cutting_border_vert = []
        for i in range(len(v_border) - 1):
            cutting_border_vert.append([v_border[i], v_border[i + 1]])

        while True:
            
            if v_from_id == v_to:
                print('Reach v_border')
                break
            # plot.update_object(colors=np.ones((self.patch_mesh.f.shape[0], 3)))
            # plot.add_lines(
            #     np.array([prev_v_pos]),
            #     np.array([v_from_pos]),
            #     shading={
            #         "line_color": "red",
            #         "line_width": 1
            #     }
            # )
            # if insert_v_num > raw_insert_v_num:
            #     tmp_insert_v = np.array(insert_v[raw_insert_v_num:])
            #     raw_insert_v_num = insert_v_num
            #     plot.add_points(
            #         tmp_insert_v,
            #         shading={
            #             'point_color': 'red',
            #             'point_size': 5
            #         }
            #     )
            #     plot.update_object()
            # else:
            #     plot.add_lines(
            #         np.array([prev_v_pos]),
            #         np.array([v_from_pos]),
            #         shading={
            #             "line_color": "red",
            #             "line_width": 1
            #         }
            #     )


            if v_from_type == 'V':
                v_from_h = mesh.vertex_handle(v_from_id)
            else:
                v_from_h = mesh.edge_handle(v_from_id)
            
            insert_type = None
            sample = False
            sample_num = 0
            if v_from_type == 'V':
                neighbor_f = []
                for f_h in mesh.vf(v_from_h):
                    f_id = f_h.idx()
                    if f_id not in f_visit:
                        neighbor_f.append(f_id)
                        f_visit.add(f_id)
                
                next_v_info = []
                for f_id in neighbor_f:
                    # f_colors[f_id] = [1, 0, 0]
                    # plot.update_object(colors=f_colors)
                    # result = insection_cylinder_f(mesh, cylinder_params, mesh.face_handle(f_id), v_from_pos)
                    result = insection_cylinder_f_by_vert(mesh, cylinder_params, mesh.face_handle(f_id), v_from_pos, v_from_type, v_from_id)
                    if (result['type'] == 've'):
                        next_v_info.append(result)
                        break
                    if (result['type'] == 'vv'):
                        next_v_info.append(result)
                if len(next_v_info) == 0:
                    print('Cannot find next intersect face')
                    break

                # get right next vert
                # 1. ve > vv
                # 2. remove vert in v_border
                # 3. choose the duplicate one
                next_v_info_id = -1
                next_v_set = defaultdict(list)
                for i in range(len(next_v_info)):
                    tmp_next_v_info = next_v_info[i]
                    if tmp_next_v_info['type'] == 've':
                        next_v_info_id = i
                        break
                    elif tmp_next_v_info['next_v_id'] == v_to:
                        next_v_info_id = i
                        break                       
                    elif tmp_next_v_info['next_v_id'] in v_border:
                        continue
                    else:
                        tmp_next_v_id = tmp_next_v_info['next_v_id']
                        if tmp_next_v_id not in next_v_set:
                            next_v_info_id = i
                            next_v_set[tmp_next_v_id].append(tmp_next_v_info)
                        else:
                            next_v_info_id = i
                            break
                if next_v_info_id == -1:
                    raise('[Trace Cut]: cannot find next v id')
                next_v_info = next_v_info[next_v_info_id]

                if next_v_info['type'] == 'vv':
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = next_v_info['next_v_id']
                    v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                    v_from_type = 'V'
                    f_id = next_v_info['f_id']

                    insert_type = 'vv'
                    # cutting_dist = np.linalg.norm(v_from_pos - prev_v_pos)
                    # cutting_border_vert.append([prev_v_id, v_from_id])
                           
                else:
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = next_v_info['next_e_id']
                    v_from_pos = next_v_info['v_insert']
                    v_from_type = 'E'
                    f_id = next_v_info['f_id']

                    insert_type = 've'
                    insert_v.append(v_from_pos)
                    insert_v_num += 1
                    insert_v_id = raw_v_num + insert_v_num
                    # prev_insert_v_id = insert_v_id
                    # cutting_border_vert.append([prev_v_id, insert_v_id])
                    # cutting_dist = np.linalg.norm(v_from_pos - prev_v_pos)
                    # insert_v.append(v_from_pos)
                    # insert_v_num += 1
                    # insert_v_id = raw_v_num + insert_v_num
                    # cutting_border_vert.append([prev_v_id, insert_v_id])
            
            elif v_from_type == 'E':
                neighbor_f = -1
                for f_id in e_h2f(mesh, v_from_h):
                    if f_id not in f_visit:
                        neighbor_f = f_id
                        f_visit.add(f_id)
                if neighbor_f == -1:
                    print('Cannot find next not visited face')
                    break
                
                result = insection_cylinder_f_by_vert(mesh, cylinder_params, mesh.face_handle(neighbor_f), v_from_pos, v_from_type, v_from_id)
                # result = insection_cylinder_f(mesh, cylinder_params, mesh.face_handle(neighbor_f), v_from_pos)

                if result['type'] == 'ev':
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = result['next_v_id']
                    v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
                    v_from_type = 'V'
                    f_id = result['f_id']

                    insert_type = 'ev'
                    # prev_insert_v_id = raw_v_num + insert_v_num
                    # cutting_border_vert.append([prev_insert_v_id, v_from_id])

                elif result['type'] == 'ee':
                    prev_v_id = v_from_id
                    prev_v_pos = v_from_pos
                    v_from_id = result['to_e_id']
                    v_from_pos = result['v_insert']
                    v_from_type = 'E'
                    f_id = result['f_id']

                    insert_type = 'ee'
                    # prev_insert_v_id = raw_v_num + insert_v_num
                    insert_v.append(v_from_pos)
                    insert_v_num += 1
                    insert_v_id = raw_v_num + insert_v_num
                    # cutting_border_vert.append([prev_insert_v_id, insert_v_id])
                    # prev_insert_v_id = insert_v_id

                
            # insert faces
            if insert_type == 'vv':

                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_params, f_id, 
                    prev_v_pos, v_from_pos, threshold_sample_len
                )

                if sample_points is None:
                    cutting_border_vert.append([prev_v_id, v_from_id])
                    continue
                
                sample_num = sample_points.shape[0]
                insert_v_range = range(raw_v_num + insert_v_num, raw_v_num + insert_v_num + sample_num)
                insert_v_num += sample_num
                for i in range(sample_num):
                    insert_v.append(sample_points[i])
            
                cutting_border_vert.append([prev_v_id, insert_v_range[0]])
                for i in range(sample_num - 1):
                    cutting_border_vert.append([insert_v_range[i], insert_v_range[i + 1]])
                cutting_border_vert.append([insert_v_range[-1], v_from_id])

                # triangulate
                mid_v = (prev_v_pos + v_from_pos) / 2.
                insert_v.append(mid_v)
                insert_v_num += 1
                mid_v_id = raw_v_num + insert_v_num

                op_v_id = v2op_v(mesh, prev_v_id, v_from_id, f_id)
                cur_e_id = v_pair2e(mesh, prev_v_id, v_from_id)
                op_f_id = e2op_f(mesh, cur_e_id, f_id)
                op_f_op_v_id = e_h2op_v_in_f_h(mesh, mesh.edge_handle(cur_e_id), mesh.face_handle(op_f_id))
                
                # 1. samples -- op_v_id
                insert_f[f_id].append([
                    prev_v_id,
                    op_v_id, 
                    insert_v_range[0]
                ])
                for i in range(0, sample_num - 1):
                    insert_f[f_id].append([
                        insert_v_range[i],
                        insert_v_range[i+1],
                        op_v_id
                    ])
                insert_f[f_id].append([
                    insert_v_range[-1],
                    op_v_id,
                    v_from_id
                ])


                # 2. samples -- mid
                insert_f[f_id].append([
                    prev_v_id,
                    mid_v_id, 
                    insert_v_range[0]
                ])
                for i in range(0, sample_num - 1):
                    insert_f[f_id].append([
                        insert_v_range[i],
                        insert_v_range[i+1],
                        mid_v_id
                    ])
                insert_f[f_id].append([
                    insert_v_range[-1],
                    mid_v_id,
                    v_from_id
                ])


                # 3. mid -- op_v_id
                del_f.append(op_f_id)
                insert_f[op_f_id].append([
                    prev_v_id,
                    mid_v_id,
                    op_f_op_v_id
                ])
                insert_f[op_f_id].append([
                    op_f_op_v_id,
                    mid_v_id,
                    v_from_id
                ])


            elif insert_type == 've':
                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_params, f_id, 
                    prev_v_pos, v_from_pos, threshold_sample_len
                )

                if sample_points is not None:
                    
                
                    sample_num = sample_points.shape[0]
                    insert_v_range = range(raw_v_num + insert_v_num + 1, raw_v_num + insert_v_num + 1 + sample_num)
                    insert_v_num += sample_num
                    for i in range(sample_num):
                        insert_v.append(sample_points[i])
                
                    cutting_border_vert.append([prev_v_id, insert_v_range[0]])
                    for i in range(sample_num - 1):
                        cutting_border_vert.append([insert_v_range[i], insert_v_range[i + 1]])
                    cutting_border_vert.append([insert_v_range[-1], insert_v_id])

                    # triangulate
                    n_v0_id, n_v1_id = e_h2v(mesh, mesh.edge_handle(v_from_id))

                    # 1. samples -- n_v0_id
                    insert_f[f_id].append([
                        prev_v_id,
                        insert_v_range[0],
                        n_v0_id
                    ])
                    for i in range(0, sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            n_v0_id,
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        insert_v_id,
                        n_v0_id
                    ])


                    # 2. samples -- n_v1_id
                    insert_f[f_id].append([
                        prev_v_id,
                        insert_v_range[0],
                        n_v1_id
                    ])
                    for i in range(0, sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            n_v1_id,
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        insert_v_id,
                        n_v1_id,
                    ])


                # raw_face = self.patch_mesh.f[f_id]
                # pre_v_index = np.where(raw_face == prev_v_id)[0][0]

                # insert_f[f_id].append([
                #     raw_face[pre_v_index],
                #     raw_face[(pre_v_index + 1) % 3],
                #     insert_v_id
                # ])
                # insert_f[f_id].append([
                #     raw_face[(pre_v_index - 1) % 3],
                #     raw_face[pre_v_index],
                #     insert_v_id
                # ])
            
            elif insert_type == 'ev':
                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_params, f_id, 
                    prev_v_pos, v_from_pos, threshold_sample_len
                )

                if sample_points is None:
                    continue
                
                sample_num = sample_points.shape[0]
                insert_v_range = range(raw_v_num + insert_v_num + 1, raw_v_num + insert_v_num + 1 + sample_num)
                insert_v_num += sample_num
                for i in range(sample_num):
                    insert_v.append(sample_points[i])

                cutting_border_vert.append([prev_insert_v_id, insert_v_range[0]])
                for i in range(sample_num - 1):
                    cutting_border_vert.append([insert_v_range[i], insert_v_range[i + 1]])
                cutting_border_vert.append([insert_v_range[-1], v_from_id])

                # triangulate

                n_v0_id, n_v1_id = e_h2v(mesh, mesh.edge_handle(prev_v_id))

                # 1. samples -- n_v0_id
                insert_f[f_id].append([
                    prev_insert_v_id,
                    insert_v_range[0],
                    n_v0_id
                ])
                for i in range(0, sample_num - 1):
                    insert_f[f_id].append([
                        insert_v_range[i],
                        insert_v_range[i+1],
                        n_v0_id,
                    ])
                insert_f[f_id].append([
                    insert_v_range[-1],
                    v_from_id,
                    n_v0_id
                ])


                # 2. samples -- n_v1_id
                insert_f[f_id].append([
                    prev_insert_v_id,
                    insert_v_range[0],
                    n_v1_id
                ])
                for i in range(0, sample_num - 1):
                    insert_f[f_id].append([
                        insert_v_range[i],
                        insert_v_range[i+1],
                        n_v1_id,
                    ])
                insert_f[f_id].append([
                    insert_v_range[-1],
                    v_from_id,
                    n_v1_id,
                ])

                # raw_face = self.patch_mesh.f[f_id]
                # pre_v_index = np.where(raw_face == v_from_id)[0][0]

                # insert_f[f_id].append([
                #     raw_face[pre_v_index],
                #     raw_face[(pre_v_index + 1) % 3],
                #     prev_insert_v_id
                # ])
                # insert_f[f_id].append([
                #     raw_face[(pre_v_index - 1) % 3],
                #     raw_face[pre_v_index],
                #     prev_insert_v_id
                # ])
            
            elif insert_type == 'ee':
                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_params, f_id, 
                    prev_v_pos, v_from_pos, threshold_sample_len
                )

                if sample_points is None:
                    cutting_border_vert.append([prev_insert_v_id, insert_v_id])

                    raw_face = self.patch_mesh.f[f_id]
                    
                    prev_e_id = prev_v_id
                    cur_e_id = v_from_id
                    op_v_id0 = e_h2ov_in_face(mesh, mesh.edge_handle(prev_e_id), raw_face)
                    op_v_index0 = np.where(raw_face == op_v_id0)[0][0]
                    op_v_id1 = e_h2ov_in_face(mesh, mesh.edge_handle(cur_e_id), raw_face)
                    op_v_index1 = np.where(raw_face == op_v_id1)[0][0]

            
                    if op_v_index1 == (op_v_index0 + 1) % 3:
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            raw_face[(op_v_index0 + 1) % 3],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            prev_insert_v_id,
                            raw_face[(op_v_index0 - 1) % 3],
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                    else:
                        insert_f[f_id].append([
                            raw_face[(op_v_index0 - 1) % 3],
                            raw_face[op_v_index0],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[(op_v_index0 + 1) % 3],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            raw_face[op_v_index0],
                            insert_v_id,
                            prev_insert_v_id
                        ])
                else:
                
                    sample_num = sample_points.shape[0]
                    insert_v_range = range(raw_v_num + insert_v_num + 1, raw_v_num + insert_v_num + 1 + sample_num)
                    insert_v_num += sample_num
                    for i in range(sample_num):
                        insert_v.append(sample_points[i])
                
                    cutting_border_vert.append([prev_insert_v_id, insert_v_range[0]])
                    for i in range(sample_num - 1):
                        cutting_border_vert.append([insert_v_range[i], insert_v_range[i + 1]])
                    cutting_border_vert.append([insert_v_range[-1], insert_v_id])

                    # triangulate
                    # mid_sample_id = raw_v_num + insert_v_num + int(sample_num / 2)

                    common_v_id = e_pair2v(mesh, prev_v_id, v_from_id)
                    prev_ano_v_id = e2ano_v(mesh, prev_v_id, common_v_id)
                    cur_ano_v_id = e2ano_v(mesh, v_from_id, common_v_id)

                    mid_v = (mesh.point(mesh.vertex_handle(prev_ano_v_id)) + mesh.point(mesh.vertex_handle(cur_ano_v_id))) / 2.
                    insert_v.append(mid_v)
                    insert_v_num += 1
                    mid_v_id = raw_v_num + insert_v_num


                    op_e_id = v_pair2e(mesh, prev_ano_v_id, cur_ano_v_id)
                    op_f_id = e2op_f(mesh, op_e_id, f_id)
                    op_f_op_v_id = e_h2op_v_in_f_h(mesh, mesh.edge_handle(op_e_id), mesh.face_handle(op_f_id))

                    
                    # 1. samples -- common_v_id
                    insert_f[f_id].append([
                        prev_insert_v_id,
                        common_v_id, 
                        insert_v_range[0]
                    ])
                    for i in range(0, sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            common_v_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        common_v_id, 
                        insert_v_id
                    ])

                    
                    # 2.1 part1-samples -- mid
                    insert_f[f_id].append([
                        prev_insert_v_id,
                        prev_ano_v_id, 
                        insert_v_range[0]
                    ])
                    for i in range(0, int((sample_num - 1) / 2)):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            prev_ano_v_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[int((sample_num - 1) / 2)],
                        mid_v_id,
                        prev_ano_v_id
                    ])

                    # 2.2 part2-samples -- mid
                    insert_f[f_id].append([
                        insert_v_range[int((sample_num - 1) / 2)],
                        mid_v_id,
                        cur_ano_v_id
                    ])
                    for i in range(int((sample_num - 1) / 2), sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            cur_ano_v_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        insert_v_id,
                        cur_ano_v_id
                    ])


                    # 3. mid -- op_v_id
                    del_f.append(op_f_id)
                    insert_f[op_f_id].append([
                        prev_ano_v_id,
                        mid_v_id,
                        op_f_op_v_id
                    ])
                    insert_f[op_f_id].append([
                        op_f_op_v_id,
                        mid_v_id,
                        cur_ano_v_id
                    ])

            prev_insert_v_id = insert_v_id

        # update vertices
        insert_v = np.array(insert_v)
        new_mesh_vertices = np.concatenate((self.patch_mesh.v, insert_v), axis=0)
        
        # check normal direction
        for f_id in insert_f:
            insert_f[f_id] = np.array(insert_f[f_id])

            raw_face_normal = face_normal(new_mesh_vertices, self.patch_mesh.f[f_id])

            insert_face_normal = [face_normal(new_mesh_vertices, insert_f[f_id][i]) for i in range(insert_f[f_id].shape[0])]

            for i in range(len(insert_face_normal)):
                if np.dot(raw_face_normal, insert_face_normal[i]) < 0:
                    insert_f[f_id][i] = insert_f[f_id][i][[0, 2, 1]]
        
        
              

        # remove old faces & insert new faces with patch id
        remove_f = list(set(insert_f.keys()) | set(del_f))
        new_mesh_faces = np.delete(self.patch_mesh.f, remove_f, axis = 0)
        new_f_id2raw_f_id = [i for i in range(len(self.patch_mesh.f)) if i not in remove_f]
        raw_f_len = new_mesh_faces.shape[0]
        
        new_f_p_id = {}
        new_f_id = raw_f_len
        for raw_f_id in insert_f.keys():
            for _ in range(insert_f[raw_f_id].shape[0]):
                new_f_p_id[new_f_id] = self.patch_mesh.f2patch[raw_f_id]
                new_f_id += 1
            new_mesh_faces = np.concatenate((new_mesh_faces, insert_f[raw_f_id]))
        1
        # check
        plot = mp.plot(new_mesh_vertices, new_mesh_faces, np.ones((new_mesh_faces.shape[0], 3)), shading={'wireframe': True})
        cutting_border_vert_array = np.array(cutting_border_vert)
        plot.add_lines(
            new_mesh_vertices[cutting_border_vert_array[:, 0]],
            new_mesh_vertices[cutting_border_vert_array[:, 1]],
            shading={
                "line_color": "red",
                "line_width": 1
            }
        )

        # split mesh
        # noting the insert edge, and split by connectivity of face Graph
        cutting_border_vert_ordered = []
        for v_pair in cutting_border_vert:
            v0_id, v1_id = v_pair
            if v0_id > v1_id:
                cutting_border_vert_ordered.append([v1_id, v0_id])
            else:
                cutting_border_vert_ordered.append([v0_id, v1_id])

        cutting_graph = nx.Graph()
        # add node
        for i in range(new_mesh_faces.shape[0]):
            cutting_graph.add_node(i)
        # edge to face
        edge2face = defaultdict(list)
        for f_id in range(new_mesh_faces.shape[0]):
            for i in range(3):
                v0, v1 = new_mesh_faces[f_id][i], new_mesh_faces[f_id][(i + 1) % 3]
                if v0 > v1:
                    v0, v1 = v1, v0
                edge2face[(v0, v1)].append(f_id)
        # add edges
        for edge, n_f in edge2face.items():
            if len(n_f) == 2:
                if [edge[0], edge[1]] in cutting_border_vert_ordered:
                    continue
                cutting_graph.add_edge(n_f[0], n_f[1])
        components = list(nx.connected_components(cutting_graph))
        if len(components) != 2:
            raise ValueError("SPLIT ERROR]: tracing cut graph components != 2")
        f_in, f_out = list(components[0]), list(components[1])


        self.plot_mesh_with_cutting_cylinder(
            new_mesh_vertices,
            new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            # new_mesh_faces[f_in],
            cylinder_params,
            elev=20,
            azim=-60
        )
        self.plot_mesh_with_cutting_cylinder(
            new_mesh_vertices,
            new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            # new_mesh_faces[f_in],
            cylinder_params,
            elev=40,
            azim=-60
        )
        self.plot_mesh_with_cutting_cylinder(
            new_mesh_vertices,
            new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            # new_mesh_faces[f_in],
            cylinder_params,
            elev=60,
            azim=-60
        )
        self.plot_mesh_with_cutting_cylinder(
            new_mesh_vertices,
            new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            # new_mesh_faces[f_in],
            cylinder_params,
            elev=80,
            azim=-60
        )
        self.plot_mesh_with_cutting_cylinder(
            new_mesh_vertices,
            new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            # new_mesh_faces[f_in],
            cylinder_params,
            elev=100,
            azim=-60
        )
        self.plot_mesh_with_cutting_cylinder(
            new_mesh_vertices,
            new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            # new_mesh_faces[f_in],
            cylinder_params,
            elev=120,
            azim=-60
        )
        self.plot_mesh_with_cutting_cylinder(
            new_mesh_vertices,
            new_mesh_faces[f_in],
            new_mesh_faces[f_out],
            # new_mesh_faces[f_in],
            cylinder_params,
            elev=140,
            azim=-60
        )


        # convert patch
        patch2f_in = defaultdict(list)
        for f_id in f_in:
            if f_id >= raw_f_len:
                p_id = new_f_p_id[f_id]
            else:
                raw_f_id = new_f_id2raw_f_id[f_id]
                p_id = self.patch_mesh.f2patch[raw_f_id]
            patch2f_in[p_id].append(f_id)
        patch2f_in = [patch2f_in[p_id] for p_id in patch2f_in]

        for i in range(len(patch2f_in)):
            for j in range(len(patch2f_in[i])):
                if patch2f_in[i][j] in f_in:
                    patch2f_in[i][j] = f_in.index(patch2f_in[i][j])


        patch2f_out = defaultdict(list)
        for f_id in f_out:
            if f_id >= raw_f_len:
                p_id = new_f_p_id[f_id]
            else:
                raw_f_id = new_f_id2raw_f_id[f_id]
                p_id = self.patch_mesh.f2patch[raw_f_id]
            patch2f_out[p_id].append(f_id)
        patch2f_out = [patch2f_out[p_id] for p_id in patch2f_out]

        for i in range(len(patch2f_out)):
            for j in range(len(patch2f_out[i])):
                if patch2f_out[i][j] in f_out:
                    patch2f_out[i][j] = f_out.index(patch2f_out[i][j])
        

        # final split result
        c, axis, r = cylinder_params[:3].squeeze(), cylinder_params[3:6].squeeze(), cylinder_params[6].item()
        v0, f0 = get_submesh(new_mesh_vertices, new_mesh_faces, f_in)
        cylinder_filling = CylinderHoleFilling(v0, f0, c, axis, r, max_area=1.0, visualize=True)
        v0, f0, adding_f_id = cylinder_filling.new_v, cylinder_filling.new_f, cylinder_filling.adding_f_id
        if len(adding_f_id) != 0:
            patch2f_in.append(list(adding_f_id))
        # hole_filling0 = HoleFilling(v0, f0, visualize=False)
        # new_v0, new_f0, filled_regions = hole_filling0.new_v, hole_filling0.new_f, hole_filling0.region_triangles
        # if len(filled_regions) != 0:
        #     cur_id = f0.shape[0]
        #     for region in filled_regions:
        #         patch2f_in.append(list(np.arange(cur_id, cur_id + region.shape[0])))
        #         cur_id += len(region)
        # v0, f0 = new_v0, new_f0
    

        v1, f1 = get_submesh(new_mesh_vertices, new_mesh_faces, f_out)
        cylinder_filling = CylinderHoleFilling(v1, f1, c, axis, r, max_area=1.0, visualize=True)
        v1, f1, adding_f_id = cylinder_filling.new_v, cylinder_filling.new_f, cylinder_filling.adding_f_id
        if len(adding_f_id) != 0:
            patch2f_out.append(list(adding_f_id))
        # hole_filling1 = HoleFilling(v1, f1, visualize=False)
        # new_v1, new_f1, filled_regions =  hole_filling1.new_v, hole_filling1.new_f, hole_filling1.region_triangles
        # if len(filled_regions) != 0:
        #     cur_id = f1.shape[0]
        #     for region in filled_regions:
        #         patch2f_out.append(list(np.arange(cur_id, cur_id + region.shape[0])))
        #         cur_id += len(region)
        # v1, f1 = new_v1, new_f1
        

        mp.plot(v0, f0, np.ones((f0.shape[0], 3)), shading={'wireframe': True})
        mp.plot(v1, f1, np.ones((f1.shape[0], 3)), shading={'wireframe': True})

        return True, [
            {
                'v': v0, 'f': f0, 'patch': patch2f_in
            },
            {
                'v': v1, 'f': f1, 'patch': patch2f_out
            }
        ]
        

    def cut_mesh_by_plane(self, plane_param):
        e_cut = {}
        mesh = self.patch_mesh.om_mesh
        insert_v = []
        insert_v_num = 0
        raw_v_num = self.patch_mesh.v.shape[0]
        for e_h in mesh.edges():
            v0, v1 = e_h2v_p(mesh, e_h)
            
            d0 = np.dot(plane_param[:3], v0) + plane_param[3]
            d1 = np.dot(plane_param[:3], v1) + plane_param[3]
            

            epsilon = 1e-10
            # if (d0 * d1 < 0):
            if (d0 * d1 < 0) and (abs(d0) > epsilon) and (abs(d1) > epsilon):
                x0, y0, z0 = v0
                x1, y1, z1 = v1
                a, b, c, d = plane_param
                
                denominator = a * (x1 - x0) + b * (y1 - y0) + c * (z1 - z0)
                numerator = -(a * x0 + b * y0 + c * z0 + d)
                
                
                if abs(denominator) > epsilon:
                    t = numerator / denominator
                    # intersection in line segment
                    # if -epsilon <= t <= 1 + epsilon:
                        # t_clamped = np.clip(t, 0, 1)
                        # x_new = x0 + t_clamped * (x1 - x0)
                        # y_new = y0 + t_clamped * (y1 - y0)
                        # z_new = z0 + t_clamped * (z1 - z0)
                    t = np.clip(t, 0, 1)
                    in_v = v0 + t * (v1 - v0)
                    insert_v.append(in_v)
                    e_cut[e_h.idx()] = raw_v_num + insert_v_num
                    insert_v_num += 1

                        # v_new_id = self.patch_mesh.v.shape[0]
                        # self.patch_mesh.v = np.concatenate((self.patch_mesh.v, np.array([x_new, y_new, z_new])[np.newaxis, :]))
                        # e_cut[e_h.idx()] = v_new_id
                    # else:
                        # print(e_h.idx())
                        # print("Intersection out of line segment, but points on different sides")
                else:
                    # parallel or on plane, has been covered
                    # print(e_h.idx())
                    print("Parallel or on plane, but points on different sides")

        insert_v = np.array(insert_v)
        new_mesh_v = np.concatenate((self.patch_mesh.v, insert_v), axis=0)

        # show
        # plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f_num, 3)), shading={"wireframe": True})
        # plot.add_points(
        #     insert_v,
        #     shading={
        #         "point_color": "red",
        #         "point_size": 3
        #     }
        # )
        # f_center = np.mean(self.patch_mesh.v[self.patch_mesh.f], axis=1)
        # f_normal = np.cross(self.patch_mesh.v[self.patch_mesh.f[:, 1]] - self.patch_mesh.v[self.patch_mesh.f[:, 0]], self.patch_mesh.v[self.patch_mesh.f[:, 2]] - self.patch_mesh.v[self.patch_mesh.f[:, 1]])
        # f_normal = f_normal / np.linalg.norm(f_normal, axis=1)[:, np.newaxis]
        # # draw normal direction
        # plot.add_lines(
        #     f_center,
        #     f_center + f_normal,
        #     shading={
        #         "line_color": "red",
        #         "line_width": 1
        #     }
        # )
        
        # triangulate
        # cut and remove faces
        f_cut = {}
        for e in e_cut:
            e_h = mesh.edge_handle(e)
            nf0, nf1 = e_h2nf(mesh, e_h)
            # he_h0, he_h1 = mesh.halfedge_handle(e_h, 0), mesh.halfedge_handle(e_h, 1)
            # f0, f1 = mesh.face_handle(he_h0), mesh.face_handle(he_h1)

            if nf0 not in f_cut:
                f_cut[nf0] = []
            if nf1 not in f_cut:
                f_cut[nf1] = []
            f_cut[nf0].append(e)
            f_cut[nf1].append(e)
            # if (f0.idx() != -1) & (f0.idx() not in f_cut):
            #     f_cut[f0.idx()] = []
            # if (f1.idx() != -1) & (f1.idx() not in f_cut):
            #     f_cut[f1.idx()] = []
            # if f0.idx() != -1:
            #     f_cut[f0.idx()].append(e)
            # if f1.idx() != -1:
            #     f_cut[f1.idx()].append(e)

        
        # insert faces
        # insert new_f[f] in face f
        new_f = {}
        for f in f_cut.keys():
            v_new_id = [e_cut[e] for e in f_cut[f]]
            new_f[f] = []
            ''' 
                insert one
            '''
            if len(v_new_id) == 1:
                v_new_id = v_new_id[0]

                # find opposite vertex
                e = f_cut[f][0]
                op_v_id = e_h2ov_in_face(mesh, mesh.edge_handle(e), self.patch_mesh.f[f])
                v_raw = self.patch_mesh.f[f]
                op_v_index = np.where(v_raw == op_v_id)[0][0]
                
                # insert
                new_f[f].append([v_raw[op_v_index], v_raw[(op_v_index + 1)%3], v_new_id])
                new_f[f].append([v_raw[(op_v_index - 1)%3], v_raw[op_v_index], v_new_id])
                
            elif len(v_new_id) == 2:
                # find first opposite vertex
                v_new_id0 = v_new_id[0]
                e = f_cut[f][0]
                op_v_id0 = e_h2ov_in_face(mesh, mesh.edge_handle(e), self.patch_mesh.f[f])
                v_raw = self.patch_mesh.f[f]
                op_v_index0 = np.where(v_raw == op_v_id0)[0][0]
                
                # find second opposite vertex
                v_new_id1 = v_new_id[1]
                e = f_cut[f][1]
                op_v_id1 = e_h2ov_in_face(mesh, mesh.edge_handle(e), self.patch_mesh.f[f])
                op_v_index1 = np.where(v_raw == op_v_id1)[0][0]
                
                if op_v_index1 == (op_v_index0 + 1)%3:
                    new_f[f].append([v_raw[op_v_index0], v_raw[(op_v_index0 + 1)%3], v_new_id0])
                    new_f[f].append([v_new_id0, v_raw[(op_v_index0 - 1)%3], v_new_id1])
                    new_f[f].append([v_raw[op_v_index0], v_new_id0, v_new_id1])
                else:
                    new_f[f].append([v_raw[(op_v_index0 - 1)%3], v_raw[op_v_index0], v_new_id0])
                    new_f[f].append([v_new_id0, v_raw[op_v_index0], v_new_id1])
                    new_f[f].append([v_raw[(op_v_index0 + 1)%3], v_new_id0, v_new_id1])
            new_f[f] = np.array(new_f[f])

        # cleanup duplicated faces
        f_cut_id = list(f_cut.keys())
        new_mesh_f = np.delete(self.patch_mesh.f, f_cut_id, axis=0)
        new_f_id2raw_f_id = [i for i in range(len(self.patch_mesh.f)) if i not in f_cut_id]
        raw_f_len = new_mesh_f.shape[0]
        # new_mesh_f = self.patch_mesh.f.copy()
        new_f_p_id = {}
        new_f_id = new_mesh_f.shape[0]
        for f in new_f.keys():
            if len(new_f[f]) == 0:
                continue
            for _ in range(new_f[f].shape[0]):
                new_f_p_id[new_f_id] = self.patch_mesh.f2patch[f]
                new_f_id += 1
            new_mesh_f = np.concatenate((new_mesh_f, new_f[f]))
        
        f_colors = np.ones((new_mesh_f.shape[0], 3))
        for i in range(f_colors.shape[0]):
            if i < raw_f_len:
                f_colors[i] = colors[self.patch_mesh.f2patch[new_f_id2raw_f_id[i]]]
            else:
                f_colors[i] = colors[new_f_p_id[i]]
        mp.plot(new_mesh_v, new_mesh_f, f_colors)

        # split mesh
        # if   |center - plane| > threshold, split
        # else consider (face-normal · plane-normal)
        f_in = set()
        f_out = set()
        threshold = 1e-6
        for i in range(new_mesh_f.shape[0]):
            center = np.mean(new_mesh_v[new_mesh_f[i]], axis=0)
            d = np.dot(plane_param[:3], center) + plane_param[3]
            if d > threshold:
                f_in.add(i)
            elif d < -threshold:
                f_out.add(i)
            else:
                v = new_mesh_v[new_mesh_f[i]]
                n = np.cross(v[1] - v[0], v[2] - v[1])
                if np.dot(n, plane_param[:3]) < 0:
                    f_in.add(i)
                else:
                    f_out.add(i)
        f_in, f_out = list(f_in), list(f_out)

        # convert patch
        patch2f_in = {}
        for f_id in f_in:
            if f_id < raw_f_len:
                raw_f_id = new_f_id2raw_f_id[f_id]
                p_id = self.patch_mesh.f2patch[raw_f_id]
            else:
                p_id = new_f_p_id[f_id]
            if p_id not in patch2f_in:
                patch2f_in[p_id] = []
            patch2f_in[p_id].append(f_id)
        patch2f_in = [patch2f_in[p_id] for p_id in patch2f_in]
        patch2f_out = {}
        for f_id in f_out:
            if f_id < raw_f_len:
                raw_f_id = new_f_id2raw_f_id[f_id]
                p_id = self.patch_mesh.f2patch[raw_f_id]
            else:
                p_id = new_f_p_id[f_id]
            if p_id not in patch2f_out:
                patch2f_out[p_id] = []
            patch2f_out[p_id].append(f_id)
        patch2f_out = [patch2f_out[p_id] for p_id in patch2f_out]
        
        v0, f0 = get_submesh(new_mesh_v, new_mesh_f, f_in)
        mp.plot(v0, f0, np.ones((f0.shape[0], 3)), shading={"wireframe": True})
        # substitute f in patch2f0/patch2f1 with f_in/f_out
        for i in range(len(patch2f_in)):
            for j in range(len(patch2f_in[i])):
                if patch2f_in[i][j] in f_in:
                    patch2f_in[i][j] = f_in.index(patch2f_in[i][j])
        for i in range(len(patch2f_out)):
            for j in range(len(patch2f_out[i])):
                if patch2f_out[i][j] in f_out:
                    patch2f_out[i][j] = f_out.index(patch2f_out[i][j])
        
        
        v1, f1 = get_submesh(new_mesh_v, new_mesh_f, f_out)
        mp.plot(v1, f1, np.ones((f1.shape[0], 3)), shading={"wireframe": True})
        
        
        return True, [{"v": v0, "f": f0, "patch": patch2f_in}, {"v": v1, "f": f1, "patch": patch2f_out}]


    def cut_mesh_by_cylinder(self, cylinder_param, v_id, p1, p2):
        cylinder_param = np.squeeze(np.asarray(cylinder_param))
        cylinder_center = cylinder_param[0:3]
        cylinder_normal = cylinder_param[3:6]
        cylinder_normal /= np.linalg.norm(cylinder_normal)
        cylinder_radius = cylinder_param[6]
        
        def point_in_cylinder(v):
            dist = np.linalg.norm(np.cross(v - cylinder_center, cylinder_normal), axis=1)
            return dist <= cylinder_radius + 1e-1
        
        # mesh = self.patch_mesh.om_mesh
        # v_in, v_out = [], []
        # for v_h in mesh.vertices():
        #     v_pos = mesh.point(v_h)
        #     v_in_cylinder = point_in_cylinder(v_pos.reshape(1, -1))
        #     if v_in_cylinder:
        #         v_in.append(v_h.idx())
        #     else:
        #         v_out.append(v_h.idx())
        
        # v_color = np.zeros((self.patch_mesh.v.shape[0], 3))
        # v_color[v_in] = [0., 1., 0.]
        # v_color[v_out] = [1., 0., 0.]
        # mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=v_color, shading={"wireframe": True})

        # e_cut = {}
        # for e_h in mesh.edges():
        #     he_h = mesh.halfedge_handle(e_h, 0)
        #     v0_h = mesh.to_vertex_handle(he_h)
        #     v1_h = mesh.from_vertex_handle(he_h)
        #     v0 = mesh.point(v0_h)
        #     v1 = mesh.point(v1_h)
        #     v0_in_cylinder = point_in_cylinder(v0.reshape(1, -1))
        #     v1_in_cylinder = point_in_cylinder(v1.reshape(1, -1))

        #     if v0_in_cylinder != v1_in_cylinder:


        
        
        # find patches where cylinder project to
        target_p_id = []
        for p_id in self.patch_mesh.graph.nodes:
            if p_id == p1:
                continue
            if p_id == p2:
                continue
            if self.patch_mesh.graph.nodes[p_id]["type"] == "Plane":
                patch_normal = self.patch_mesh.graph.nodes[p_id]["params"][:3]
                patch_normal /= np.linalg.norm(patch_normal)
                if abs(np.dot(cylinder_normal, patch_normal)) < 5*1e-2:
                    continue
                target_p_id.append(p_id)
            else:
                #TODO: test other shape
                continue
        
        # # show
        # f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
        # target_p_id = [0]
        # for p_id in target_p_id:
        #     f_id = self.patch_mesh.graph.nodes[p_id]['f']
        #     f_colors[f_id] = [1., 0., 0.]
        # plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={"wireframe": True})
        
        
        plane_param = self.patch_mesh.graph.nodes[target_p_id[0]]["params"]
        
        v_project = np.zeros((len(v_id), 3))
        for i in range(len(v_id)):
            v_pos = self.patch_mesh.v[v_id[i]]
            # project to plane
            d = np.dot(plane_param[:3], v_pos) + plane_param[3]
            v_project[i] = v_pos - d*plane_param[:3]
        
        # show
        # plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f.shape[0], 3)), shading={"wireframe": True})
        # plot.add_points(
        #     v_project,
        #     shading={
        #         "point_color": "red",
        #         "point_size": 3
        #     }
        # )
        
        # test point in triangle
        def is_point_in_triangle(P, A, B, C):
            epsilon = 1e-6
            vec_ab = B - A
            vec_ac = C - A
            cross_abc = np.cross(vec_ab, vec_ac)
            area_abc = abs(cross_abc) / 2
            
            if area_abc < epsilon:
                return False
            
            vec_ap = P - A
            cross_abp = np.cross(vec_ab, vec_ap)
            area_abp = abs(cross_abp) / 2
            
            vec_bc = C - B
            vec_bp = P - B
            cross_bcp = np.cross(vec_bc, vec_bp)
            area_bcp = abs(cross_bcp) / 2
            
            vec_ca = A - C
            vec_cp = P - C
            cross_cap = np.cross(vec_ca, vec_cp)
            area_cap = abs(cross_cap) / 2
            
            total_area = area_abp + area_bcp + area_cap
            
            return abs(total_area - area_abc) < epsilon
        
        v_project_2d = v_project[:, 1:]
        v_in_f = np.zeros(v_project.shape[0] - 2, dtype=int)
        for i in range(v_project.shape[0] - 2):
            cur_v = v_project_2d[i+1]
            for f_id in self.patch_mesh.graph.nodes[target_p_id[0]]["f"]:
                A, B, C = self.patch_mesh.v[self.patch_mesh.f[f_id]][:, 1:]
                if is_point_in_triangle(cur_v, A, B, C):
                    v_in_f[i] = f_id
                    break
        

        v_new = v_project[1:-1]
        f_remove = set()
        v_triangle_id = {
            0: 0,
            3745: 0,
            3744: 0,
            35: 0,
            256: 0,
            74: 0
        }
        for f in v_in_f:
            f_remove.add(f)
            v_triangle_id[f] += 1
        
        visit_v_triangle_id = {
            0: 0,
            3745: 0,
            3744: 0,
            35: 0,
            256: 0,
            74: 0
        }
        for i in range(v_project.shape[0] - 2):
            cur_v_id = self.patch_mesh.v.shape[0]
            cur_f = self.patch_mesh.f[v_in_f[i]]
            self.patch_mesh.v = np.concatenate((self.patch_mesh.v, v_new[i].reshape(1, 3)), axis=0)
            visit_v_triangle_id[v_in_f[i]] += 1
            
            if (v_in_f[i] == 35):
                if visit_v_triangle_id[v_in_f[i]] == 1:
                    self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                        [cur_v_id, cur_f[0], cur_f[1]]
                    ])), axis=0)
                else:
                    if visit_v_triangle_id[v_in_f[i]] == v_triangle_id[v_in_f[i]]:
                        self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                            [cur_v_id, cur_f[2], cur_f[0]]
                        ])), axis=0)
                    self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                        [cur_v_id-1, cur_v_id, cur_f[0]],
                        [cur_v_id, cur_v_id-1, cur_f[1]]
                    ])), axis=0)
            elif (v_in_f[i] == 0):
                self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                    [cur_v_id-1, cur_v_id, cur_f[0]],
                    [cur_v_id, cur_v_id-1, cur_f[1]]
                ])), axis=0)
                if visit_v_triangle_id[v_in_f[i]] == v_triangle_id[v_in_f[i]]:
                    self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                        [cur_v_id, cur_f[1], cur_f[2]]
                    ])), axis=0)
            elif (v_in_f[i] == 3745):
                self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                    [cur_v_id-1, cur_v_id, cur_f[1]],
                    [cur_v_id, cur_v_id-1, cur_f[2]]
                ])), axis=0)
                if visit_v_triangle_id[v_in_f[i]] == v_triangle_id[v_in_f[i]]:
                    self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                        [cur_v_id, cur_f[0], cur_f[1]]
                    ])), axis=0)
            elif (v_in_f[i] == 3744):
                self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                    [cur_v_id-1, cur_v_id, cur_f[0]],
                    [cur_v_id, cur_v_id-1, cur_f[2]]
                ])), axis=0)
                if visit_v_triangle_id[v_in_f[i]] == v_triangle_id[v_in_f[i]]:
                    self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                        [cur_v_id, cur_f[0], cur_f[1]]
                    ])), axis=0)
            elif (v_in_f[i] == 74):
                self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                    [cur_v_id-1, cur_v_id, cur_f[1]],
                    [cur_v_id, cur_v_id-1, cur_f[2]]
                ])), axis=0)
                if visit_v_triangle_id[v_in_f[i]] == v_triangle_id[v_in_f[i]]:
                    self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                        [cur_v_id, cur_f[0], cur_f[2]]
                    ])), axis=0)
            elif (v_in_f[i] == 256):
                self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                    [cur_v_id-1, cur_v_id, cur_f[1]],
                    [cur_v_id, cur_v_id-1, cur_f[2]]
                ])), axis=0)
                if visit_v_triangle_id[v_in_f[i]] == v_triangle_id[v_in_f[i]]:
                    self.patch_mesh.f = np.concatenate((self.patch_mesh.f, np.array([
                        [cur_v_id, cur_f[0], cur_f[2]],
                        [cur_v_id, cur_f[0], cur_f[1]],
                    ])), axis=0)
            

            
        f_remove_arr = np.array(list(f_remove))        
        self.patch_mesh.f = np.delete(self.patch_mesh.f, f_remove_arr, axis=0)

        # plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f.shape[0], 3)), shading={"wireframe": True})
        # for v_id in range(len(v_in_f)):
        #     _v = self.patch_mesh.v[raw_v_len + v_id]
        #     plot.add_points(
        #         _v.reshape(1, 3),
        #         c=np.array(colors[v_in_f[v_id] % 500]).reshape(1, 3),
        #         shading={
        #             "point_size": 3
        #         }
        #     )

        
        # split mesh
        f_in = []
        f_out = []
        for i in range(self.patch_mesh.f.shape[0]):
            center = np.mean(self.patch_mesh.v[self.patch_mesh.f[i]], axis=0)
            in_cylinder = point_in_cylinder(center.reshape(1, -1))
            if in_cylinder:
                f_in.append(i)
            else:
                f_out.append(i)
        
        v0, f0 = get_submesh(self.patch_mesh.v, self.patch_mesh.f, f_in)
        # mp.plot(v0, f0, np.ones((f0.shape[0], 3)), shading={"wireframe": True})
        
        v1, f1 = get_submesh(self.patch_mesh.v, self.patch_mesh.f, f_out)
        # mp.plot(v1, f1, np.ones((f1.shape[0], 3)), shading={"wireframe": True})

        # hole_filling0 = HoleFilling(v0, f0, visualize=True)
        # filled_f0 = hole_filling0.new_f
        # mp.plot(v0, filled_f0, np.ones((filled_f0.shape[0], 3)), shading={"wireframe": True})
        
        # hole_filling1 = HoleFilling(v1, f1, visualize=True)
        # filled_f1 = hole_filling1.new_f
        # mp.plot(v1, filled_f1, np.ones((filled_f1.shape[0], 3)), shading={"wireframe": True})
    
        
        return True, [{"v": [v0], "f": [f0], "patch": []}, {"v": [v1], "f": [f1], "patch": []}]

    
    def project_vert_to_patch(self, v, axis, p_id):
        p_type = self.patch_mesh.graph.nodes[p_id]['type']
        p_param = self.patch_mesh.graph.nodes[p_id]['params']
        
        if p_type == 'Plane':
            v_proj = proj_vert_to_plane(v, axis, p_param)

        else:
            raise Exception(f'[PROJECT]: not support project to {p_type}')
        
        return v_proj


    '''
        Cut mesh by embedded cylinder with closed cylinder curve
            1.distinguish v by cylinder
            2.find projection patch p
            3.project v_border to p
            4.triangulate
                4.1 triangulate: v(p, out), boundary(p, out), v_border_proj, boundary(v_border_proj)
                4.2 triangulate: v(p, in), boundary(p, in), v_border_proj, boundary(v_border_proj)
                4.3 side: v_border, v_border_proj
        INPUT
            - neighbor patch pair: dict{
                    p0:{id, type, params},
                    p1:{id, type, params},
                    border_v: [...]}
            - cylinder id: p \in {p0, p1}
    '''
    def cut_mesh_by_close_cylinder(self, neighbor_patch_pair):
        # initial
        p0_type, p1_type = neighbor_patch_pair['p0']['type'], neighbor_patch_pair['p1']['type']
        border_v = neighbor_patch_pair['border_v'][:-1]
        if (p0_type == 'Cylinder') & (p1_type == 'Plane'):
            p_n_side_param = neighbor_patch_pair['p0']['params']
            p_top_id = neighbor_patch_pair['p1']['id']
        elif (p1_type == 'Cylinder') & (p0_type == 'Plane'):
            p_n_side_param = neighbor_patch_pair['p1']['params']
            p_top_id = neighbor_patch_pair['p0']['id']
        else:
            raise Exception(f"close cylinder border with {p0_type} and {p1_type} neighbor")
        
        c, axis, r = p_n_side_param[0:3].squeeze(), p_n_side_param[3:6].squeeze(), p_n_side_param[6].item()

        # 1.distinguish v by cylinder
        v_on_patch = [vert_on_cylinder(c, axis, r, p) for p in self.patch_mesh.v]
        
        
        # 2.find destination p
        # 2.1 all edges intersect with cylinder
        threshold_parallel = 1e-2
        mesh = self.patch_mesh.om_mesh
        e_cut = {} # dict{e_id: [v_in, v_out]}
        for e_h in mesh.edges():
            v0_id, v1_id = e_h2v(mesh, e_h)            
            v0_type, v1_type = v_on_patch[v0_id], v_on_patch[v1_id]
            if v0_type * v1_type < 0:
                if (v0_type == 1) & (v1_type == -1):
                    e_cut[e_h.idx()] = [v0_id, v1_id]
                elif (v0_type == -1) & (v1_type == 1):
                    e_cut[e_h.idx()] = [v1_id, v0_id]
                
        # 2.2 find destinate patch
        v_all = set()
        for e in e_cut.values():
            v_all = v_all.union(set(e))
        v_all = list(v_all)
        v_all_position = []
        for v_id in v_all:
            v_all_position.extend(self.patch_mesh.v2patch[v_id])

        counter = Counter(v_all_position)
        most_count = counter.most_common()
        most_count_id = [item[0] for item in most_count]
        
        p_proj = most_count_id[0]

        self.show_single_patch(p_proj)


        # 3. project to patch & get all boundary and holes
        p_proj_param = self.patch_mesh.graph.nodes[p_proj]['params']
        # 3.1 project
        plane_normal = p_proj_param[:3]
        origin_3d = get_origin_from_plane(p_proj_param)
        u, v = get_uv_from_normal(plane_normal)
        all_v_proj_2d = np.array([project_point_to_2d(p, origin_3d, u, v) for p in self.patch_mesh.v])
        v_border_proj_3d = np.array([self.project_vert_to_patch(self.patch_mesh.v[v_id], axis, p_proj) for v_id in border_v])
        v_border_proj_2d = all_v_proj_2d[border_v]

        # 3.2 get boundary and holes of patch
        p_f = self.patch_mesh.f[self.patch_mesh.patch2f[p_proj]]
        all_boundarys = igl.boundary_loop_all(p_f)
        # distinguish boundary and holes: loop with largest area
        def polygon_area_2d(points):
            x, y = points[:,0], points[:,1]
            return 0.5 * np.sum(x*np.roll(y, -1) - y*np.roll(x, -1))
        areas = []
        for loop in all_boundarys:
            pts = all_v_proj_2d[loop]
            area = polygon_area_2d(pts)
            areas.append(area)
        
        outer_idx = np.argmax(np.abs(areas))
        outer = all_boundarys[outer_idx]
        holes = [loop for i, loop in enumerate(all_boundarys) if i != outer_idx]

        # distinguish holes inside or outside projection
        holes_inside = []
        holes_inside_id = []
        holes_outside = []
        holes_outside_id = []
        for hole in holes:
            sample_p = all_v_proj_2d[hole[0]]
            hole_2d = np.array(all_v_proj_2d[hole])
            if point_in_2d_polygon(sample_p, v_border_proj_2d):
                holes_inside.append(hole_2d)
                holes_inside_id.append(hole)
            else:
                holes_outside.append(hole_2d)
                holes_outside_id.append(hole)
        
        # 4. triangulate
        # 4.1 projection plane
        # triangulate [boundary, [projection, hole_outside]]
        outer_v_proj_2d, outer_triangles = triangulate_2d_plane_with_hole(
            np.array(all_v_proj_2d[outer]),
            holes_outside + [v_border_proj_2d]
            # [v_border_proj_2d] + holes_outside
            # min_angle=10
        )
        # triangulate [projection, [hole_inside]]
        inner_v_proj_2d, inner_triangles = triangulate_2d_plane_with_hole(
            v_border_proj_2d,
            holes_inside
            # min_angle=10
        )
        # output1: [outer, hole_outside, v_border_proj, other1]
        # output2: [v_border_proj, hole_inside, other2]
        # recover v
        # adding v_border_proj_3d, adding others1, adding others2
        len_v_proj = v_border_proj_3d.shape[0]
        len_outer = len(outer)
        len_holes_inside = sum([hole.shape[0] for hole in holes_inside])
        len_holes_outside = sum([hole.shape[0] for hole in holes_outside])

        len_other1 = outer_v_proj_2d.shape[0] - len_outer - len_holes_outside - len_v_proj
        len_other2 = inner_v_proj_2d.shape[0] - len_v_proj - len_holes_inside
        assert (len_other1 >= 0) & (len_other2 >= 0)
        
        if len_other1 == 0:
            v_other1 = np.empty((0, 3))
        else:
            v_other1 = np.array([reproject_point_2d_to_plane(p, origin_3d, u, v, p_proj_param) \
                    for p in outer_v_proj_2d[-len_other1:]])
        if len_other2 == 0:
            v_other2 = np.empty((0, 3))
        else:
            v_other2 = np.array([reproject_point_2d_to_plane(p, origin_3d, u, v, p_proj_param) \
                    for p in inner_v_proj_2d[-len_other2:]])
        
        cur_v_len = self.patch_mesh.v.shape[0]
        id_v_border_proj = list(range(cur_v_len, cur_v_len + len_v_proj))
        cur_v_len += len_v_proj
        id_v_other1 = list(range(cur_v_len, cur_v_len + len_other1))
        cur_v_len += len_other1
        id_v_other2 = list(range(cur_v_len, cur_v_len + len_other2))
        
        new_v = np.concatenate((
            self.patch_mesh.v,
            v_border_proj_3d,
            v_other1,
            v_other2
        ), axis=0)

        # recover f
        # map: output1 -> new_v
        map_output1 = outer.copy()
        for hole_outside_id in holes_outside_id:
            map_output1.extend(hole_outside_id)
        map_output1.extend(id_v_border_proj)
        map_output1.extend(id_v_other1)
        outer_triangles = np.array([[map_output1[i] for i in tri] for tri in outer_triangles])

        # map: output2 -> new_v
        map_output2 = id_v_border_proj.copy()
        for hole_inside_id in holes_inside_id:
            map_output2.extend(hole_inside_id)
        map_output2.extend(id_v_other2)
        inner_triangles = np.array([[map_output2[i] for i in tri] for tri in inner_triangles])


        # check direction
        example_f_id = self.patch_mesh.patch2f[p_proj][0]
        example_f_coord = self.patch_mesh.v[self.patch_mesh.f[example_f_id]]
        raw_p_proj_normal = np.cross(
            example_f_coord[1] - example_f_coord[0],
            example_f_coord[2] - example_f_coord[1]
        )
        for i in range(outer_triangles.shape[0]):
            f_coord = new_v[outer_triangles[i]]
            f_normal = np.cross(
                f_coord[1] - f_coord[0],
                f_coord[2] - f_coord[1]
            )
            if np.dot(f_normal, raw_p_proj_normal) < 0:
                # reverse
                outer_triangles[i] = outer_triangles[i][::-1]
        for i in range(inner_triangles.shape[0]):
            f_coord = new_v[inner_triangles[i]]
            f_normal = np.cross(
                f_coord[1] - f_coord[0],
                f_coord[2] - f_coord[1]
            )
            if np.dot(f_normal, raw_p_proj_normal) < 0:
                # reverse
                inner_triangles[i] = inner_triangles[i][::-1]

        # new_f = np.concatenate((
        #     self.patch_mesh.f,
        #     outer_triangles,
        #     # inner_triangles
        # ), axis=0)
        # new_f = np.delete(new_f, self.patch_mesh.patch2f[p_proj], axis=0)

        # mp.plot(new_v, new_f, np.ones((new_f.shape[0], 3)), shading={
        #     'wireframe': True
        # })

        # new_f = np.concatenate((
        #     self.patch_mesh.f,
        #     # outer_triangles,
        #     inner_triangles
        # ), axis=0)
        # new_f = np.delete(new_f, self.patch_mesh.patch2f[p_proj], axis=0)

        # mp.plot(new_v, new_f, np.ones((new_f.shape[0], 3)), shading={
        #     'wireframe': True
        # })

        
        new_f = np.delete(self.patch_mesh.f, self.patch_mesh.patch2f[p_proj], axis=0)
        # new_f_id2raw_f_id = {i: i for i in range(raw_f_len) if i not in self.patch_mesh.patch2f[p_proj]}
        new_f = np.concatenate((
            new_f,
            outer_triangles,
            inner_triangles
        ), axis=0)
        raw_f_len = new_f.shape[0]

        mp.plot(new_v, new_f, np.ones((new_f.shape[0], 3)), shading={
            'wireframe': True
        })

        # 4.2 construct new mesh
        # split by v_border and project_v_border
        cutting_border_vert = []
        for i in range(len(border_v)):
            cutting_border_vert.append([border_v[i], border_v[(i+1)%len(border_v)]])
        for i in range(len(id_v_border_proj)):
            cutting_border_vert.append([id_v_border_proj[i], id_v_border_proj[(i+1)%len(id_v_border_proj)]])

        # check
        plot = mp.plot(new_v, new_f, np.ones((new_f.shape[0], 3)), shading={'wireframe': True})
        cutting_border_vert_array = np.array(cutting_border_vert)
        plot.add_lines(
            new_v[cutting_border_vert_array[:, 0]],
            new_v[cutting_border_vert_array[:, 1]],
            shading={
                "line_color": "red",
                "line_width": 1
            }
        )

        # cutting_border_vert_ordered = np.sort(cutting_border_vert, axis=1)
        cutting_border_set = set((int(min(a, b)), int(max(a, b))) for a, b in cutting_border_vert)

        cutting_graph = nx.Graph()
        # add node
        for i in range(new_f.shape[0]):
            cutting_graph.add_node(i)
        # edge to face
        edge2face = defaultdict(list)
        for f_id in range(new_f.shape[0]):
            for i in range(3):
                v0, v1 = new_f[f_id][i], new_f[f_id][(i + 1) % 3]
                if v0 > v1:
                    v0, v1 = v1, v0
                edge2face[(v0, v1)].append(f_id)
        # add edges
        for edge, n_f in edge2face.items():
            if len(n_f) == 2:
                if (edge[0], edge[1]) in cutting_border_set:
                    continue
                cutting_graph.add_edge(n_f[0], n_f[1])
        components = list(nx.connected_components(cutting_graph))
        
        # f_colors = np.ones((new_f.shape[0], 3))
        # plot = mp.plot(new_v, new_f, f_colors, shading={'wireframe': True})
        # for comp in components:
        #     f_colors = np.ones((new_f.shape[0], 3))
        #     f_colors[list(comp), :] = [1, 0, 0]
        #     plot.update_object(colors = f_colors)
        if len(components) != 2:
            raise ValueError(f"[SPLIT ERROR]: tracing cut graph components {len(components)} != 2")
        raw_f_out, raw_f_in = list(components[0]), list(components[1])

        # inside mesh: v_border and project_v_border
        f_in_side_faces = []
        for i in range(len_v_proj):
            f_in_side_faces.append([id_v_border_proj[i], id_v_border_proj[(i+1)%len_v_proj], border_v[(i+1)%len_v_proj]])
            f_in_side_faces.append([id_v_border_proj[i], border_v[(i+1)%len_v_proj], border_v[i]])
            # f_in_side_faces.append([id_v_border_proj[(i+1)%len_v_proj], id_v_border_proj[i], border_v[(i+1)%len_v_proj]])
            # f_in_side_faces.append([border_v[(i+1)%len_v_proj], id_v_border_proj[i], border_v[i]])
        f_in_side_faces = np.array(f_in_side_faces)
        # check normal direction outof cylinder axis
        f_in_side_faces, _ = fix_face_normals_for_cylinder(new_v, f_in_side_faces, c, axis)

        # f_in = np.concatenate((new_f[raw_f_in], f_in_side_faces))
        f_in = raw_f_in
        # f_in = raw_f_in + list(range(raw_f_len, raw_f_len + f_in_side_faces.shape[0]))
        new_f = np.concatenate((new_f, f_in_side_faces), axis=0)
        raw_f_len = new_f.shape[0]

        # outside mesh: v_border and project_v_border
        f_out_side_faces = f_in_side_faces[:, [0, 2, 1]]
        # f_out = np.concatenate((new_f[raw_f_out], f_out_side_faces))
        f_out = raw_f_out + list(range(raw_f_len, raw_f_len + f_out_side_faces.shape[0]))
        new_f = np.concatenate((new_f, f_out_side_faces), axis=0)
        # raw_f_len = new_f.shape[0]

        # show_mesh(new_v, f_in, show_normal=False)
        # show_mesh(new_v, f_out, show_normal=False)


        # patch2f_in = defaultdict(list)
        # for f_id in raw_f_in:
        #     if f_id < raw_f_len:
        #         raw_f_id = new_f_id2raw_f_id[f_id]
        #         p_id = self.patch_mesh.f2patch[raw_f_id]
        #         p_max_id = max(p_max_id, p_id)
        #         patch2f_in[p_id].append(f_id)
        # patch2f_in = [patch2f_in[p_id] for p_id in patch2f_in]

        # for i in range(len(patch2f_in)):
        #     for j in range(len(patch2f_in[i])):
        #         if patch2f_in[i][j] in raw_f_in:
        #             patch2f_in[i][j] = raw_f_in.index(patch2f_in[i][j])

        # patch2f_out = defaultdict(list)
        # for f_id in raw_f_out:
        #     if f_id < raw_f_len:
        #         raw_f_id = new_f_id2raw_f_id[f_id]
        #         p_id = self.patch_mesh.f2patch[raw_f_id]
        # patch2f_out = [patch2f_out[p_id] for p_id in patch2f_out]

        # for i in range(len(patch2f_out)):
        #     for j in range(len(patch2f_out[i])):
        #         if patch2f_out[i][j] in raw_f_out:
        #             patch2f_out[i][j] = raw_f_out.index(patch2f_out[i][j])

        
        
        v1, f1 = get_submesh(new_v, new_f, f_in)
        v2, f2 = get_submesh(new_v, new_f, f_out)
        
        plot = mp.plot(new_v, new_f[f_in], np.ones((f1.shape[0], 3)), shading={'wireframe': True})
        plot.add_points(
            self.patch_mesh.v[border_v],
            shading={
                'point_size': 3,
                'point_color': 'red'
            }
        )
        plot.add_points(
            new_v[id_v_border_proj],
            shading={
                'point_size': 3,
                'point_color': 'blue'
            }
        )

        # show_mesh(v1, f1, wireframe=True, show_normal=True)
        # show_mesh(v2, f2, wireframe=True, show_normal=True)

        mp.plot(v1, f1, np.ones((f1.shape[0], 3)), shading={'wireframe': True})
        mp.plot(v2, f2, np.ones((f2.shape[0], 3)), shading={'wireframe': True})

        patch2f_in = self.pre_split(v1, f1)
        patch2f_out = self.pre_split(v2, f2)

        return True, [
            {"v": v1, "f": f1, "patch": patch2f_in}, 
            {"v": v2, "f": f2, "patch": patch2f_out}
        ]


    ''' 
        Pre Split by concavity

    '''
    def pre_split(self, v, f):
        mesh = om.TriMesh(v, f)
    
        f_center = np.mean(v[f], axis=1)
        f_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        f_normal = f_normal / np.linalg.norm(f_normal, axis=1).reshape(-1, 1)
        threshold_convexity = np.cos(87.5 * np.pi / 180)
        

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

        # show cutting edge
        plot = mp.plot(v, f, np.ones((f.shape[0], 3)), shading={'wireframe': True})
        if len(edges) > 0:
            plot.add_lines(
                v[edges[:, 0]],
                v[edges[:, 1]],
                shading={
                    "line_color": "red",
                    "line_width": 1
                }
            )
            centroid = np.mean(v[f], axis=1)
            normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
            normal = (normal.T / np.linalg.norm(normal, axis=1)).T
            plot.add_lines(
                centroid,
                centroid + normal,
                shading={
                    'line_color': 'blue',
                    'line_width': 2
                }

            )

        
        splited_f = split(mesh, edge)

        return splited_f




    '''
        INPUT:
        + nodes in patch mesh, each node has
            + v, f
            + type, params
        + edges in patch mesh, each edge has
            + borde_ type
    '''
    def cut_by_concave_curve(self):
        concave_curve_cut = ConcaveCurveCut(self.patch_mesh)
        return concave_curve_cut.aggrete_all_borders()
        
        # all concave border
        concave_edges = []
        for edge in self.patch_mesh.graph.edges:
            p1, p2 = edge
            border_type = self.patch_mesh.graph[p1][p2]["border_type"]
            if (border_type == 2) | (border_type == 3):
                continue
            concave_edges.append([p1, p2])
        if len(concave_edges) == 0:
            return False, []
        
        '''
            get all candidate neighbor patch's params
            [
                border: {
                    p0:{id, type, params},
                    p1:{id, type, params},
                    border_v: [...]
                }
            ]
        '''
        n_p_param = []
        for pair in concave_edges:
            p0, p1 = pair
            v_border = self.get_patch_border_v(p0, p1)
            p0_type = self.patch_mesh.graph.nodes[p0]["type"]
            p0_param = self.patch_mesh.graph.nodes[p0]["params"]
            p1_type = self.patch_mesh.graph.nodes[p1]["type"]
            p1_param = self.patch_mesh.graph.nodes[p1]["params"]
            for single_border in v_border:
                n_p_param.append({
                    'p0': {
                        'id': p0,
                        'type': p0_type,
                        'params': p0_param
                    },
                    'p1': {
                        'id': p1,
                        'type': p1_type,
                        'params': p1_param,
                    },
                    'border_v': single_border
                })
            # n_p_param.append({
            #     'p0': {
            #         'id': p0,
            #         'type': p0_type,
            #         'params': p0_param
            #     },
            #     'p1': {
            #         'id': p1,
            #         'type': p1_type,
            #         'params': p1_param,
            #     },
            #     'border_v': v_border
            # })
        
        # # case1: 2 borders share same patch 1 & 2
        # patches_of_border = []
        # for _border in n_p_param:
        #     patches_of_border.append([_border['p0']['id'], _border['p1']['id']].sort())
        # patches_of_border = np.array(patches_of_border)

        # same_patch_borders = None
        # find = False
        # for i in range(patches_of_border.shape[0]):
        #     if find:
        #         break
        #     for j in range(i+1, patches_of_border.shape[0]):
        #         if np.array_equal(patches_of_border[i], patches_of_border[j]):
        #             same_patch_borders = [i, j]
        #             find = True
        #             break
        # if same_patch_borders is not None:
        #     border1, border2 = n_p_param[same_patch_borders[0]], n_p_param[same_patch_borders[1]]

        #     # get direction
        #     def get_line_dir(v0, v1):
        #         dir = v1 - v0
        #         dir = dir / np.linalg.norm(dir)
        #         return dir

        #     border_dir = get_line_dir(border1['border_v'][0], border1['border_v'][-1])

        #     border2_v0, border2_v1 = border2['border_v'][0], border2['border_v'][-1]
        #     # test which larger on direction of border dir
        #     border2_dir = get_line_dir(border2_v0, border2_v1)
        #     border2_inserse = np.dot(border2_dir, border_dir) > 0

        #     # generate two cutting operations and cut


        
        # detect symmetric situation
        p_plane = defaultdict(list)
        p_cylinder = defaultdict(list)
        for i in range(len(n_p_param)):
            n_p = n_p_param[i]

            p0, p0_type, p0_param = n_p['p0']['id'], n_p['p0']['type'], n_p['p0']['params']
            p1, p1_type, p1_param = n_p['p1']['id'], n_p['p1']['type'], n_p['p1']['params']
            v_border = n_p['border_v']
            if p0_type == 'Plane':
                p_plane[p0].append(i)
            if p1_type == 'Plane':
                p_plane[p1].append(i)
            if p0_type == 'Cylinder':
                p_cylinder[p0].append(i)
            if p1_type == 'Cylinder':
                p_cylinder[p1].append(i)
        

        plane_cluster = []
        for p_id in p_plane:
            cur_plane_param = self.patch_mesh.graph.nodes[p_id]["params"]
            find_cluster = False
            for c_id in range(len(plane_cluster)):
                c_p_id = plane_cluster[c_id][0]
                c_plane_param = self.patch_mesh.graph.nodes[c_p_id]["params"]
                if is_plane_same(cur_plane_param, c_plane_param):
                    find_cluster = True                    
                    plane_cluster[c_id].append(p_id)
            if not find_cluster:
                plane_cluster.append([p_id])
        
        # globally same patches >= 3
        # get same plane
        common_patch = None
        for c in plane_cluster:
            if len(c) > 2:
                common_patch = c
                break
        if common_patch is not None:
            common_patch_border = {}
            # show
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            for p_id in common_patch:
                f_colors[self.patch_mesh.patch2f[p_id]] = [1, 0, 0]
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={"wireframe": False})

            for p_id in common_patch:
                n_p_list = p_plane[p_id]
                if len(n_p_list) == 1:
                    common_patch_border[p_id] = {
                        'v_border': n_p_param[n_p_list[0]]['border_v'],
                        'v_from': n_p_param[n_p_list[0]]['border_v'][0],
                        'v_to': n_p_param[n_p_list[0]]['border_v'][-1]
                    }
                else:
                    # connect border_v
                    border_v_pair = []
                    for n_p_id in n_p_list:
                        v_border = n_p_param[n_p_id]['border_v']
                        border_v_pair.append([v_border[0], v_border[-1]])
                    ordered_border_id = connect_v_pair_to_line(border_v_pair)

                    

                    if len(ordered_border_id) != 1:
                        raise Exception('[BORDER CONNECT]:cannot connect border of one patch to 1 line')
                    
                    ordered_border_id = ordered_border_id[0]

                    first_order = ordered_border_id[0][1]
                    ordered_border_v = [n_p_param[n_p_list[ordered_border_id[0][0]]]['border_v'][0] if first_order == 0 else n_p_param[n_p_list[ordered_border_id[0][0]]]['border_v'][-1]]
                    for i in range(len(ordered_border_id)):
                        cur_id, cur_dir = ordered_border_id[i]
                        if cur_dir == 0:
                            ordered_border_v.extend(n_p_param[n_p_list[cur_id]]['border_v'][1:])
                        else:
                            ordered_border_v.extend(n_p_param[n_p_list[cur_id]]['border_v'][::-1][1:])
                    v_from, v_to = ordered_border_v[0], ordered_border_v[-1]
                    common_patch_border[p_id] = {
                        'v_border': ordered_border_v,
                        'v_from': v_from,
                        'v_to': v_to
                    }
            # print(common_patch_border)
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f.shape[0], 3)), shading={"wireframe": False})
            for p_id in common_patch_border:
                plot.add_lines(
                    self.patch_mesh.v[common_patch_border[p_id]['v_border'][:-1]],
                    self.patch_mesh.v[common_patch_border[p_id]['v_border'][1:]],
                    shading={
                        'line_width': 2,
                        'line_color': 'blue'
                    }
                )
            
            # get order
            # project to 2d:
            plane_param = self.patch_mesh.graph.nodes[common_patch[0]]['params']
            plane_normal = plane_param[:3]
            origin_3d = get_origin_from_plane(plane_param)
            u, v = get_uv_from_normal(plane_normal)

            c_2d = []
            for p_id in common_patch:
                patch_v_id = np.unique(self.patch_mesh.f[self.patch_mesh.patch2f[p_id]].flatten())
                c = np.mean(self.patch_mesh.v[patch_v_id], axis=0)
                vec = c - origin_3d
                u_coord = np.dot(vec, u)
                v_coord = np.dot(vec, v)
                c_2d.append([u_coord, v_coord])
            c_2d = np.array(c_2d)

            # plot c_2d on plt
            figure = plt.figure()
            ax = figure.add_subplot(111)
            ax.scatter(c_2d[:, 0], c_2d[:, 1], c='red')
            plt.show()

            # fit to circle
            center, radius = fit_circle(c_2d)
            
            # clockwise order
            vectors = c_2d - center
            angles = np.arctan2(vectors[:,1], vectors[:,0])

            angles = np.mod(angles, 2 * np.pi)
            order = np.argsort(-angles)

            # change order
            common_patch = [common_patch[i] for i in order]
            c_2d = c_2d[order]

            # judge order of v_border
            for i in range(len(common_patch)):
                p_id = common_patch[i]
                v_from, v_to = common_patch_border[p_id]['v_from'], common_patch_border[p_id]['v_to']
                v_from_pos, v_to_pos = self.patch_mesh.v[v_from], self.patch_mesh.v[v_to]
                v_from_2d, v_to_2d = project_point_to_2d(v_from_pos, origin_3d, u, v), project_point_to_2d(v_to_pos, origin_3d, u, v)

                cross_from = np.cross(c_2d[i] - center, v_from_2d - c_2d[i])
                cross_to = np.cross(c_2d[i] - center, v_to_2d - c_2d[i])
                if cross_from > 0 and cross_to < 0:
                    continue
                elif cross_from < 0 and cross_to > 0:
                    common_patch_border[p_id]['v_border'] = common_patch_border[p_id]['v_border'][::-1]
                    common_patch_border[p_id]['v_from'], common_patch_border[p_id]['v_to'] = common_patch_border[p_id]['v_to'], common_patch_border[p_id]['v_from']
                else:
                    raise Exception('[BORDER ORDER]: cannot judge border order by cross product')
                
                # show
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f.shape[0], 3)), shading={"wireframe": False})
            for p_id in common_patch:
                print(p_id)
                v_from, v_to = common_patch_border[p_id]['v_from'], common_patch_border[p_id]['v_to']
                plot.add_points(
                    np.array([self.patch_mesh.v[v_from]]),
                    shading={
                        'point_size': 1,
                        'point_color': 'red'
                    }
                )
                plot.add_points(
                    np.array([self.patch_mesh.v[v_to]]),
                    shading={
                        'point_size': 1,
                        'point_color': 'blue'
                    }
                )

            # cut for rotation symmetric
            # from common_patch_border[p_id]['v_to'] to common_patch_border[next_p_id]['v_from']
            return self.cut_mesh_by_rotation_symmetric_plane_tracing(plane_param, common_patch, common_patch_border)

        cylinder_cluster = []
        for p_id in p_cylinder:
            cur_cylinder_param = self.patch_mesh.graph.nodes[p_id]["params"]
            find_cluster = False
            for c_id in range(len(cylinder_cluster)):
                c_p_id = cylinder_cluster[c_id][0]
                c_cylinder_param = self.patch_mesh.graph.nodes[c_p_id]["params"]
                if is_cylinder_same(cur_cylinder_param, c_cylinder_param):
                    find_cluster = True
                    cylinder_cluster[c_id].append(p_id)
            if not find_cluster:
                cylinder_cluster.append([p_id])
        for c in cylinder_cluster:
            if len(c) > 2:
                common_patch = c
                break
        if common_patch is not None:
            common_patch_border = {}
            # show
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            # for p_id in common_patch:
            #     f_colors[self.patch_mesh.patch2f[p_id]] = [1, 0, 0]
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={"wireframe": False})

            for p_id in common_patch:
                n_p_list = p_cylinder[p_id]
                if len(n_p_list) == 2:
                    common_patch_border[p_id] = [
                    {
                        'v_border': n_p_param[n_p_list[0]]['border_v'],
                        'v_from': n_p_param[n_p_list[0]]['border_v'][0],
                        'v_to': n_p_param[n_p_list[0]]['border_v'][-1]
                    },
                    {
                        'v_border': n_p_param[n_p_list[1]]['border_v'],
                        'v_from': n_p_param[n_p_list[1]]['border_v'][0],
                        'v_to': n_p_param[n_p_list[1]]['border_v'][-1]
                    }]
                
            cylinder_param = self.patch_mesh.graph.nodes[common_patch[0]]['params']
            cylinder_c, cylinder_axis, cylinder_r = cylinder_param[0:3].squeeze(), cylinder_param[3:6].squeeze(), cylinder_param[6].item()
            
            # concate cylinder_axis and 0 to plane param
            plane_param = np.zeros((4,))
            plane_param[:3] = cylinder_axis
            origin_3d = get_origin_from_plane(plane_param)
            u, v = get_uv_from_normal(cylinder_axis)

            c_2d = []
            for p_id in common_patch:
                patch_v_id = np.unique(self.patch_mesh.f[self.patch_mesh.patch2f[p_id]].flatten())
                c = np.mean(self.patch_mesh.v[patch_v_id], axis=0)
                vec = c - origin_3d
                u_coord = np.dot(vec, u)
                v_coord = np.dot(vec, v)
                c_2d.append([u_coord, v_coord])
            c_2d = np.array(c_2d)

            figure = plt.figure()
            ax = figure.add_subplot(111)
            ax.scatter(c_2d[:, 0], c_2d[:, 1])
            plt.show()

            center, radius = fit_circle(c_2d)

            vectors = c_2d - center
            angles = np.arctan2(vectors[:,1], vectors[:,0])
            angles = np.mod(angles, 2 * np.pi)
            order = np.argsort(-angles)

            common_patch = [common_patch[i] for i in order]

            # reverse v_border
            base_v_pos = self.patch_mesh.v[common_patch_border[common_patch[0]][0]['v_from']]
            d = - np.dot(base_v_pos, cylinder_axis)
            base_plane_param = np.zeros((4,))
            base_plane_param[:3] = cylinder_axis
            base_plane_param[3] = d

            for i in range(len(common_patch)):
                for j in range(2):
                    p_id = common_patch[i]
                    tmp_v_from = common_patch_border[p_id][j]['v_from']
                    if abs(np.dot(self.patch_mesh.v[tmp_v_from], cylinder_axis) + d) > 1e-1:
                        common_patch_border[p_id][j]['v_border'] = common_patch_border[p_id][j]['v_border'][::-1]
                        common_patch_border[p_id][j]['v_from'] = common_patch_border[p_id][j]['v_border'][0]
                        common_patch_border[p_id][j]['v_to'] = common_patch_border[p_id][j]['v_border'][-1]
            

            return self.cut_mesh_by_rotation_symmetric_cylinder_tracing(cylinder_param, common_patch, common_patch_border)
        
        
        print('start_cutting')
        # connect border to curve
        border_v_pair = defaultdict(list)
        for i in range(len(n_p_param)):
            border = n_p_param[i]
            
            border_v_pair[i] = [border['border_v'][0], border['border_v'][-1]]
        # border_set_visited = set()
        border_set_unvisited = set(range(len(n_p_param)))
        border_ordered = []
        while len(border_set_unvisited) > 0:
            cur_border_id = border_set_unvisited.pop()
            # border_set_visited.add(cur_border_id)

            cur_to_v = border_v_pair[cur_border_id][1]
            cur_border_list = [[cur_border_id, 0]]
            find_next_border = True
            while find_next_border:
                find_next_border = False
                # find next of cur_to_v
                for next_border_id in border_set_unvisited:
                    next_from_v, next_to_v = border_v_pair[next_border_id]
                    if cur_to_v == next_from_v:
                        cur_border_list.append([next_border_id, 0])
                        cur_to_v = next_to_v
                        find_next_border = True
                        border_set_unvisited.remove(next_border_id)
                        break
                    elif cur_to_v == next_to_v:
                        cur_border_list.append([next_border_id, 1])
                        cur_to_v = next_from_v
                        find_next_border = True
                        border_set_unvisited.remove(next_border_id)
                        break
                if find_next_border:
                    continue
                # reverse cur_border_list, change 0 / 1 to 1/0
                new_cur_border_list = []
                for i in range(len(cur_border_list)-1, -1, -1):
                    b_id, dir = cur_border_list[i]
                    new_cur_border_list.append((b_id, 1 - dir))
                cur_border_list = new_cur_border_list
                # find next of cur_from_v
                dir = cur_border_list[-1][1]
                cur_to_v = border_v_pair[cur_border_list[-1][0]][1 - dir]
                for next_border_id in border_set_unvisited:
                    next_from_v, next_to_v = border_v_pair[next_border_id]
                    if cur_to_v == next_from_v:
                        cur_border_list.append([next_border_id, 0])
                        cur_to_v = next_to_v
                        find_next_border = True
                        border_set_unvisited.remove(next_border_id)
                        break
                    elif cur_to_v == next_to_v:
                        cur_border_list.append([next_border_id, 1])
                        cur_to_v = next_from_v
                        find_next_border = True
                        border_set_unvisited.remove(next_border_id)
                        break
            border_ordered.append(cur_border_list)
        

        # select border
        for i in range(len(border_ordered)):
            # mp.offline()
            print(f"border {i}:")

            # save_ply_with_edges(
            #     f'border{i}.ply',
            #     self.patch_mesh.v,
            #     self.patch_mesh.f,
            #     border_ordered[i]
            # )
            
            cur_border_ordered = border_ordered[i]
            b_id_from, dir_from = cur_border_ordered[0]
            b_id_to, dir_to = cur_border_ordered[-1]
            border_v = [n_p_param[b_id_from]['border_v'][0] if dir_from == 0 else n_p_param[b_id_from]['border_v'][-1]]
            for i in range(len(cur_border_ordered)):
                b_id, dir = cur_border_ordered[i]
                if dir == 0:
                    border_v.extend(n_p_param[b_id]['border_v'][1:])
                else:
                    border_v.extend(n_p_param[b_id]['border_v'][::-1][1:])
            
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f.shape[0], 3)), shading={"wireframe": False})
            plot.add_lines(
                self.patch_mesh.v[border_v[:-1]],
                self.patch_mesh.v[border_v[1:]],
                shading={
                    'line_width': 2,
                    'line_color': 'red'
                }
            )
        border_id = int(input(f"Input border id:"))
        border_ordered = border_ordered[border_id]

        # form vertex border
        b_id_from, dir_from = border_ordered[0]
        b_id_to, dir_to = border_ordered[-1]
        border_v = [n_p_param[b_id_from]['border_v'][0] if dir_from == 0 else n_p_param[b_id_from]['border_v'][-1]]
        for i in range(len(border_ordered)):
            b_id, dir = border_ordered[i]
            if dir == 0:
                border_v.extend(n_p_param[b_id]['border_v'][1:])
            else:
                border_v.extend(n_p_param[b_id]['border_v'][::-1][1:])
        v_from, v_to = border_v[0], border_v[-1]


        # check: if close -> cut by projection
        if v_from == v_to:
            return self.cut_mesh_by_close_cylinder(n_p_param[b_id_from])


        # else: not close -> cut by surface-tracing
        # 1. select patch/param of border
        if len(border_ordered) > 1:
            border_from_p0, border_from_p0_type, border_from_p0_params = n_p_param[b_id_from]['p0']['id'], n_p_param[b_id_from]['p0']['type'], n_p_param[b_id_from]['p0']['params']
            border_from_p1, border_from_p1_type, border_from_p1_params = n_p_param[b_id_from]['p1']['id'], n_p_param[b_id_from]['p1']['type'], n_p_param[b_id_from]['p1']['params']
            border_to_p0, border_to_p0_type, border_to_p0_params = n_p_param[b_id_to]['p0']['id'], n_p_param[b_id_to]['p0']['type'], n_p_param[b_id_to]['p0']['params']
            border_to_p1, border_to_p1_type, border_to_p1_params = n_p_param[b_id_to]['p1']['id'], n_p_param[b_id_to]['p1']['type'], n_p_param[b_id_to]['p1']['params']
            
            candidate_border_p = [border_from_p0, border_from_p1, border_to_p0, border_to_p1]
            candidate_border_type = [border_from_p0_type, border_from_p1_type, border_to_p0_type, border_to_p1_type]

            for i in range(4):
                print(f'[{i+1}]: {candidate_border_type[i]} patch {candidate_border_p[i]}')

                f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
                f_colors[self.patch_mesh.patch2f[candidate_border_p[i]]] = [0., 0., 1.]
                plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={"wireframe": False})
                plot.add_lines(
                    self.patch_mesh.v[border_v[:-1]],
                    self.patch_mesh.v[border_v[1:]],
                    shading={
                        'line_width': 2,
                        'line_color': 'red'
                    }
                )
        
            select_param = input(f'Input which cut param (1: {border_from_p0}, 2: {border_from_p1}, 3: {border_to_p0}, 4: {border_to_p1}):')
            select_type = None
    
            if select_param == '1':
                cut_param = border_from_p0_params
                select_type = border_from_p0_type
            elif select_param == '2':
                cut_param = border_from_p1_params
                select_type = border_from_p1_type
            elif select_param == '3':
                cut_param = border_to_p0_params
                select_type = border_to_p0_type
            elif select_param == '4':
                cut_param = border_to_p1_params
                select_type = border_to_p1_type
            else:
                print('Invalid cut param')
                return False, []
        else:
            b_id, _ = border_ordered[0]
            border_p0, border_p0_type, border_p0_params = n_p_param[b_id]['p0']['id'], n_p_param[b_id]['p0']['type'], n_p_param[b_id]['p0']['params']
            border_p1, border_p1_type, border_p1_params = n_p_param[b_id]['p1']['id'], n_p_param[b_id]['p1']['type'], n_p_param[b_id]['p1']['params']
            
            candidate_border_p = [border_p0, border_p1]
            candidate_border_type = [border_p0_type, border_p1_type]

            for i in range(2):
                print(f'[{i+1}]: {candidate_border_type[i]} patch {candidate_border_p[i]}')

                f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
                f_colors[self.patch_mesh.patch2f[candidate_border_p[i]]] = [0., 0., 1.]
                plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={"wireframe": False})
                plot.add_lines(
                    self.patch_mesh.v[border_v[:-1]],
                    self.patch_mesh.v[border_v[1:]],
                    shading={
                        'line_width': 2,
                        'line_color': 'red'
                    }
                )

        
            select_param = input(f'Input which cut param (1: {border_p0}, 2: {border_p1}')
            select_type = None
    
            if select_param == '1':
                cut_param = border_p0_params
                select_type = border_p0_type
            elif select_param == '2':
                cut_param = border_p1_params
                select_type = border_p1_type
            else:
                print('Invalid cut param')
                return False, []


        # 2.cut through param by different types 
        if select_type == 'Plane':
            print(f"cut by plane: {cut_param}")
            time_start = time.time()
            cut_bool, cut_result = self.cut_mesh_by_plane_tracing(cut_param, v_from, v_to, border_v)
            # planar_cut = PlaneTracingCut(self.patch_mesh)
            # objs = planar_cut.cut_from_to([(v_from, v_to, border_v)], cut_param, viz=True)
            time_end = time.time()
            print(f"cut by plane time: {time_end - time_start:.4f} seconds")
            # return True, objs
            return True, cut_result
        elif select_type == 'Cylinder':
            print(f"cut by cylinder: {cut_param}")
            time_start = time.time()
            cut_bool, cut_result = self.cut_mesh_by_cylinder_tracing(cut_param, v_from, v_to, border_v)
            # cylinder_cut = CylinderTracingCut(self.patch_mesh)
            # objs = cylinder_cut.cut_from_to([(v_from, v_to, border_v)], cut_param, viz=True)
            time_end = time.time()
            print(f"cut by cylinder time: {time_end - time_start:.4f} seconds")
            return cut_bool, cut_result
            # return True, objs
        else:
            print('Invalid cut type')
            return False, []


    def _make_plane_quad(self, plane_abcd, center, diag_len, scale=1.2):
        """
        生成位于平面上的一个矩形（四边形），用于可视化。
        - plane_abcd: (a,b,c,d) 表示 ax+by+cz+d=0
        - center: 平面上的一个中心点
        - diag_len: 根据 mesh 尺度选择的边长标度
        - scale: 放大系数
        """
        a, b, c, _ = plane_abcd
        n = np.array([a, b, c], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise ValueError("平面法向量长度为0，请检查参数 (a,b,c,d)")
        n = n / n_norm
        def _orthonormal_basis_from_normal(n):
            """给定单位法向量 n，构造与其正交的二维基 (u, v)，均归一化。"""
            n = n / (np.linalg.norm(n) + 1e-12)
            # 选择与 n 不平行的参考向量
            ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(n, ref)
            u /= (np.linalg.norm(u) + 1e-12)
            v = np.cross(n, u)
            v /= (np.linalg.norm(v) + 1e-12)
            return u, v
        u, v = _orthonormal_basis_from_normal(n)
        half = 0.5 * diag_len * scale
        corners = [
            center + (+half)*u + (+half)*v,
            center + (-half)*u + (+half)*v,
            center + (-half)*u + (-half)*v,
            center + (+half)*u + (-half)*v,
        ]
        return np.array(corners)


    
    def plot_mesh_cut_with_plane(self, 
        v, f, f_in, f_out, v_cut_idx, plane_abcd,
        mesh_color='#CCCCCC', mesh_alpha=1.0,
        plane_alpha=0.18, cut_color='crimson', line_width=3.0,
        view=(45, 30), zoom=1.4
    ):
        v = np.asarray(v, float)
        f = np.asarray(f, int)
        a, b, c, d = plane_abcd
        n = np.array([a, b, c], float)
        n /= (np.linalg.norm(n) + 1e-12)

        # --- plane quad (for visualization) ---
        center = v.mean(0)
        center_on_plane = center - (np.dot(center, n) + d) * n
        ref = np.array([1,0,0]) if abs(n[0]) < 0.9 else np.array([0,1,0])
        u = np.cross(n, ref); u /= (np.linalg.norm(u) + 1e-12)
        vdir = np.cross(n, u)
        diag = np.linalg.norm(v.max(0) - v.min(0))
        s = diag * 0.6
        quad = np.array([
            center_on_plane + s*u + s*vdir,
            center_on_plane - s*u + s*vdir,
            center_on_plane - s*u - s*vdir,
            center_on_plane + s*u - s*vdir
        ])

        # --- classify faces relative to plane ---
        # sd = v @ n + d                          # signed distance per vertex
        # tri_sd = sd[f]                           # (M,3)
        # eps = 1e-9
        # all_pos = (tri_sd >  eps).all(axis=1)
        # all_neg = (tri_sd < -eps).all(axis=1)
        # cross  = ~(all_pos | all_neg)           # faces intersecting/straddling plane

        # Back (far) and Front (near) sets:
        # 为了稳定可视化：把跨越平面的三角形同时放入两侧，避免出现裂缝
        # back_faces  = f[all_neg | cross]
        # front_faces = f[all_pos | cross]
        back_faces = f[f_in]
        front_faces = f[f_out]

        # --- figure ---
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection='3d')

        # helper to add a surface (no wireframe)
        def add_surface(faces, color, alpha=1.0):
            if len(faces) == 0: return
            coll = Poly3DCollection([v[fi] for fi in faces],
                                    facecolor=color, edgecolor='none', alpha=alpha)
            ax.add_collection3d(coll)

        # --- draw order: back mesh -> plane -> front mesh ---
        # add_surface(back_faces, mesh_color, mesh_alpha)

        # plane_poly = Poly3DCollection([quad], facecolor='gold', edgecolor='none', alpha=plane_alpha)
        # 细分成两个三角有时能稍微改善透明排序
        tri1 = [quad[0], quad[1], quad[2]]; tri2 = [quad[0], quad[2], quad[3]]
        plane_poly = Poly3DCollection([tri1, tri2], facecolor='gold', edgecolor='none', alpha=plane_alpha)
        ax.add_collection3d(plane_poly)

        add_surface(front_faces, mesh_color, mesh_alpha)

        # --- cutting polyline ---
        if v_cut_idx is not None and len(v_cut_idx) >= 2:
            cut = v[np.asarray(v_cut_idx, int)]
            segs = [[cut[i], cut[i+1]] for i in range(len(cut)-1)]
            ax.add_collection3d(Line3DCollection(segs, colors=cut_color, linewidths=line_width))

        # --- framing / zoom / view ---
        allp = np.vstack([v, quad])
        mins, maxs = allp.min(0), allp.max(0)
        mid, r = (maxs + mins)/2, (maxs - mins).max()/2
        r /= zoom
        ax.set_xlim(mid[0]-r, mid[0]+r)
        ax.set_ylim(mid[1]-r, mid[1]+r)
        ax.set_zlim(mid[2]-r, mid[2]+r)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=view[1], azim=view[0])

        # clean figure
        ax.axis('off'); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_facecolor('white')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
