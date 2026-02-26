import numpy as np
import igl
from queue import PriorityQueue
import meshplot as mp
# import timeit
import openmesh as om
import trimesh
import os
import json

from utils.utils import *
from primitive_fitting import *
from distance import *

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