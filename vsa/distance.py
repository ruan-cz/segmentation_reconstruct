import numpy as np
import igl
from queue import PriorityQueue
import meshplot as mp
# import timeit
import openmesh as om
import trimesh
import os
import json

from utils import *
from primitive_fitting import *

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
    
    