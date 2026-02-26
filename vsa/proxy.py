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
from partition import *


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
            