import numpy as np
import openmesh as om
import os

from utils import *
from primitive_fitting import *
from vsa import *

class SharpFeatureSplit:
    def __init__(self, v, f):
        self.v, self.f = v, f
        self.mesh = om.Trimesh(v, f)

        self.f_center = np.mean(v[f], axis=1)
        self.f_normal = igl.per_face_normals(v, f, np.ones([1., 0., 0.]))
        self.f_normal = self.f_normal / np.linalg.norm(self.f_normal, axis=1).reshape(-1, 1)
        self.f_area   = igl.doublearea(v, f) / 2.

        self.threshold_convexity = np.cos(82.5 * np.pi / 180.)
        self.threshold_fitting_rate = 0.9

        self.splited_f = self.sharpedge_split()

    def sharpedge_split(self):
        convex_edge, concave_edge = get_border_edge(
            self.mesh,
            self.f_center, self.f_normal,
            self.threshold_convexity
        )

        sharp_edge = []
        for _sharp_edge in [convex_edge, concave_edge]:
            if len(_sharp_edge) > 0:
                sharp_edge.extend(_sharp_edge)
        sharp_edge = np.array(sharp_edge)

        splited_f = split(
            self.mesh,
            sharp_edge
        )

        return splited_f


class VSASplit:
    def __init__(self,
        v, f,
        seed_type='INTERSECTION', 
        proxy_num=50, 
        max_iters=100
    ):
        vsa = VSA(
            v, f,
            proxy_num=proxy_num,
            max_iters=max_iters
        )
        _, self.patch_cluster = vsa.iteration(
            seed_type
        )
        

            









