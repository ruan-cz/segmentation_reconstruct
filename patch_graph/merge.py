import numpy as np
import igl
import openmesh as om

from primitive_fitting.fitting import *



class FeatureMerge:
    def __init__(self,
        v, f, patch2f,
        f_center, f_normal, f_area,
        feature_type='DA'
    ):
        self.v, self.f = v, f
        self.mesh = om.Trimesh(v, f)
        self.patch2f = patch2f
        self.f_center, self.f_normal, self.f_area = f_center, f_normal, f_area

    def pca(self, vectors):
        c = np.mean(vectors, axis=1)
        v_c = vectors - c
        cov = np.cov(v_c.T)
        eigenvalue, eigenvector = np.linalg.eigh(cov)

        return eigenvalue, eigenvector


    def get_nring_f(self, f_id, num_ring=3):
        
        pass


    def get_nring_normal(self):
        mesh = self.mesh



class BorderMerge:
    def __init__(self,
        v, f, patch2f,
        feature_type='DA'
    ):
        pass


class PrimitiveMerge:
    def __init__(self,
        v, f, patch2f,
        feature_type='DA'
    ):
        pass


