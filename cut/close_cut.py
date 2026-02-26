import numpy as np
import networkx as nx
import igl
import openmesh as om
import meshplot as mp
from collections import defaultdict

import copy

from geometry_utils.geometry_computation import *
from CylinderHoleFilling import *
from render.vtk_render import *
from utils import *


class CylinderCloseCut:
    def __init__(self, patch_mesh):
        self.patch_mesh = patch_mesh


    def close_cut(self, border_v, side_cylinder_param):
        cyl_c, cyl_axis, cyl_r = side_cylinder_param[:3].squeeze(), side_cylinder_param[3:6].squeeze(), side_cylinder_param[3:6].squeeze(), side_cylinder_param[6].item()

        # 1. position of border vertice to cylinder

        # 2. find destinate projection patch

        # 3. project to patch
        
        # 4. get all boundary and holes

        # 5. triangluate

        # 6. post process

        # 7. translalation


