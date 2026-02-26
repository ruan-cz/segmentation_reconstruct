import numpy as np
import matplotlib.pyplot as plt
import igl
import meshplot as mp
import openmesh as om
import networkx as nx
import trimesh
import trimesh.visual
from collections import defaultdict


def get_opposite_vertex(mesh, e_h, f_h):
    he_h0, he_h1 = mesh.halfedge_handle(e_h, 0), mesh.halfedge_handle(e_h, 1)
    opposite_v_h = [mesh.opposite_vh(he_h0), mesh.opposite_vh(he_h1)]
    
    for op_v_h in opposite_v_h:
        for v_h in mesh.fv_range(f_h):
            if op_v_h.idx() == v_h.idx():
                return op_v_h.idx()
    print("Error: No opposite vertex found.")
    return None

def get_between_vertex(mesh, e_h0, e_h1, f_h):
    op_v_idx0 = get_opposite_vertex(mesh, e_h0, f_h)
    op_v_idx1 = get_opposite_vertex(mesh, e_h1, f_h)
    
    for v_h in mesh.fv_range(f_h):
        if (v_h.idx() != op_v_idx0) and (v_h.idx()!= op_v_idx1):
            return v_h.idx()
    print("Error: No between vertex found.")
    return None

def get_another_vertex(mesh, e_h, v_h):
    he_h = mesh.halfedge_handle(e_h, 0)
    v0_h, v1_h = mesh.from_vertex_handle(he_h), mesh.to_vertex_handle(he_h)
    return v0_h.idx() if v0_h.idx() != v_h.idx() else v1_h.idx()    
    