import numpy as np
import matplotlib.pyplot as plt
import igl
import meshplot as mp
import openmesh as om
import networkx as nx
import trimesh
import trimesh.visual
from collections import defaultdict

from geometry.openmesh_utils import *





class MeshIntersection:
    def __init__(self, v, f, patch2f):
        self.v, self.f = v, f
        self.patch2f = patch2f
        
    def edge_plane_intersect(v0, v1, plane_params):
        a, b, c, d = plane_params
        
        d0 = np.dot(v0, np.array([a, b, c])) + d
        d1 = np.dot(v1, np.array([a, b, c])) + d
        # no intersect
        if d0 * d1 > 0:
            return 1, None
        # intersect at one point
        elif d0 * d1 == 0:
            if (d0 == 0) and (d1 != 0):
                return 2, [0]
            elif (d0 != 0) and (d1 == 0):
                return 2, [1]
            else:
                return 2, [0, 1]
        # cut
        else:
            epsilon = 1e-9
            x0, y0, z0 = v0
            x1, y1, z1 = v1
            
            denominator = a * (x1 - x0) + b * (y1 - y0) + c * (z1 - z0)
            numerator = - (a * x0 + b * y0 + c * z0 + d)
            
            if abs(denominator) > epsilon:
                t = numerator / denominator
                if - epsilon < t < 1 + epsilon:
                    t_clamped = np.clip(t, 0, 1)
                    x_new = x0 + t_clamped * (x1 - x0)
                    y_new = y0 + t_clamped * (y1 - y0)
                    z_new = z0 + t_clamped * (z1 - z0)
                    return 3, np.array([x_new, y_new, z_new])
            else:
                print("parallel to plane, but points on different sides")
    
    '''
        + one vertex on cylinder:
            + one intersection(another in or out)
            + two intersection(another out)
        + two vertex on cylinder:
            + two intersection
                + parallel
                + not parallel
        + no vertex on cylinder:
            + no intersection
            + one intersection(one in one out)
            + two intersection(two out)
    '''
    def edge_cylinder_intersect(self, p0, p1, cylinder_params):
        cylinder_center = cylinder_params[0:3]
        cylinder_normal = cylinder_params[3:6]
        cylinder_normal /= np.linalg.norm(cylinder_normal)
        cylinder_radius = cylinder_params[6]
        epsilon = 1e-8
        
        vec_cp0 = p0 - cylinder_center
        vec_cp1 = p1 - cylinder_center
        
        vec_p0p1 = p1 - p0
        dot_cp0_cp0 = np.dot(vec_cp0, vec_cp0)
        dot_cp0_p0p1 = np.dot(vec_cp0, vec_p0p1)
        dot_p0p1_p0p1 = np.dot(vec_p0p1, vec_p0p1)
        dot_cp0_n = np.dot(vec_cp0, cylinder_normal)
        dot_p0p1_n = np.dot(vec_p0p1, cylinder_normal)
        A = dot_p0p1_p0p1 - dot_p0p1_n ** 2
        B = 2 * (dot_cp0_p0p1 - dot_cp0_n * dot_p0p1_n)
        C = dot_cp0_cp0 - dot_cp0_n ** 2 - cylinder_radius ** 2
        
        candidate_intersections = []
        if abs(A) < epsilon:
            d_proj_p0 = np.linalg.norm(np.cross(vec_cp0, cylinder_normal))
            if abs(d_proj_p0 - cylinder_radius) < epsilon:
                candidate_intersections.append({'type': 'E', 'id': 0, 'p': p0})
                candidate_intersections.append({'type': 'E', 'id': 1, 'p': p1})
        else:
            discriminant = B ** 2 - 4 * A * C
            if discriminant >= - epsilon:
                sqrt_discriminant = np.sqrt(max(discriminant, 0))
                t1 = (-B - sqrt_discriminant) / (2 * A)
                t2 = (-B + sqrt_discriminant) / (2 * A)
                for t in [t1, t2]:
                    if - epsilon <= t <= 1 + epsilon:
                        t_clamped = np.clip(t, 0, 1)
                        p_new = p0 + t_clamped * vec_p0p1
                        candidate_intersections.append({'type': 'N', 'p': p_new})
        
        intersect_exist_id = []
        intersect_exist = []
        intersect_new = []
        for point in candidate_intersections:
            if point['type'] == 'E':
                intersect_exist_id.append(point['id'])
                intersect_exist.append(point['p'])
            else:
                unique = True
                for p in intersect_new:
                    if np.linalg.norm(p - point['p']) < epsilon:
                        unique = False
                        break
                for p in intersect_exist:
                    if np.linalg.norm(p - point['p']) < epsilon:
                        unique = False
                        break
                if unique:
                    intersect_new.append(point['p'])
        
        return intersect_exist_id, intersect_new
                
        
            
        
        
        

    def cutting_from_plane(self, plane_params):        
        mesh = om.TriMesh(self.v, self.f)
        raw_num_v = self.v.shape[0]
        # get intersect edge and points
        intersect_edge = {}
        new_point = []
        for e_h in mesh.edges():
            he_h = mesh.halfedge_handle(e_h, 0)
            v0_h = mesh.from_vetrtex_handle(he_h)
            v1_h = mesh.to_vertex_handle(he_h)
            if v0_h.idx() > v1_h.idx():
                v0_h, v1_h = v1_h, v0_h
            
            p0, p1 = mesh.point(v0_h), mesh.point(v1_h)
            
            inter_type, inter_result = self.edge_plane_intersect(p0, p1, plane_params)
            
            e_idx = e_h.idx()
            if inter_type == 3:
                intersect_edge[e_idx] = raw_num_v + len(new_point)
                new_point.append(inter_result)
        
        # get intersect face: plane cut a face at most 2 times
        intersect_face = {}
        for e_idx in intersect_edge.keys():
            e_h = mesh.edge_handle(e_idx)
            he_h0, he_h1 = mesh.halfedge_handle(e_h, 0), mesh.halfedge_handle(e_h, 1)
            f0_h, f1_h = mesh.face_handle(he_h0), mesh.face_handle(he_h1)
            f0_idx, f1_idx = f0_h.idx(), f1_h.idx()
            if f0_idx not in intersect_face.keys():
                intersect_face[f0_idx] = []
            if f1_idx not in intersect_face.keys():
                intersect_face[f1_idx] = []
            intersect_face[f0_idx].append(e_idx)
            intersect_face[f1_idx].append(e_idx)
        
        # triangulate intersect face
        # 1. get all intersect points
        # 2. triangulate
        
        '''
            one point: 
                + on vertex      None
                + on edge        case 1  
            two points:
                + 2 on edge      case 2
                + 1 on edge      case 1
                + 2 on vertex    None
        '''
        insert_triangle = []
        del_triangle = []
        
        for f_idx in intersect_face.keys():
            e_idx_list = intersect_face[f_idx]
            f_h = mesh.face_handle(f_idx)
            
            new_vertex_id = set()
            if len(e_idx_list) == 1:
                e_idx = e_idx_list[0]
                v_id = intersect_edge[e_idx]
                new_vertex_id.add((v_id, e_idx))
            elif len(e_idx_list) == 2:
                e_idx0, e_idx1 = e_idx_list[0], e_idx_list[1]
                v_id0, v_id1 = intersect_edge[e_idx0], intersect_edge[e_idx1]
                new_vertex_id.add(v_id0)
                new_vertex_id.add(v_id1)
            
            
            if len(new_vertex_id) == 1:
                new_v_idx, e_idx = new_vertex_id.pop()
                
                e_h = mesh.edge_handle(e_idx)
                op_v_idx = get_opposite_vertex(mesh, e_h, f_h)
                op_v_idx_in_f = np.where(self.f[f_idx] == op_v_idx)[0][0]
                prev_v_idx = self.f[f_idx][(op_v_idx_in_f + 1) % 3]
                next_v_idx = self.f[f_idx][(op_v_idx_in_f - 1) % 3]
                
                insert_triangle.append([op_v_idx, prev_v_idx, new_v_idx])
                insert_triangle.append([op_v_idx, new_v_idx, next_v_idx])
                del_triangle.append(f_idx)    
            elif len(new_vertex_id) == 2:
                new_v_idx0, e_idx0 = new_vertex_id.pop()
                new_v_idx1, e_idx1 = new_vertex_id.pop()
                e_h0, e_h1 = mesh.edge_handle(e_idx0), mesh.edge_handle(e_idx1)
                
                btw_v_idx = get_between_vertex(mesh, e_h0, e_h1, f_h)
                btw_v_h = mesh.vertex_handle(btw_v_idx)
                prev_v_idx = get_another_vertex(mesh, e_h0, btw_v_h)
                next_v_idx = get_another_vertex(mesh, e_h1, btw_v_h)
                
                is_clockwise = (prev_v_idx - btw_v_idx) % 3 == 1
                if is_clockwise:
                    insert_triangle.append([0, new_v_idx0, new_v_idx1])
                    insert_triangle.append([1, new_v_idx1, new_v_idx0])
                    insert_triangle.append([new_v_idx1, 1, 2])
                else:
                    insert_triangle.append([0, new_v_idx1, new_v_idx0])
                    insert_triangle.append([1, new_v_idx0, new_v_idx1])
                    insert_triangle.append([new_v_idx0, 1, 2])
                del_triangle.append(f_idx)    
        return np.array(insert_triangle), np.array(del_triangle)
            
                    
    def cutting_from_cylinder(self, cylinder_params):
        cylinder_center = cylinder_params[0:3]
        cylinder_normal = cylinder_params[3:6]
        cylinder_normal /= np.linalg.norm(cylinder_normal)
        cylinder_radius = cylinder_params[6]
        
        mesh = om.TriMesh(self.v, self.f)
        raw_num_v = self.v.shape[0]
        
        # get intersect edge and points
        intersect_edge = {}
        intersect_point = []
        
        for e_h in mesh.edges():
            he_h = mesh.halfedge_handle(e_h, 0)
            v0_h = mesh.from_vetrtex_handle(he_h)
            v1_h = mesh.to_vertex_handle(he_h)
            if v0_h.idx() > v1_h.idx():
                v0_h, v1_h = v1_h, v0_h

            p0, p1 = mesh.point(v0_h), mesh.point(v1_h)
            
            intersect_exist_id, intersect_new = self.edge_cylinder_intersect(p0, p1, cylinder_params)
            intersect_exist_id = [v0_h.idx() for i in intersect_exist_id if i == 0]
            intersect_exist_id = [v1_h.idx() for i in intersect_exist_id if i == 1]
            
            
            e_idx = e_h.idx()
            
            intersect_edge[e_idx] = {
                'exist': intersect_exist_id,
                'new': []
            }
            for p in intersect_new:
                intersect_edge[e_idx]['new'].append(raw_num_v + len(intersect_point))
                intersect_point.append(p)
            
            
                
        # get intersect face
        intersect_face = {}
        for e_idx in intersect_edge.keys():
            e_h = mesh.edge_handle(e_idx)
            he_h0, he_h1 = mesh.halfedge_handle(e_h, 0), mesh.halfedge_handle(e_h, 1)
            f0_h, f1_h = mesh.face_handle(he_h0), mesh.face_handle(he_h1)
            f0_idx, f1_idx = f0_h.idx(), f1_h.idx()
            if f0_idx not in intersect_face.keys():
                intersect_face[f0_idx] = []
            if f1_idx not in intersect_face.keys():
                intersect_face[f1_idx] = []
            intersect_face[f0_idx].append(e_idx)
            intersect_face[f1_idx].append(e_idx)
        
        
        # triangluate intersect face
        '''
            one point:
                + on vertex      None
                + on edge        impossible
            two points:
                + 2 on vertex    case 1
                + vertex & edge  case 2
                + 2 on edge      case 3
            three points:        Not considered
        '''
        
        insert_triangle = []
        del_triangle = []
        
        for f_idx in intersect_face.keys():
            e_idx_list = intersect_face[f_idx]
            f_h = mesh.face_handle(f_idx)
            
            
        
        
                
                    
        
        