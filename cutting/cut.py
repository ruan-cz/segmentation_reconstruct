import numpy as np
import networkx as nx
import igl
import openmesh as om
import meshplot as mp
from collections import defaultdict

import copy

from geometry.geometry_computation import *
from hole_filling.cylinder_hole_filling import *
# from utils.vtk_render import *
from utils.mesh_utils import *


V_FROM_V = 'V'
V_FROM_E = 'E'


class GraphCut:
    def __init__(self, patch_mesh):
        self.patch_mesh = patch_mesh
        
        
    def result(self, viz=False):
        patch_mesh = self.patch_mesh
        v, f, patch2f, graph = patch_mesh.v, patch_mesh.f, patch_mesh.patch2f, patch_mesh.graph
        raw_graph = copy.deepcopy(graph)

        for edge in graph.edges:
            p1, p2 = edge
            if graph[p1][p2]['border_type'] == 1:
                graph.remove_edge(p1, p2)
        
        subgraphs = nx.connected_components(graph)

        sub_models = []
        for sub_patch in subgraphs:
            # convert v, f
            all_f = []
            for p_id in sub_patch:
                all_f.extend(patch2f[p_id])            
            new_v, new_f = get_submesh(v, f, all_f)
            
            # convert patch
            new_patch2f = []
            all_f = np.array(all_f)
            for p_id in sub_patch:
                cur_f = patch2f[p_id]
                new_patch2f.append(np.array([np.where(all_f == f_id)[0][0] for f_id in cur_f]))
        
            sub_models.append({
                'v': new_v,
                'f': new_f,
                'patch': new_patch2f
            })

        if viz:
            # get edges cause separation
            node_to_model = defaultdict(int)
            model_index = 0
            subgraphs = nx.connected_components(graph)
            transparent_face = []
            for single_model in subgraphs:
                if model_index == 0:
                    for p_id in single_model:
                        transparent_face.extend(patch2f[p_id])
                for patch_node in single_model:
                    node_to_model[patch_node] = model_index
                model_index += 1
            
            cutting_edges = []
            for p1, p2 in raw_graph.edges():
                if node_to_model[p1] != node_to_model[p2]:
                    cutting_edges.append([p1, p2])
            edges = []
            for pair in cutting_edges:
                p1, p2 = pair
                v_border = get_patch_border_v(patch2f[p1], patch2f[p2], f)
                for single_border in v_border:
                    for i in range(len(single_border) - 1):
                        edges.append([single_border[i], single_border[i+1]])
            edges_color = np.tile([1., 0, 0], (len(edges), 1))
            faces_color = np.ones((f.shape[0], 3))
            xml_path = '../meshsegment_paper/camera/2/concave_loop.xml'
            render_transparent_mesh_vtk(
                v, f,
                'render/2/concave_loop.png', xml_path=xml_path, scale=5,
                draw_f=True,  _f_color=faces_color, 
                transparent_face=transparent_face, face_opacity=0.4,
                e = edges, draw_e=True, _e_color = edges_color, edge_radius=0.3
                # edge_radius=0.007
                # draw_p=True, _p=all_endpoints, edge_radius=0.3, _p_size=0.7
            )
            
            subgraphs = nx.connected_components(graph)
            model_index = 0
            for single_model in subgraphs:
                single_model_f = []
                for p_id in single_model:
                    single_model_f.extend(patch2f[p_id])
                part_v, part_f = get_submesh(v, f, single_model_f)
                faces_color = np.ones((part_f.shape[0], 3))
                render_mesh_vtk(
                    part_v, part_f,
                    f'render/2/seg/iter1/part{model_index}.png', xml_path=xml_path, scale=5,
                    draw_f=True,  _f_color=faces_color, 
                    # e = edges, draw_e=True, _e_color = edges_color, 
                    # edge_radius=0.007
                    # draw_p=True, _p=all_endpoints, edge_radius=0.3, _p_size=0.7
                )
                model_index += 1
            
                
        
        return len(sub_models) > 1, sub_models

            
class PlaneCut:
    def __init__(self, patch_mesh):
        self.patch_mesh = patch_mesh
        
        
    def cut_by_plane(self, plane_param):
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

        # substitute f in patch2f0/patch2f1 with f_in/f_out
        for i in range(len(patch2f_in)):
            for j in range(len(patch2f_in[i])):
                if patch2f_in[i][j] in f_in:
                    patch2f_in[i][j] = f_in.index(patch2f_in[i][j])
        for i in range(len(patch2f_out)):
            for j in range(len(patch2f_out[i])):
                if patch2f_out[i][j] in f_out:
                    patch2f_out[i][j] = f_out.index(patch2f_out[i][j])
        
        v0, f0 = get_submesh(new_mesh_v, new_mesh_f, f_in)
        mp.plot(v0, f0, np.ones((f0.shape[0], 3)), shading={"wireframe": True})
        
        
        v1, f1 = get_submesh(new_mesh_v, new_mesh_f, f_out)
        mp.plot(v1, f1, np.ones((f1.shape[0], 3)), shading={"wireframe": True})
        
        
        return [{"v": v0, "f": f0, "patch": patch2f_in}, {"v": v1, "f": f1, "patch": patch2f_out}]


class CylinderCut:
    def __init__(self, patch_mesh):
        self.patch_mesh = patch_mesh
    
    def cut_by_cylinder(self, cylinder_param):
        cylinder_param = np.squeeze(np.asarray(cylinder_param))
        cylinder_c = cylinder_param[:3]
        cylinder_n = cylinder_param[3:6]
        cylinder_r = cylinder_param[6]

        e_cut = {}
        mesh = self.patch_mesh.om_mesh
        insert_v = []
        insert_v_num = 0
        raw_v_num = self.patch_mesh.v.shape[0]
        def point_in_cylinder(p):
            p = np.asarray(p, dtype=float)
            c = np.asarray(cylinder_c, dtype=float)
            n = np.asarray(cylinder_n, dtype=float)

            nn = np.linalg.norm(n)
            n = n / nn

            v = p - c
            v_perp = v - np.dot(v, n) * n
            return np.linalg.norm(v_perp)
            # dist = np.linalg.norm(np.cross(v - cylinder_c, cylinder_n), axis=1)
            # return dist
        def segment_cylinder_intersection_one(v0, v1, cylinder_r, cylinder_c, cylinder_n, eps=1e-12):
            v0 = np.asarray(v0, dtype=float)
            v1 = np.asarray(v1, dtype=float)
            c  = np.asarray(cylinder_c, dtype=float)
            n  = np.asarray(cylinder_n, dtype=float)

            nn = np.linalg.norm(n)
            if nn < eps:
                raise ValueError("cylinder_n is near zero.")
            n = n / nn  # normalize

            d = v1 - v0              # segment direction (not unit)
            m0 = v0 - c              # from cylinder axis point to v0

            # Remove components along cylinder axis (project onto plane orthogonal to n)
            d_perp  = d  - n * np.dot(d,  n)
            m0_perp = m0 - n * np.dot(m0, n)

            A = np.dot(d_perp, d_perp)
            B = 2.0 * np.dot(d_perp, m0_perp)
            C = np.dot(m0_perp, m0_perp) - float(cylinder_r)**2

            # If A is ~0, the segment is parallel to the cylinder axis (in terms of perpendicular motion)
            if abs(A) < eps:
                # Then distance to axis is constant along the segment.
                if abs(C) < 1e-9:
                    # Segment lies on the cylinder surface (infinitely many intersections) -> violates assumption
                    raise ValueError("Segment lies on cylinder surface (infinite intersections).")
                else:
                    raise ValueError("Segment parallel to cylinder axis: no side-surface intersection.")

            disc = B*B - 4*A*C
            if disc < -1e-12:
                raise ValueError("No real intersection (discriminant < 0).")
            disc = max(disc, 0.0)
            sqrt_disc = np.sqrt(disc)

            t1 = (-B - sqrt_disc) / (2*A)
            t2 = (-B + sqrt_disc) / (2*A)

            # We assume only one intersection: choose the one inside [0,1] if possible.
            candidates = []
            for t in (t1, t2):
                if -1e-9 <= t <= 1 + 1e-9:
                    candidates.append(t)

            if len(candidates) == 1:
                t = candidates[0]
            elif len(candidates) == 2:
                # Both are in [0,1], but user claims only one intersection.
                # Choose the one closer to the segment middle (more stable), or you can pick min(t) / max(t).
                t = min(candidates, key=lambda x: abs(x - 0.5))
            else:
                # None in [0,1]; pick the closer one (if you still want a point on the infinite line)
                t = min((t1, t2), key=lambda x: min(abs(x), abs(x-1)))
                # but this violates segment intersection, so better to error:
                raise ValueError(f"Intersections are outside segment: t1={t1}, t2={t2}")

            p = v0 + t * d
            return p

        for e_h in mesh.edges():
            v0, v1 = e_h2v_p(mesh, e_h)
            
            d0 = point_in_cylinder(v0)
            d1 = point_in_cylinder(v1)

            eps = 1e-4
            if ((d0 > cylinder_r+eps) & (d1 < cylinder_r-eps)) | ((d0 < cylinder_r-eps) & (d1 > cylinder_r+eps)):
                # compute the intersection of v0, v1 and cylinder
                in_v = segment_cylinder_intersection_one(v0, v1, cylinder_r, cylinder_c, cylinder_n, eps=1e-5)
                insert_v.append(in_v)
                e_cut[e_h.idx()] = raw_v_num + insert_v_num
                insert_v_num += 1

        insert_v = np.array(insert_v)
        print(insert_v.shape)

        new_mesh_v = np.concatenate((self.patch_mesh.v, insert_v), axis=0)

        # show
        plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f_num, 3)), shading={"wireframe": True})
        plot.add_points(
            insert_v,
            shading={
                "point_color": "red",
                "point_size": 0.1
            }
        )
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
                # if op_v_id1 == -1:
                #     f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
                #     f_colors[f] = np.array([1, 0, 0])
                #     plot.update_object(colors=f_colors)
                #     a = 1
                #     continue


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
        threshold = 1e-3

        def normal_points_outward_of_cylinder(
            p, n, eps=1e-12
        ):
            p = np.asarray(p, dtype=float)
            n = np.asarray(n, dtype=float)
            c = np.asarray(cylinder_c, dtype=float)
            a = np.asarray(cylinder_n, dtype=float)

            na = np.linalg.norm(a)
            if na < eps:
                raise ValueError("cylinder_axis is zero vector")
            a = a / na

            # radial direction (axis -> p)
            v = p - c
            r = v - np.dot(v, a) * a
            nr = np.linalg.norm(r)

            if nr < eps:
                # p is on axis -> undefined radial direction
                return 0

            s = np.dot(n, r)

            if s >= 0:
                return +1   # outward
            elif s < 0:
                return -1   # inward
            # else:
            #     return 0    # tangential / ambiguous


        for i in range(new_mesh_f.shape[0]):
            center = np.mean(new_mesh_v[new_mesh_f[i]], axis=0)
            dist_center = point_in_cylinder(center)
            if dist_center < cylinder_r - threshold:
                f_in.add(i)
            elif dist_center > cylinder_r + threshold:
                f_out.add(i)
            else:
                v = new_mesh_v[new_mesh_f[i]]
                c = np.mean(v, axis=0)
                n = np.cross(v[1] - v[0], v[2] - v[1])
                direction = normal_points_outward_of_cylinder(c, n)
                if direction > 0:
                    f_in.add(i)
                else:
                    f_out.add(i)
            # d = np.dot(plane_param[:3], center) + plane_param[3]
            # if d > threshold:
            #     f_in.add(i)
            # elif d < -threshold:
            #     f_out.add(i)
            # else:
            #     v = new_mesh_v[new_mesh_f[i]]
            #     n = np.cross(v[1] - v[0], v[2] - v[1])
            #     if np.dot(n, plane_param[:3]) < 0:
            #         f_in.add(i)
            #     else:
            #         f_out.add(i)
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

        # substitute f in patch2f0/patch2f1 with f_in/f_out
        for i in range(len(patch2f_in)):
            for j in range(len(patch2f_in[i])):
                if patch2f_in[i][j] in f_in:
                    patch2f_in[i][j] = f_in.index(patch2f_in[i][j])
        for i in range(len(patch2f_out)):
            for j in range(len(patch2f_out[i])):
                if patch2f_out[i][j] in f_out:
                    patch2f_out[i][j] = f_out.index(patch2f_out[i][j])
        
        v0, f0 = get_submesh(new_mesh_v, new_mesh_f, f_in)
        mp.plot(v0, f0, np.ones((f0.shape[0], 3)), shading={"wireframe": True})
        
        
        v1, f1 = get_submesh(new_mesh_v, new_mesh_f, f_out)
        mp.plot(v1, f1, np.ones((f1.shape[0], 3)), shading={"wireframe": True})
        
        
        return [{"v": v0, "f": f0, "patch": patch2f_in}, {"v": v1, "f": f1, "patch": patch2f_out}]

class PlaneTracingCut:
    def __init__(self, patch_mesh):
        self.patch_mesh = patch_mesh
        self.raw_v_num = patch_mesh.v.shape[0]
        self.insert_v_num = -1
    

    def single_cut_from_to(self, v_from, v_to, v_border, plane_param, viz=False):
        mesh = self.patch_mesh.om_mesh


        from_v_id = v_from
        from_v_pos = mesh.point(mesh.vertex_handle(from_v_id))
        from_v_type = V_FROM_V
        
        prev_v_id = from_v_id
        prev_v_pos = from_v_pos

        insert_v = []
        cutting_border_v = []
        for i in range(len(v_border) - 1):
            cutting_border_v.append([v_border[i], v_border[i+1]])

        insert_f = defaultdict(list)
        visit_f = set()

        if viz:
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={'wireframe': True})


        while True:
            if viz:
                plot.add_lines(
                    np.array([prev_v_pos]),
                    np.array([from_v_pos]),
                    shading={'line_color': 'red', 'line_width': 1}
                )
                plot.update_object()

            if from_v_id == v_to:
                print(f"Cutting meet vert {v_to}")
                break
            
            if from_v_type == V_FROM_V:
                from_v_h = mesh.vertex_handle(from_v_id)
                
                # get neighbor f
                neighbor_f_id = []
                for f_h in mesh.vf(from_v_h):
                    if f_h.idx() in visit_f:
                        continue
                    neighbor_f_id.append(f_h.idx())
                    visit_f.add(f_h.idx())

                # get all candidate intersect info
                next_v_info = []
                for f_id in neighbor_f_id:
                    result = insection_plane_f(mesh, plane_param, mesh.face_handle(f_id), from_v_pos)
                    if (result['type'] == 'vv') or (result['type'] == 've'):
                        next_v_info.append(result)
                if len(next_v_info) == 0:
                    raise ValueError(f"[Trace Cut] No more next vertex found from {from_v_id}, cutting stop")

                # choose next vertex
                # 1. choose ve
                # 2. for vv,
                # 2.1 not in border
                # 2.2 duplicate one
                # print(next_v_info)
                next_info_id = -1
                next_v_set = defaultdict(list)
                for i in range(len(next_v_info)):
                    _info = next_v_info[i]
                    if _info['type'] == 've':
                        next_info_id = i
                    else:
                        next_v_id = _info['next_v_id']
                        if next_v_id == v_to:
                            next_info_id = i
                            break
                        elif next_v_id in v_border:
                            continue
                        elif next_v_id not in next_v_set:
                            next_info_id = i
                            next_v_set[next_v_id].append(_info)
                        else:
                            next_info_id = i
                            break
                if next_info_id == -1:
                    raise ValueError('[Trace Cut] Cannot find next v')
                next_v_info = next_v_info[next_info_id]
            
            else:
                from_e_h = mesh.edge_handle(from_v_id)

                # get neighbor f
                neighbor_f_id = -1
                for f_id in e_h2f(mesh, from_e_h):
                    if f_id not in visit_f:
                        neighbor_f_id = f_id
                        visit_f.add(f_id)
                        break
                if neighbor_f_id == -1:
                    raise ValueError(f"[Trace Cut] No more next face found from edge{from_v_id}, cutting stop")
                
                next_v_info = insection_plane_f(mesh, plane_param, mesh.face_handle(neighbor_f_id), from_v_pos)
                


            # process next info
            if next_v_info['type'] == 'vv':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['next_v_id']
                from_v_pos = mesh.point(mesh.vertex_handle(from_v_id))
                from_v_type = V_FROM_V
                f_id = next_v_info['f_id']

                cutting_border_v.append([prev_v_id, from_v_id])

            elif next_v_info['type'] == 've':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['next_e_id']
                from_v_pos = next_v_info['v_insert']
                from_v_type = V_FROM_E
                f_id = next_v_info['f_id']

                # insert v
                insert_v.append(from_v_pos)
                self.insert_v_num += 1
                insert_v_id = self.raw_v_num + self.insert_v_num
                cutting_border_v.append([prev_v_id, insert_v_id])

                # insert f
                raw_face = self.patch_mesh.f[f_id]
                pre_v_idx_in_f = np.where(raw_face == prev_v_id)[0][0]
                insert_f[f_id].append([
                    raw_face[pre_v_idx_in_f],
                    raw_face[(pre_v_idx_in_f + 1) % 3],
                    insert_v_id
                ])
                insert_f[f_id].append([
                    raw_face[(pre_v_idx_in_f - 1) % 3],
                    raw_face[pre_v_idx_in_f],
                    insert_v_id
                ])

            elif next_v_info['type'] == 'ev':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['next_v_id']
                from_v_pos = mesh.point(mesh.vertex_handle(from_v_id))
                from_v_type = V_FROM_V
                f_id = next_v_info['f_id']

                prev_insert_v_id = self.raw_v_num + self.insert_v_num
                cutting_border_v.append([prev_insert_v_id, from_v_id])

                # insert f
                raw_face = self.patch_mesh.f[f_id]
                pre_v_idx_in_f = np.where(raw_face == from_v_id)[0][0]
                insert_f[f_id].append([
                    raw_face[pre_v_idx_in_f],
                    raw_face[(pre_v_idx_in_f + 1) % 3],
                    prev_insert_v_id
                ])
                insert_f[f_id].append([
                    raw_face[(pre_v_idx_in_f - 1) % 3],
                    raw_face[pre_v_idx_in_f],
                    prev_insert_v_id
                ])

            elif next_v_info['type'] == 'ee':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['to_e_id']
                from_v_pos = next_v_info['v_insert']
                from_v_type = V_FROM_E
                f_id = next_v_info['f_id']

                # insert v
                prev_insert_v_id = self.raw_v_num + self.insert_v_num
                insert_v.append(from_v_pos)
                self.insert_v_num += 1
                insert_v_id = self.raw_v_num + self.insert_v_num
                cutting_border_v.append([prev_insert_v_id, insert_v_id])

                # insert f
                raw_face = self.patch_mesh.f[f_id]
                op_v_id0 = e_h2ov_in_face(mesh, mesh.edge_handle(prev_v_id), raw_face)
                op_v_index0 = np.where(raw_face == op_v_id0)[0][0]
                op_v_id1 = e_h2ov_in_face(mesh, mesh.edge_handle(from_v_id), raw_face)
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
                raise ValueError(f"[Trace Cut] Unknown next info type {next_v_info['type']}")
            

        insert_v = np.array(insert_v)
        for f_id in insert_f.keys():
            insert_f[f_id] = np.array(insert_f[f_id])
        

        return insert_v, insert_f, cutting_border_v
        
    
    def cut_from_to(self, cut_operation, plane_param, reference_p_id=None, viz=False):
        new_vertices = self.patch_mesh.v
        global_insert_f = {}
        global_cutting_border_v = []
        for v_from, v_to, v_border in cut_operation:
            insert_v, insert_f, cutting_border_v = self.single_cut_from_to(v_from, v_to, v_border, plane_param, viz)
            new_vertices = np.vstack((new_vertices, insert_v))
            global_insert_f |= insert_f
            global_cutting_border_v.extend(cutting_border_v)
        global_cutting_border_v = np.array(global_cutting_border_v)

        # remove old face & add new face
        remove_f_id = list(global_insert_f.keys())
        new_faces = np.delete(self.patch_mesh.f, remove_f_id, axis=0)
        raw_f_len = new_faces.shape[0]
        for raw_f_id in global_insert_f.keys():
            new_faces = np.vstack((new_faces, global_insert_f[raw_f_id]))

        # print(self.patch_mesh.f.shape[0], len(remove_f_id), new_faces.shape[0])

        # update patch
        new_f_id2raw_f_id = [i for i in range(len(self.patch_mesh.f)) if i not in remove_f_id]
        new_f_p_id = {}
        new_f_id = raw_f_len
        for raw_f_id in global_insert_f.keys():
            for _ in range(insert_f[raw_f_id].shape[0]):
                new_f_p_id[new_f_id] = self.patch_mesh.f2patch[raw_f_id]
                new_f_id += 1


        if viz:
            
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            f_colors[self.patch_mesh.patch2f[reference_p_id]] = colors[41]
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={'wireframe': False})
            # plot.add_lines(
            #     self.patch_mesh.v[select_line[:-1]],
            #     self.patch_mesh.v[select_line[1:]],
            #     shading={'line_width':2, 'line_color':'red'}
            # )
            # plot = mp.plot(new_vertices, new_faces, np.ones((new_faces.shape[0], 3)), shading={'wireframe': True})
            plot.add_lines(
                new_vertices[np.array(global_cutting_border_v)[:,0]],
                new_vertices[np.array(global_cutting_border_v)[:,1]],
                shading={'line_color': 'red', 'line_width': 1}
            )

        
        # split mesh along border
        cutting_border_v_ordered = np.sort(global_cutting_border_v, axis=1)
        print(cutting_border_v_ordered.shape)

        cutting_graph = nx.Graph()
        # add node
        for i in range(new_faces.shape[0]):
            cutting_graph.add_node(i)
        # edge to face
        edge2face = defaultdict(list)
        for f_id in range(new_faces.shape[0]):
            for i in range(3):
                v0, v1 = new_faces[f_id][i], new_faces[f_id][(i + 1) % 3] 
                if v0 > v1:
                    v0, v1 = v1, v0
                edge2face[(v0, v1)].append(f_id)
        # add edges
        for edge, n_f in edge2face.items():
            if len(n_f) == 2:
                if np.any(np.all(cutting_border_v_ordered == edge, axis=1)):
                    continue
                cutting_graph.add_edge(n_f[0], n_f[1])
        components = list(nx.connected_components(cutting_graph))
        if len(components) < 2:
            raise ValueError(f"[Trace Cut] SPLIT ERROR]")

        f_splited = [list(c) for c in components]
        
        # convert patch
        new_patch2f = []
        for obj_id in range(len(f_splited)):
            patch2f_i = defaultdict(list)
            for f_id in f_splited[obj_id]:
                if f_id >= raw_f_len:
                    p_id = new_f_p_id[f_id]
                else:
                    raw_f_id = new_f_id2raw_f_id[f_id]
                    p_id = self.patch_mesh.f2patch[raw_f_id]
                patch2f_i[p_id].append(f_id)
            patch2f_i = [patch2f_i[p_id] for p_id in patch2f_i]

            for i in range(len(patch2f_i)):
                for j in range(len(patch2f_i[i])):
                    if patch2f_i[i][j] in f_splited[obj_id]:
                        patch2f_i[i][j] = f_splited[obj_id].index(patch2f_i[i][j])
            
            new_patch2f.append(patch2f_i)
        
        # construct new patchmesh
        objs = []
        for obj_id in range(len(f_splited)):
            tmp_v, tmp_f = get_submesh(new_vertices, new_faces, f_splited[obj_id])
            objs.append({
                'v': tmp_v,
                'f': tmp_f,
                'patch': new_patch2f[obj_id],
            })
            if viz:
                f_colors = np.ones((tmp_f.shape[0], 3))
                # for p_id in range(len(new_patch2f[obj_id])):
                #     f_colors[new_patch2f[obj_id][p_id]] = colors[p_id]
                plot = mp.plot(tmp_v, tmp_f, f_colors, shading={'wireframe': False})
                plot.add_lines(
                    new_vertices[np.array(global_cutting_border_v)[:,0]],
                    new_vertices[np.array(global_cutting_border_v)[:,1]],
                    shading={'line_color': 'red', 'line_width': 1}
                )
        return objs



class CylinderTracingCut:
    def __init__(self, patch_mesh):
        self.patch_mesh = patch_mesh
        self.raw_v_num = patch_mesh.v.shape[0]
        self.insert_v_num = -1

        
    def single_cut_from_to(self, v_from, v_to, v_border, cylinder_param, viz=False):
        c, axis, r = cylinder_param[:3].squeeze(), cylinder_param[3:6].squeeze(), cylinder_param[6]
        threshold_sample_len = r * np.pi / 36
        
        mesh = self.patch_mesh.om_mesh
        
        from_v_id = v_from
        from_v_pos = mesh.point(mesh.vertex_handle(from_v_id))
        from_v_type = V_FROM_V
        
        prev_v_id = from_v_id
        prev_v_pos = from_v_pos

        insert_v = []
        # insert_v_num = -1
        # raw_v_num = self.patch_mesh.v.shape[0]
        cutting_border_v = []
        for i in range(len(v_border) - 1):
            cutting_border_v.append([v_border[i], v_border[i+1]])

        insert_f = defaultdict(list)
        del_f = []
        visit_f = set()

        if viz:
            f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, c=f_colors, shading={'wireframe': True})


        while True:
            if viz:
                plot.add_lines(
                    np.array([prev_v_pos]),
                    np.array([from_v_pos]),
                    shading={'line_color': 'red', 'line_width': 1}
                )
                plot.update_object()

            if from_v_id == v_to:
                print(f"Cutting meet vert{v_to}")
                break
            
            # get next info
            if from_v_type == V_FROM_V:
                from_v_h = mesh.vertex_handle(from_v_id)
                
                # get neighbor f
                neighbor_f_id = []
                for f_h in mesh.vf(from_v_h):
                    neighbor_f_id.append(f_h.idx())
                    visit_f.add(f_h.idx())

                # get all candidate intersect info
                next_v_info = []
                for f_id in neighbor_f_id:
                    result = insection_cylinder_f_by_vert(mesh, cylinder_param, mesh.face_handle(f_id), from_v_pos, from_v_type, from_v_id)
                    if result['type'] == 'vv' or result['type'] == 've':
                        next_v_info.append(result)
                if len(next_v_info) == 0:
                    raise ValueError(f"[Trace Cut] No more next vertex found from {from_v_id}, cutting stop")

                # choose next vertex
                # 1. choose ve
                # 2. for vv,
                # 2.1 not in border
                # 2.2 duplicate one 
                next_info_id = -1
                next_v_set = defaultdict(list)
                for i in range(len(next_v_info)):
                    _info = next_v_info[i]
                    if _info['type'] == 've':
                        next_info_id = i
                    else:
                        next_v_id = _info['next_v_id']
                        if next_v_id == v_to:
                            next_info_id = i
                            break
                        elif next_v_id in v_border:
                            continue
                        elif next_v_id not in next_v_set:
                            next_info_id = i
                            next_v_set[next_v_id].append(_info)
                        else:
                            next_info_id = i
                            break
                if next_info_id == -1:
                    raise ValueError('[Trace Cut] Cannot find next v')
                next_v_info = next_v_info[next_info_id]
                a=1
            
            else:
                from_e_h = mesh.edge_handle(from_v_id)

                # get neighbor f
                neighbor_f_id = -1
                for f_id in e_h2f(mesh, from_e_h):
                    if f_id not in visit_f:
                        neighbor_f_id = f_id
                        visit_f.add(f_id)
                        break
                if neighbor_f_id == -1:
                    raise ValueError(f"[Trace Cut] No more next face found from edge{from_v_id}, cutting stop")
                
                next_v_info = insection_cylinder_f_by_vert(mesh, cylinder_param, mesh.face_handle(neighbor_f_id), from_v_pos, from_v_type, from_v_id)
                a=1


            # process next info
            if next_v_info['type'] == 'vv':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['next_v_id']
                from_v_pos = mesh.point(mesh.vertex_handle(from_v_id))
                from_v_type = V_FROM_V
                f_id = next_v_info['f_id']

                # insert v
                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_param, f_id,
                    prev_v_pos, from_v_pos, threshold_sample_len
                )
                if sample_points is None:
                    cutting_border_v.append([prev_v_id, from_v_id])
                    # continue
                else:
                    sample_num = sample_points.shape[0]
                    insert_v_range = range(self.raw_v_num + self.insert_v_num, self.raw_v_num + self.insert_v_num + sample_num)
                    self.insert_v_num += sample_num
                    insert_v.extend(sample_points)
                    cutting_border_v.append([prev_v_id, insert_v_range[0]])
                    for i in range(sample_num - 1):
                        cutting_border_v.append([insert_v_range[i], insert_v_range[i+1]])
                    cutting_border_v.append([insert_v_range[-1], from_v_id])

                    # insert f: connect all to mid of prev_v and from_v
                    mid_v_pos = (prev_v_pos + from_v_pos) / 2.
                    insert_v.append(mid_v_pos)
                    self.insert_v_num += 1
                    mid_v_id = self.raw_v_num + self.insert_v_num
                    op_v_id = v2op_v(mesh, prev_v_id, from_v_id, f_id)
                    cur_e_id = v_pair2e(mesh, prev_v_id, from_v_id)
                    op_f_id = e2op_f(mesh, cur_e_id, f_id)
                    op_f_op_v_id = e2op_v_in_f(mesh, cur_e_id, op_f_id)

                    # 1. samples -- op_v
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
                        from_v_id
                    ])

                    # 2. samples -- mid_v
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
                        from_v_id
                    ])

                    # 3. mid_v -- op_f_op_v
                    del_f.append(op_f_id)
                    insert_f[op_f_id].append([
                        prev_v_id,
                        mid_v_id,
                        op_f_op_v_id
                    ])
                    insert_f[op_f_id].append([
                        op_f_op_v_id,
                        mid_v_id,
                        from_v_id
                    ])


            elif next_v_info['type'] == 've':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['next_e_id']
                from_v_pos = next_v_info['v_insert']
                from_v_type = V_FROM_E
                f_id = next_v_info['f_id']

                # insert v
                insert_v.append(from_v_pos)
                self.insert_v_num += 1
                insert_v_id = self.raw_v_num + self.insert_v_num
                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_param, f_id,
                    prev_v_pos, from_v_pos, threshold_sample_len
                )
                if sample_points is None:
                    cutting_border_v.append([prev_v_id, insert_v_id])
                    # continue
                else:
                    sample_num = sample_points.shape[0]
                    insert_v_range = range(self.raw_v_num + self.insert_v_num + 1, self.raw_v_num + self.insert_v_num + 1 + sample_num)
                    self.insert_v_num += sample_num
                    for i in range(sample_num):
                        insert_v.append(sample_points[i])
                    
                    cutting_border_v.append([prev_v_id, insert_v_range[0]])
                    for i in range(sample_num - 1):
                        cutting_border_v.append([insert_v_range[i], insert_v_range[i+1]])
                    cutting_border_v.append([insert_v_range[-1], insert_v_id])

                    # insert f
                    n_v0_id, n_v1_id = e2v(mesh, from_v_id)
                    
                    # 1. samples -- n_v0
                    insert_f[f_id].append([
                        prev_v_id,
                        insert_v_range[0],
                        n_v0_id
                    ])
                    for i in range(0, sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            n_v0_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        insert_v_id,
                        n_v0_id
                    ])

                    # 2. samples -- n_v1
                    insert_f[f_id].append([
                        prev_v_id,
                        insert_v_range[0],
                        n_v1_id
                    ])
                    for i in range(0, sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            n_v1_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        insert_v_id,
                        n_v1_id
                    ])
                prev_insert_v_id = insert_v_id

            elif next_v_info['type'] == 'ev':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['next_v_id']
                from_v_pos = mesh.point(mesh.vertex_handle(from_v_id))
                from_v_type = V_FROM_V
                f_id = next_v_info['f_id']

                # insert v
                # prev_insert_v_id = self.raw_v_num + self.insert_v_num
                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_param, f_id,
                    prev_v_pos, from_v_pos, threshold_sample_len
                )
                if sample_points is None:
                    cutting_border_v.append([prev_insert_v_id, from_v_id])
                    # continue
                else:
                    sample_num = sample_points.shape[0]
                    insert_v_range = range(self.raw_v_num + self.insert_v_num + 1, self.raw_v_num + self.insert_v_num + 1 + sample_num)
                    self.insert_v_num += sample_num
                    for i in range(sample_num):
                        insert_v.append(sample_points[i])
                    
                    cutting_border_v.append([prev_insert_v_id, insert_v_range[0]])
                    for i in range(sample_num - 1):
                        cutting_border_v.append([insert_v_range[i], insert_v_range[i+1]])
                    cutting_border_v.append([insert_v_range[-1], from_v_id])

                    # insert f
                    n_v0_id, n_v1_id = e2v(mesh, prev_v_id)
                    # 1. samples -- n_v0
                    insert_f[f_id].append([
                        prev_insert_v_id,
                        insert_v_range[0],
                        n_v0_id
                    ])
                    for i in range(0, sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            n_v0_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        from_v_id,
                        n_v0_id
                    ])
                    # 2. samples -- n_v1
                    insert_f[f_id].append([
                        prev_insert_v_id,
                        insert_v_range[0],
                        n_v1_id
                    ])
                    for i in range(0, sample_num - 1):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            n_v1_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[-1],
                        from_v_id,
                        n_v1_id
                    ])

                prev_insert_v_id = insert_v_id


            elif next_v_info['type'] == 'ee':
                prev_v_id  = from_v_id
                prev_v_pos = from_v_pos
                from_v_id  = next_v_info['to_e_id']
                from_v_pos = next_v_info['v_insert']
                from_v_type = V_FROM_E
                f_id = next_v_info['f_id']

                prev_e_id, cur_e_id = prev_v_id, from_v_id

                # insert v
                # prev_insert_v_id = self.raw_v_num + self.insert_v_num
                insert_v.append(from_v_pos)
                self.insert_v_num += 1
                insert_v_id = self.raw_v_num + self.insert_v_num
                sample_points = sample_from_cylinder_on_face(
                    mesh, cylinder_param, f_id,
                    prev_v_pos, from_v_pos, threshold_sample_len
                )
                # insert f
                # if none sample, split triangle into 4
                if sample_points is None:
                    cutting_border_v.append([prev_insert_v_id, insert_v_id])
                    
                    f_v_id = self.patch_mesh.f[f_id]
                    op_v0_index = np.where(f_v_id == e2ov_in_face(mesh, prev_e_id, f_v_id))[0][0]
                    op_v1_index = np.where(f_v_id == e2ov_in_face(mesh, cur_e_id, f_v_id))[0][0]
                    if op_v1_index == (op_v0_index + 1) % 3:
                        insert_f[f_id].append([
                            f_v_id[op_v0_index],
                            f_v_id[(op_v0_index + 1) % 3],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            prev_insert_v_id,
                            f_v_id[(op_v0_index - 1) % 3],
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            f_v_id[op_v0_index],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                    else:
                        insert_f[f_id].append([
                            f_v_id[(op_v0_index - 1) % 3],
                            f_v_id[op_v0_index],
                            prev_insert_v_id
                        ])
                        insert_f[f_id].append([
                            f_v_id[(op_v0_index + 1) % 3],
                            prev_insert_v_id,
                            insert_v_id
                        ])
                        insert_f[f_id].append([
                            f_v_id[op_v0_index],
                            insert_v_id,
                            prev_insert_v_id
                        ])
                    
                else:
                    sample_num = sample_points.shape[0]
                    insert_v_range = range(self.raw_v_num + self.insert_v_num + 1, self.raw_v_num + self.insert_v_num + 1 + sample_num)
                    self.insert_v_num += sample_num
                    for i in range(sample_num):
                        insert_v.append(sample_points[i])
                    
                    cutting_border_v.append([prev_insert_v_id, insert_v_range[0]])
                    for i in range(sample_num - 1):
                        cutting_border_v.append([insert_v_range[i], insert_v_range[i+1]])
                    cutting_border_v.append([insert_v_range[-1], insert_v_id])

                    # insert f
                    common_v_id = e_pair2v(mesh, prev_e_id, cur_e_id)
                    prev_ano_v_id = e2ano_v(mesh, prev_e_id, common_v_id)
                    cur_ano_v_id = e2ano_v(mesh, cur_e_id, common_v_id)

                    mid_v_pos = (mesh.point(mesh.vertex_handle(prev_ano_v_id)) + mesh.point(mesh.vertex_handle(cur_ano_v_id))) / 2.
                    insert_v.append(mid_v_pos)
                    self.insert_v_num += 1
                    mid_v_id = self.raw_v_num + self.insert_v_num

                    op_e_id = v_pair2e(mesh, prev_ano_v_id, cur_ano_v_id)
                    op_f_id = e2op_f(mesh, op_e_id, f_id)
                    op_f_op_v_id = e2op_v_in_f(mesh, op_e_id, op_f_id)

                    # 1. samples -- common_v
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

                    half_sample_id = int((sample_num - 1) / 2)
                    # 2.1 samples-part1 -- mid_v
                    insert_f[f_id].append([
                        prev_insert_v_id,
                        prev_ano_v_id,
                        insert_v_range[0]
                    ])
                    for i in range(0, half_sample_id):
                        insert_f[f_id].append([
                            insert_v_range[i],
                            insert_v_range[i+1],
                            prev_ano_v_id
                        ])
                    insert_f[f_id].append([
                        insert_v_range[half_sample_id],
                        mid_v_id,
                        prev_ano_v_id
                    ])
                    # 2.2 samples-part2 -- mid_v
                    insert_f[f_id].append([
                        insert_v_range[half_sample_id],
                        mid_v_id,
                        cur_ano_v_id
                    ])
                    for i in range(half_sample_id, sample_num - 1):
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
                    # 3. mid_v -- op_f_op_v
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
            else:
                raise ValueError(f"[Trace Cut] Unknown next info type {next_v_info['type']}")

        insert_v = np.array(insert_v)
        new_vertices = np.vstack((self.patch_mesh.v, insert_v))
        for f_id in insert_f.keys():
            insert_f[f_id] = np.array(insert_f[f_id])
            # check normal direction
            raw_face_normal = face_normal(new_vertices, self.patch_mesh.f[f_id])
            insert_face_normal = [face_normal(new_vertices, insert_f[f_id][i]) for i in range(insert_f[f_id].shape[0])]
            for i in range(insert_f[f_id].shape[0]):
                if np.dot(raw_face_normal, insert_face_normal[i]) < 0:
                    insert_f[f_id][i] = insert_f[f_id][i][[0,2,1]]

        return insert_v, insert_f, cutting_border_v, del_f

    def cut_from_to(self, cut_operation, cylinder_param,  viz=False):
        new_vertices = self.patch_mesh.v
        global_insert_f = {}
        global_cutting_border_v = []
        delete_f = []
        for v_from, v_to, v_border in cut_operation:
            insert_v, insert_f, cutting_border_v, del_f = self.single_cut_from_to(v_from, v_to, v_border, cylinder_param, viz)
            new_vertices = np.vstack((new_vertices, insert_v))
            global_insert_f |= insert_f
            global_cutting_border_v.extend(cutting_border_v)
            delete_f.extend(del_f)
        global_cutting_border_v = np.array(global_cutting_border_v)

        # remove old face & add new face
        remove_f_id = list(set(global_insert_f.keys()))
        # remove_f_id = list(set(global_insert_f.keys()) | set(delete_f))
        new_faces = np.delete(self.patch_mesh.f, remove_f_id, axis=0)
        raw_f_len = new_faces.shape[0]
        for raw_f_id in global_insert_f.keys():
            new_faces = np.vstack((new_faces, global_insert_f[raw_f_id]))

        # update patch
        new_f_id2raw_f_id = [i for i in range(len(self.patch_mesh.f)) if i not in remove_f_id]
        new_f_p_id = {}
        new_f_id = raw_f_len
        for raw_f_id in global_insert_f.keys():
            for _ in range(insert_f[raw_f_id].shape[0]):
                new_f_p_id[new_f_id] = self.patch_mesh.f2patch[raw_f_id]
                new_f_id += 1


        if viz:
            plot = mp.plot(new_vertices, new_faces, np.ones((new_faces.shape[0], 3)), shading={'wireframe': True})
            plot.add_lines(
                new_vertices[np.array(global_cutting_border_v)[:,0]],
                new_vertices[np.array(global_cutting_border_v)[:,1]],
                shading={'line_color': 'red', 'line_width': 1}
            )

        
        # split mesh along border
        cutting_border_v_ordered = np.sort(global_cutting_border_v, axis=1)

        cutting_graph = nx.Graph()
        for i in range(new_faces.shape[0]):
            cutting_graph.add_node(i)
        # add node
        for i in range(new_faces.shape[0]):
            cutting_graph.add_node(i)
        # edge to face
        edge2face = defaultdict(list)
        for f_id in range(new_faces.shape[0]):
            for i in range(3):
                v0, v1 = new_faces[f_id][i], new_faces[f_id][(i + 1) % 3]
                if v0 > v1:
                    v0, v1 = v1, v0
                edge2face[(v0, v1)].append(f_id)
        # add edges
        for edge, n_f in edge2face.items():
            if len(n_f) == 2:
                if np.any(np.all(cutting_border_v_ordered == edge, axis=1)):
                    continue
                cutting_graph.add_edge(n_f[0], n_f[1])
        components = list(nx.connected_components(cutting_graph))
        if len(components) < 2:
            raise ValueError(f"[Trace Cut] SPLIT ERROR]: tracing cut graph {components} components")

        f_splited = [list(c) for c in components]
        
        # convert patch
        new_patch2f = []
        for obj_id in range(len(f_splited)):
            patch2f_i = defaultdict(list)
            for f_id in f_splited[obj_id]:
                if f_id >= raw_f_len:
                    p_id = new_f_p_id[f_id]
                else:
                    raw_f_id = new_f_id2raw_f_id[f_id]
                    p_id = self.patch_mesh.f2patch[raw_f_id]
                patch2f_i[p_id].append(f_id)
            patch2f_i = [patch2f_i[p_id] for p_id in patch2f_i]

            for i in range(len(patch2f_i)):
                for j in range(len(patch2f_i[i])):
                    if patch2f_i[i][j] in f_splited[obj_id]:
                        patch2f_i[i][j] = f_splited[obj_id].index(patch2f_i[i][j])
            
            new_patch2f.append(patch2f_i)
        
        # construct new patchmesh
        objs = []
        c, axis, r = cylinder_param[:3].squeeze(), cylinder_param[3:6].squeeze(), cylinder_param[6]
        for obj_id in range(len(f_splited)):
            tmp_v, tmp_f = get_submesh(new_vertices, new_faces, f_splited[obj_id])

            # hole filling
            cylinder_filling = CylinderHoleFilling(tmp_v, tmp_f, c, axis, r, max_area=1.0, visualize=viz)
            tmp_v, tmp_f, adding_f_id = cylinder_filling.new_v, cylinder_filling.new_f, cylinder_filling.adding_f_id
            if len(adding_f_id) != 0:
                new_patch2f[obj_id].append(list(adding_f_id))
            
            objs.append({
                'v': tmp_v,
                'f': tmp_f,
                'patch': new_patch2f[obj_id],
            })
            if viz:
                f_colors = np.zeros((tmp_f.shape[0], 3))
                for p_id in range(len(new_patch2f[obj_id])):
                    f_colors[new_patch2f[obj_id][p_id]] = colors[p_id]
                mp.plot(tmp_v, tmp_f, f_colors, shading={'wireframe': True})
        return objs



class ConcaveCurveCut:
    def __init__(self, patch_mesh):
        self.patch_mesh = copy.deepcopy(patch_mesh)
        
    
    
    def aggrete_all_borders(self):
        # collect concave edges
        concave_edges = []
        for edge in self.patch_mesh.graph.edges():
            p1, p2 = edge
            if self.patch_mesh.graph[p1][p2]['border_type'] == 1:
                concave_edges.append([p1, p2])
        if len(concave_edges) == 0:
            print(f'No concave edges found')
            return False, []
        
        # get all candidate single border info
        border_info = []
        for edge in concave_edges:
            p0, p1 = edge
            border_v = self.patch_mesh.graph[p0][p1]['border_v']
            p0_type, p1_type = self.patch_mesh.graph.nodes[p0]['type'], self.patch_mesh.graph.nodes[p1]['type']
            p0_param, p1_param = self.patch_mesh.graph.nodes[p0]['params'], self.patch_mesh.graph.nodes[p1]['params']
            
            for single_border_v in border_v:
                border_info.append({
                    'p0': [p0, p0_type, p0_param],
                    'p1': [p1, p1_type, p1_param],
                    'border_v': single_border_v,
                })
            
        # case 3: general case
        # connect borders to lines
        border_endpoints = [[info['border_v'][0], info['border_v'][-1]] for info in border_info]
        border_id_ordered, border_endpoints_ordered = connect_edge_to_line(border_endpoints)
        # expand lines to verts
        lines_num = len(border_id_ordered)
        lines_vert = []
        for line_index in range(lines_num):
            line_vert = []
            for cur_border_index in range(len(border_id_ordered[line_index])):
                cur_border_vert = border_info[border_id_ordered[line_index][cur_border_index]]['border_v']
                # print(cur_border_vert[0], border_endpoints_ordered[line_index][cur_border_index])
                cur_border_dir = (cur_border_vert[0] == border_endpoints_ordered[line_index][cur_border_index])
                cur_border_vert = cur_border_vert if cur_border_dir else cur_border_vert[::-1]
                line_vert.extend(cur_border_vert[:-1])
                
                if cur_border_index == len(border_id_ordered[line_index]) - 1:
                    line_vert.append(cur_border_vert[-1])
            lines_vert.append(line_vert)
        
        # show and select
        for i in range(lines_num):
            print(f'Line {i}:')
            plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, np.ones((self.patch_mesh.f.shape[0], 3)), shading={'wireframe': False})
            plot.add_lines(
                self.patch_mesh.v[lines_vert[i][:-1]],
                self.patch_mesh.v[lines_vert[i][1:]],
                shading={'line_width':2, 'line_color':'red'}                    
            )
        select_line_id = int(input('select line id: '))
        select_line = lines_vert[select_line_id]
        select_line_v_from, select_line_v_to = select_line[0], select_line[-1]
        
        # # case 3.1: closed
        # if select_line_v_from == select_line_v_to:
        #     raise ValueError('closed line detected')
    

        def show_candidate_reference(patches, patches_type):
            for index in range(len(patches)):
                print(f'[{index}]: {patches_type[index]} patch {patches[index]}')
                
                f_colors = np.ones((self.patch_mesh.f.shape[0], 3))
                f_colors[self.patch_mesh.patch2f[patches[index]]] = colors[41]
                plot = mp.plot(self.patch_mesh.v, self.patch_mesh.f, f_colors, shading={'wireframe': False})
                plot.add_lines(
                    self.patch_mesh.v[select_line[:-1]],
                    self.patch_mesh.v[select_line[1:]],
                    shading={'line_width':2, 'line_color':'red'}
                )
        
        # case 3.2: only one border
        select_border_len = len(border_id_ordered[select_line_id])
        select_patch_id = -1
        if select_border_len == 1:
            border_id = border_id_ordered[select_line_id][0]
            
            border_p0, border_p0_type, border_p0_param = border_info[border_id]['p0']
            border_p1, border_p1_type, border_p1_param = border_info[border_id]['p1']
            
            patches = [border_p0, border_p1]
            patches_type = [border_p0_type, border_p1_type]
            patches_param = [border_p0_param, border_p1_param]
            
            show_candidate_reference(patches, patches_type)
            select_patch_index = int(input(f'Input reference surface: 0:{patches_type[0]} ; 1:{patches_type[1]}'))
            
            select_patch_id = border_p0 if select_patch_index == 0 else border_p1
            select_param = patches_param[select_patch_index]
            select_type = patches_type[select_patch_index]
            
         
        # case 3.3: several borders
        else:
            select_border_from, select_border_to = border_id_ordered[select_line_id][0], border_id_ordered[select_line_id][-1]
            border_from_p0, border_from_p0_type, border_from_p0_param = border_info[select_border_from]['p0']
            border_from_p1, border_from_p1_type, border_from_p1_param = border_info[select_border_from]['p1']
            border_to_p0, border_to_p0_type, border_to_p0_param = border_info[select_border_to]['p0']
            border_to_p1, border_to_p1_type, border_to_p1_param = border_info[select_border_to]['p1']
            
            patches = [border_from_p0, border_from_p1, border_to_p0, border_to_p1]
            patches_type = [border_from_p0_type, border_from_p1_type, border_to_p0_type, border_to_p1_type]
            patches_param = [border_from_p0_param, border_from_p1_param, border_to_p0_param, border_to_p1_param]
            
            
            show_candidate_reference(patches, patches_type)
            select_patch_index = int(input(f'Input reference surface: 0:{patches_type[0]} ; 1:{patches_type[1]} ; 2:{patches_type[2]} ; 3:{patches_type[3]}'))
            
            if select_patch_index == 0:
                select_patch_id = border_from_p0
            elif select_patch_index == 1:
                select_patch_id = border_from_p1
            elif select_patch_index == 2:
                select_patch_id = border_to_p0
            elif select_patch_index == 3:
                select_patch_id = border_to_p1
            else:
                raise ValueError(f'unsupported patch index: {select_patch_index}')
            select_param = patches_param[select_patch_index]
            select_type = patches_type[select_patch_index]
         
    
        # cut along reference surface
        if select_type == 'Plane':
            print(f'cut by plane: {select_param}')
            # planar_cut = PlaneTracingCut(self.patch_mesh)
            # objs = planar_cut.cut_from_to(
            #     [(select_line_v_from, select_line_v_to, select_line)], 
            #     select_param, reference_p_id=select_patch_id, viz=True
            # )
            planar_cut = PlaneCut(self.patch_mesh)
            objs = planar_cut.cut_by_plane(select_param)
            return True, objs
        
        elif select_type == 'Cylinder':
            print(f'cut by cylinder: {select_param}')
            # cylinder_cut = CylinderTracingCut(self.patch_mesh)
            # objs = cylinder_cut.cut_from_to(
            #     [(select_line_v_from, select_line_v_to, select_line)], 
            #     select_param, viz=True
            # )
            cylinder_cut = CylinderCut(self.patch_mesh)
            objs = cylinder_cut.cut_by_cylinder(select_param)
            return True, objs
            
        else:
            raise ValueError(f'unsupported reference surface type: {select_type}')
        
        
