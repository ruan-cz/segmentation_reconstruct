import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import igl
# from queue import PriorityQueue
# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# from distinctipy import distinctipy
# import meshplot as mp
# import timeit
# import openmesh as om
import networkx as nx
import trimesh
from collections import defaultdict
from geometry_utils.geometry_process import *



# read colors from file
file = open('global/colors_500.txt', 'r')
lines = file.readlines()
colors = np.zeros((len(lines),3))
for i in range(len(lines)):
    colors[i] = np.array(lines[i].split())

def patch_in_same_region(region2patch, p0, p1):
    for r in region2patch:
        if p0 in r:
           if p1 in r:
               return True
           return False

def show_each_patch(obj_file, patch_cluster_file):
    v, f = igl.read_triangle_mesh(obj_file)
    patch2f = read_patch_cluster(patch_cluster_file)

    for p_id in range(len(patch2f)):
        f_id = patch2f[p_id]
        f_colors = np.ones((f.shape[0], 3))
        f_colors[f_id] = [1.0, 0.0, 0.0]
        print(p_id)
        plot = mp.plot(v, f, f_colors, shading={
            'wireframe': False
        })
    
    # extract_p_id = input("input patch id to extract: ")
    # if extract_p_id is None:
    #     return
    # extract_p_id = int(extract_p_id)
    # output_name = input("input output name: ")
    # igl.write_triangle_mesh(output_name, v, f[patch2f[extract_p_id]])


'''
    Connect edges by shared vertex; seperate when v with degree>2
    - INPUT: N * 2 list
    - OUPUT:
        - edge lines: [[]], each line contains ordered e_id
        - v lines: [[]], each line contains ordered v_id
        - They are correspond:
            ORDER(edge_lines[i][j]) = (v_lines[i][j], v_lines[i][j+1])
'''
def connect_edge_to_line(edges):
    # adjacency: v -> list of (neighbor, edge_index)
    adj = defaultdict(list)
    for ei, (a, b) in enumerate(edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))

    deg = {v: len(adj[v]) for v in adj}
    terminals = [v for v, d in deg.items() if d != 2]
    visited_edge = [False] * len(edges)

    def walk_from(start_v, start_e):
        edge_seq = []
        vert_seq = [start_v]
        cur_v = start_v
        cur_e = start_e

        while True:
            visited_edge[cur_e] = True

            # traverse edge ei from cur_v to next_v
            _v0, _v1 = edges[cur_e]
            next_v = _v0 if _v1 == cur_v else _v1
            edge_seq.append(cur_e)
            vert_seq.append(next_v)

            if next_v in terminals:
                break

            # find next unvisited edge at next_v
            candidates = [ne for (_, ne) in adj[next_v] if not visited_edge[ne]]
            if not candidates:
                break

            cur_v = next_v
            cur_e = candidates[0]

        return edge_seq, vert_seq

    edge_lines = []
    vertex_lines = []

    # 1) Walk out from every terminal along each unvisited incident edge
    for t in terminals:
        for (_, ei) in adj[t]:
            if visited_edge[ei]:
                continue
            e_seq, v_seq = walk_from(t, ei)
            edge_lines.append(e_seq)
            vertex_lines.append(v_seq)

    # 2) Any unvisited edges belong to cycles (all deg==2)
    for e_id in range(len(edges)):
        if visited_edge[e_id]:
            continue

        a, b = edges[e_id]
        cycle_edges = []
        cycle_verts = [a, b]
        visited_edge[e_id] = True
        cycle_edges.append(e_id)
        prev_e = e_id
        cur_v = b
        while True:
            # pick other edge at cur_v that's not prev_e
            next_pairs = [(nbr, eii) for (nbr, eii) in adj[cur_v] if eii != prev_e]
            if not next_pairs:
                break
            next_nbr, next_ei = next_pairs[0]
            if next_ei == e_id:
                break
            cycle_edges.append(next_ei)
            cycle_verts.append(next_nbr)
            visited_edge[next_ei] = True
            prev_e = next_ei
            cur_v = next_nbr
        edge_lines.append(cycle_edges)
        vertex_lines.append(cycle_verts)

    return edge_lines, vertex_lines


def detect_circle_in_edge(edges):
    parent = {}
    def find_parent(x):
        if parent[x] != x:
            return find_parent(parent[x])
        return parent[x]

    def union(u, v):
        pu, pv = find_parent(u), find_parent(v)
        if pu == pv:
            return False
        parent[pu] = pv
        return True

    
    # init parent
    nodes = {v for e in edges for v in e}
    for node in nodes:
        parent[node] = node
    
    # test for edges
    for u, v in edges:
        if not union(u, v):
            return True
    return False


'''
    INPUT:
    - n * 2 edges, each row is an edge from v0 to v1
    OUTPUT:
    - list of connected line
    - each line is list of [edge_id, 0/1]
    - 0 for v0->v1, 1 for v1->v0
'''
def connect_v_pair_to_line(edges):
    border_set_unvisited = set(range(len(edges)))
    border_ordered = []
    while len(border_set_unvisited) > 0:
        cur_border_id = border_set_unvisited.pop()

        cur_to_v = edges[cur_border_id][1]
        cur_border_list = [[cur_border_id, 0]]
        find_next_border = True
        while find_next_border:
            find_next_border = False
            # find next of cur_to_v
            for next_border_id in border_set_unvisited:
                next_from_v, next_to_v = edges[next_border_id]
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
            cur_to_v = edges[cur_border_list[-1][0]][1 - dir]
            for next_border_id in border_set_unvisited:
                next_from_v, next_to_v = edges[next_border_id]
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
    return border_ordered



# INPUT
#   - raw_mesh: v, f of the mesh
#   - raw_f_id: id of faces need to be extract
# OUTPUT
#   - new openmesh mesh
def get_submesh(raw_mesh_v, raw_mesh_f, raw_f_id):
    original_f = raw_mesh_f[raw_f_id]
    original_v_id = np.unique(original_f.flatten())
    v_id_map = {}
    for i in range(len(original_v_id)):
        v_id_map[original_v_id[i]] = i
    
    cur_v = raw_mesh_v[original_v_id]
    cur_f = np.zeros_like(original_f)
    for i in range(cur_f.shape[0]):
        for j in range(cur_f.shape[1]):
            cur_f[i][j] = v_id_map[original_f[i][j]]        

    return cur_v, cur_f


def project_point_to_2d(point, origin, u, v):
    vec = point - origin
    x = np.dot(vec, u)
    y = np.dot(vec, v)
    return np.array([x, y])

def reproject_point_2d_to_plane(p, o, u, v, plane):
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=float)

    X0 = o + p[0] * u + p[1] * v
    n_dot_n = n.dot(n)
    t = - (n.dot(X0) + d) / n_dot_n
    X = X0 + t * n
    return X

def show_mesh(v, f, colors = None, wireframe = False, show_normal=False):
    if colors is None:
        colors = np.ones((f.shape[0], 3))
    plot = mp.plot(v, f, colors, shading={
        'wireframe': wireframe
    })
    if show_normal:
        centroid = np.mean(v[f], axis=1)
        normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
        normal = (normal.T / np.linalg.norm(normal, axis=1)).T
        plot.add_lines(
            centroid,
            centroid + normal,
            shading={
                'line_color': 'red',
                'line_width': 2
            }

        )


def edges_to_ribbons(vertices, edges, edge_colors=None, width_ratio=0.01):
    """
    Convert edges into thin ribbon faces (two triangles per edge).
    Returns new_vertices, new_faces, new_face_colors.
    """
    verts = []
    faces = []
    face_cols = []

    edge_colors = np.array(edge_colors if edge_colors is not None else (255,0,0))
    if edge_colors.ndim == 1:
        edge_colors = np.tile(edge_colors, (len(edges), 1))

    for i, (i0, i1) in enumerate(edges):
        p0 = vertices[i0]
        p1 = vertices[i1]
        d = p1 - p0
        length = np.linalg.norm(d)
        if length < 1e-9:
            continue
        d /= length

        # pick a reference vector not parallel to d
        ref = np.array([0,0,1.0]) if abs(d[2]) < 0.9 else np.array([1.0,0,0])
        n = np.cross(d, ref)
        n /= np.linalg.norm(n)

        offset = n * (length * width_ratio)

        v0 = p0 + offset
        v1 = p0 - offset
        v2 = p1 + offset
        v3 = p1 - offset

        base_idx = len(verts)
        verts.extend([v0, v1, v2, v3])

        faces.append([base_idx, base_idx+1, base_idx+2])
        faces.append([base_idx+1, base_idx+2, base_idx+3])

        c = edge_colors[i]
        face_cols.append(c)
        face_cols.append(c)

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32), np.array(face_cols, dtype=np.uint8)


def save_ply_with_edges(
    filename,
    vertices,
    faces,
    edges,
    edge_colors= None,
    vertex_colors = None,
    edge_mode = "ribbon",   # "edge" 或 "ribbon"
    width_ratio = 0.01,
):
    """
    Save a mesh with optional colored edges.
    edge_mode = "edge": write edge element (not displayed in MeshLab)
    edge_mode = "ribbon": convert edges into narrow ribbons (faces) with color
    """
    verts = np.asarray(vertices, dtype=np.float32)
    n_verts = len(verts)

    faces_arr = np.asarray(faces, dtype=np.int32) if faces is not None else None
    n_faces = 0 if faces_arr is None else len(faces_arr)

    edges_arr = np.asarray(edges, dtype=np.int32) if edges is not None else None
    n_edges = 0 if edges_arr is None else len(edges_arr)

    # vertex colors
    write_vertex_color = False
    if vertex_colors is not None:
        vcols = np.clip(np.asarray(vertex_colors, dtype=np.int32), 0, 255).astype(np.uint8)
        if vcols.shape[0] != n_verts:
            raise ValueError("vertex_colors length must match vertices")
        write_vertex_color = True
    else:
        vcols = None

    if edge_mode == "ribbon" and edges_arr is not None and n_edges > 0:
        # convert edges to ribbons
        v_rib, f_rib, fc_rib = edges_to_ribbons(verts, edges_arr, edge_colors, width_ratio=width_ratio)

        # merge with existing vertices/faces
        faces_all = []
        face_cols_all = []

        if n_faces > 0:
            faces_all.append(faces_arr)
            # 给原始 faces 白色或统一颜色
            face_cols_all.append(np.tile([200,200,200], (n_faces,1)))

        offset = len(verts)
        faces_all.append(f_rib + offset)
        face_cols_all.append(fc_rib)

        verts = np.vstack([verts, v_rib])
        faces_arr = np.vstack(faces_all)
        face_cols = np.vstack(face_cols_all)
        n_faces = len(faces_arr)

        # write as face-colored PLY
        with open(filename, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {n_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")

            for v in verts:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

            for tri, col in zip(faces_arr, face_cols):
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]} {col[0]} {col[1]} {col[2]}\n")

        print(f"Saved PLY (ribbon mode): {filename} (verts={len(verts)}, faces={n_faces})")

    else:
        # normal edge mode
        write_edge_color = False
        if edges_arr is not None and edge_colors is not None:
            ecols = np.asarray(edge_colors, dtype=np.int32)
            if ecols.ndim == 1:
                ecols = np.tile(ecols, (n_edges,1))
            ecols = np.clip(ecols, 0, 255).astype(np.uint8)
            write_edge_color = True
        else:
            ecols = None

        with open(filename, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n_verts}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if write_vertex_color:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            if n_faces > 0:
                f.write(f"element face {n_faces}\n")
                f.write("property list uchar int vertex_indices\n")
            if n_edges > 0:
                f.write(f"element edge {n_edges}\n")
                f.write("property int vertex1\nproperty int vertex2\n")
                if write_edge_color:
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")

            for i in range(n_verts):
                if write_vertex_color:
                    r,g,b = vcols[i]
                    f.write(f"{verts[i,0]} {verts[i,1]} {verts[i,2]} {r} {g} {b}\n")
                else:
                    f.write(f"{verts[i,0]} {verts[i,1]} {verts[i,2]}\n")

            if n_faces > 0:
                for tri in faces_arr:
                    f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

            if n_edges > 0:
                for ei, (v0, v1) in enumerate(edges_arr):
                    if write_edge_color:
                        r,g,b = ecols[ei]
                        f.write(f"{v0} {v1} {r} {g} {b}\n")
                    else:
                        f.write(f"{v0} {v1}\n")

        print(f"Saved PLY (edge mode): {filename} (verts={n_verts}, faces={n_faces}, edges={n_edges})")

    
def fix_face_normals_for_cylinder(vertices, faces, axis_point, axis_dir, eps=1e-9):
    """
    Check & fix face normal directions for triangular faces relative to a cylinder.

    Parameters
    ----------
    vertices : (V,3) array
        Vertex coordinates.
    faces : (F,3) int array
        Triangular faces (indices into vertices). Will not be modified in-place
        unless you reassign the return.
    axis_point : (3,) array_like
        Any point on the cylinder axis (call it c).
    axis_dir : (3,) array_like
        Cylinder axis direction vector (need not be unit length).
    eps : float
        Small threshold to detect near-zero radial vectors.

    Returns
    -------
    fixed_faces : (F,3) int array
        Faces with orientation flipped where needed so normals point outward
        from the cylinder axis (radially).
    flipped_mask : (F,) bool array
        True for faces that were flipped.
    """
    verts = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    axis_p = np.asarray(axis_point, dtype=float)
    axis_v = np.asarray(axis_dir, dtype=float)
    # normalize axis direction
    axis_len = np.linalg.norm(axis_v)
    if axis_len < eps:
        raise ValueError("axis_dir is zero-length")
    axis_u = axis_v / axis_len

    # face vertex positions
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # face centroids
    centroids = (v0 + v1 + v2) / 3.0

    # compute face normals (not normalized yet)
    # normal = cross(v1-v0, v2-v0)
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)

    # normalize normals safely (avoid division by zero)
    n_norm = np.linalg.norm(normals, axis=1)
    safe = n_norm > eps
    normals_unit = np.zeros_like(normals)
    normals_unit[safe] = normals[safe] / n_norm[safe][:, None]

    # project centroids onto axis to get closest point on axis
    # t = dot(centroid - axis_p, axis_u)
    rel = centroids - axis_p
    t = np.dot(rel, axis_u)
    proj = axis_p + np.outer(t, axis_u)  # (F,3)

    # radial vectors from axis to centroid
    radial = centroids - proj
    radial_norm = np.linalg.norm(radial, axis=1)

    # Build a decision mask: where radial norm is large enough and normal exists
    decidable = (radial_norm > eps) & safe

    # For decidable faces compute dot(normal, radial)
    dot_nr = np.einsum('ij,ij->i', normals_unit, radial)

    # faces where dot < 0 => normal points (generally) toward axis => flip
    to_flip = np.zeros(len(faces), dtype=bool)
    to_flip[decidable] = dot_nr[decidable] < 0.0

    # Flip by swapping vertex 1 and 2 for those faces
    fixed_faces = faces.copy()
    if np.any(to_flip):
        fixed_faces[to_flip, 1], fixed_faces[to_flip, 2] = faces[to_flip, 2], faces[to_flip, 1]

    return fixed_faces, to_flip


def get_origin_from_plane(plane_param):
    a, b, c, d = plane_param
    origin = -np.array([a, b, c]) * d / (a*a + b*b + c*c)
    return origin

def get_uv_from_normal(normal):
    a, b, c = normal
    
    if not np.isclose(np.abs(a), 1.0):
        u = np.array([b, -a, 0])
    else:
        u = np.array([0, c, -b])
    u_norm = np.linalg.norm(u)
    if np.isclose(u_norm, 0.0):
        u = np.array([c, 0, -a])
        u_norm = np.linalg.norm(u)
        if np.isclose(u_norm, 0.0):
            u = np.array([0, c, -b])
            u_norm = np.linalg.norm(u)
    u = u / u_norm

    v = np.cross(normal, u)
    v_norm = np.linalg.norm(v)
    v = v / v_norm

    return u, v


def get_om_vf(mesh):
    v = np.array([mesh.point(vh).tolist() for vh in mesh.vertices()])
    f = np.array([[vh.idx() for vh in mesh.fv(fh)] for fh in mesh.faces()])
    # print(v.shape, f.shape)
    return v, f


def get_single_edge_convexity(mesh, f_center, f_normal, e_h):
    threshold_convexity = np.cos(82.5 * np.pi / 180)
    threshold_convexity_dot = np.cos(82.5 * np.pi / 180)

    f0, f1 = e_h2f(mesh, e_h)

    c0, c1 = f_center[f0], f_center[f1]
    n0, n1 = f_normal[f0], f_normal[f1]

    angle0 = np.arccos(np.dot(n0, c1 - c0) / np.linalg.norm(c1 - c0))
    angle1 = np.arccos(np.dot(n1, c0 - c1) / np.linalg.norm(c0 - c1))

    metric1 = 0.5 * (np.cos(angle0) + np.cos(angle1))
    metric2 = np.dot(n0, n1)

    if metric1 >= threshold_convexity:
        return 1
    elif metric1 <= -threshold_convexity:
        return 2
    elif abs(metric2) < threshold_convexity_dot:
        if metric1 > 0:
            return 1
        elif metric1 < 0:
            return 2
    return 3

    

def get_border_edge(mesh, f_center, f_normal, threshold_convexity, get_boundary=False):
    convex_edge = []
    concave_edge = []
    boundary_edge = []
    
    for e_h in mesh.edges():
        
        if get_boundary & mesh.is_boundary(e_h):
            he_h = mesh.halfedge_handle(e_h, 0)
            v0 = mesh.from_vertex_handle(he_h).idx()
            v1 = mesh.to_vertex_handle(he_h).idx()
            boundary_edge.append([e_h.idx(), v0, v1])
            continue
        
        # if mesh.is_boundary(e_h):
        #     continue
        
        he_h0 = mesh.halfedge_handle(e_h, 0)
        he_h1 = mesh.halfedge_handle(e_h, 1)
        f0 = mesh.face_handle(he_h0).idx()
        f1 = mesh.face_handle(he_h1).idx()
        
        # normal-center angle
        c0, c1, n0, n1 = f_center[f0], f_center[f1], f_normal[f0], f_normal[f1]
        angle0 = np.arccos(np.dot(n0, c1 - c0) / np.linalg.norm(c1 - c0))
        angle1 = np.arccos(np.dot(n1, c0 - c1) / np.linalg.norm(c0 - c1))
        
        metric = 0.5 * (np.cos(angle0) + np.cos(angle1))

        # dihedral angle
        metric2 = np.dot(n0, n1)

        vi = mesh.from_vertex_handle(he_h0).idx()
        vj = mesh.to_vertex_handle(he_h0).idx()
        vi, vj = min(vi, vj), max(vi, vj)
        
        if metric >= threshold_convexity:
            concave_edge.append([e_h.idx(), vi, vj])
        elif metric <= -threshold_convexity:
            convex_edge.append([e_h.idx(), vi, vj])
        # elif abs(metric2) < np.cos(7.5 * np.pi / 180):
        # elif abs(metric2) < np.cos(12.5 * np.pi / 180):
        elif abs(metric2) < np.cos(22.5 * np.pi / 180):
        # elif abs(metric2) < threshold_convexity:
            if metric > 0:
                concave_edge.append([e_h.idx(), vi, vj])
            elif metric < 0:
                convex_edge.append([e_h.idx(), vi, vj])
                
    
    convex_edge = np.array(convex_edge)
    concave_edge = np.array(concave_edge)
    boundary_edge = np.array(boundary_edge)

    # show edges
    v, f = get_om_vf(mesh)
    plot = mp.plot(v, f, np.ones((f.shape[0], 3)), shading={
        'wireframe': True,
        # 'edge_color': 'black',
        # 'edge_width': 1
    })
    if convex_edge.shape[0] != 0:
            plot.add_lines(
            v[convex_edge[:, 1]],
            v[convex_edge[:, 2]],
            shading={
                'line_color': 'blue',
                'line_width': 3
            }
        )
    if concave_edge.shape[0] != 0:
        plot.add_lines(
            v[concave_edge[:, 1]],
            v[concave_edge[:, 2]],
            shading={
                'line_color': 'red',
                'line_width': 3
            }
        )
    if boundary_edge.shape[0] != 0:
        plot.add_lines(
            v[boundary_edge[:, 1]],
            v[boundary_edge[:, 2]],
            shading={
                'line_color': 'green',
                'line_width': 3
            }
        )
    

    if get_boundary:
        return convex_edge, concave_edge, boundary_edge
    else:
        return convex_edge, concave_edge
        
        
def split(mesh, cut_edge):
        graph = nx.Graph()
        
        
        # construct graph
        f_num = mesh.n_faces()
        for i in range(f_num):
            graph.add_node(i)
        for e_h in mesh.edges():
            if e_h.idx() in cut_edge:
                continue
            f0 = mesh.face_handle(mesh.halfedge_handle(e_h, 0)).idx()
            f1 = mesh.face_handle(mesh.halfedge_handle(e_h, 1)).idx()
            graph.add_edge(f0, f1)
        
        graph_components = nx.connected_components(graph)
        splited_f_id = []
        for c in graph_components:
            # print(len(c), c)
            splited_f_id.append(list(c))
            # splited_f_id.append(list(graph.subgraph(c).nodes))
        
        return splited_f_id
    
    
def save_points2ply(filename, points, colors=None):
    assert points.shape[1] == 3, "点云必须是Nx3的数组"
    
    has_color = colors is not None
    if has_color:
        assert colors.shape[0] == points.shape[0], "颜色与点数不匹配"
        assert colors.shape[1] == 3, "颜色必须是Nx3的数组"
    
    # 创建PLY头部信息
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ]
    
    header += [
        "end_header"
    ]
    
    # 写入文件
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        
        for i in range(len(points)):
            line = f"{points[i,0]} {points[i,1]} {points[i,2]}"
            if has_color:
                line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(line + '\n')
            
            
def write_proxy_cluster(proxy_cluster, proxy_cluster_file):
    with open(proxy_cluster_file, "w") as f:
        for cluster in proxy_cluster:
            for f_id in cluster:
                f.write(str(f_id) + " ")
            f.write("\n")

def write_region_cluster(region_cluster, region_cluster_file):
    with open(region_cluster_file, "w") as f:
        for region in region_cluster:
            if type(region) is int:
                f.write(str(region) + '\n')
            else:
                for f_id in region:
                    f.write(str(f_id) + " ")
                f.write("\n")


def read_patch_cluster(cluster_file):
    patch2f = []
    file = open(cluster_file, 'r')
    lines = file.readlines()
    for line in lines:
        patch2f.append(np.array(line.split(), dtype=int))
    return patch2f


                
def read_regioned_cluster(cluster_path, region_path):
    cluster2f = []
    region2cluster = []
    
    f = open(cluster_path, 'r')
    lines = f.readlines()
    for line in lines:
        cluster2f.append(np.array(line.split(), dtype=int))
    f = open(region_path, 'r')
    lines = f.readlines()
    for line in lines:
        region2cluster.append(np.array(line.split(), dtype=int))
        
    return cluster2f, region2cluster


def write_ply_file(v, f, f_color, ply_path):
    mesh = trimesh.Trimesh(v, f)
    mesh.visual.face_colors = (f_color * 255).astype(np.uint8)
    mesh.export(ply_path)


def write_obj_file(v, f, obj_path):
    mesh = trimesh.Trimesh(v, f)
    mesh.export(obj_path)
    
    

def get_patch_border_v(p1_f, p2_f, f):
        def face_edges(face):
            return {tuple(sorted((face[i], face[(i+1) % len(face)]))) for i in range(len(face))}
        edges1 = set()
        for fi in p1_f:
            edges1 |= face_edges(f[fi])
        edges2 = set()
        for fi in p2_f:
            edges2 |= face_edges(f[fi])
        border_edges = np.array(list(edges1 & edges2))

        edge_ordered, vert_ordered = connect_edge_to_line(border_edges)

        return vert_ordered