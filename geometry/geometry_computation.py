import numpy as np
import igl
import meshplot as mp
import openmesh as om

from geometry.geometry_process import *
from utils.mesh_utils import *

def proj_vert_to_plane(v, axis, plane_params):
    normal_p = plane_params[:3].squeeze()
    d = plane_params[3].item()

    threshold_perp = 1e-2
    dot = np.dot(normal_p, axis)
    if abs(dot) < threshold_perp:
        raise Exception('[PROJECT]:plane parallel to axis')
    t = - (np.dot(normal_p, v) + d) / dot
    v_proj = v + t * axis
    return v_proj

# def proj_vert_to_cylinder(v, axis, cylinder_params):
#     c, axis, r = cylinder_params[:3].squeeze(), cylinder_params[3:6].squeeze(), cylinder_params[6].item()



# def detect_vert_in_triangle_on_plane()

def point_in_2d_polygon(point, polygon):
    """
    Fast 2D point-in-polygon test using ray-crossing.
    
    Parameters
    ----------
    point : (2,) array-like
        Test point [x, y]
    polygon : (n,2) array-like
        Polygon vertices in order
    
    Returns
    -------
    inside : bool
    """
    x, y = point
    poly = np.asarray(polygon)
    n = len(poly)
    
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside



def insection_plane_f(mesh, plane_params, f_h, v_from_pos = None):
    # if f_h.idx() == 4903:
    #     print('')
    if v_from_pos is None:
        raise('not given first insect vertex')
    
    v_on_plane_threshold = 1e-2
    v_diff_threshold = 1e-3
    threshold_parallel = 1e-3
    # pos, id = v_pos_on_f()
    f_id = f_h.idx()

    insection_v_id = []
    for v_h in mesh.fv(f_h):
        v_pos = mesh.point(v_h)
        if abs(np.dot(v_pos, plane_params[:3]) + plane_params[3]) < v_on_plane_threshold:
            insection_v_id.append(v_h.idx())
    
    if len(insection_v_id) == 3:
        return {
            'type': 'coplanar',
            'f_id': f_id
        }
    elif len(insection_v_id) == 2:
        for i in range(2):
            v_pos = mesh.point(mesh.vertex_handle(insection_v_id[i]))
            if np.linalg.norm(v_pos - v_from_pos) < v_diff_threshold:
                return {
                    'type': 'vv',
                    'next_v_id': insection_v_id[(i + 1) % 2],
                    'f_id': f_id
                }
    elif len(insection_v_id) == 1:
        insection_v_id = insection_v_id[0]
        
        # intersect position: edge -> v 
        if np.linalg.norm(mesh.point(mesh.vertex_handle(insection_v_id)) - v_from_pos) >= v_diff_threshold:
            return {
                'type': 'ev',
                'next_v_id': insection_v_id,
                'f_id': f_id
            }
        
        # intersect position: v -> 
        op_e_id, op_v0_id, op_v1_id = v_h2op_e(mesh, mesh.vertex_handle(insection_v_id), f_h)
        op_v0_h, op_v1_h = mesh.vertex_handle(op_v0_id), mesh.vertex_handle(op_v1_id)
        op_v0, op_v1 = mesh.point(op_v0_h), mesh.point(op_v1_h)
        # d0 = np.dot(plane_params[:3], op_v0) + plane_params[3]
        # d1 = np.dot(plane_params[:3], op_v1) + plane_params[3]

        denominator = np.dot(plane_params[:3], op_v1 - op_v0)
        numerator = - (np.dot(plane_params[:3], op_v0) + plane_params[3])

        # op edge parallel to plane: not intersect to edge 
        if abs(denominator) < threshold_parallel:
            return {
                'type': 'v',
                'f_id': f_id
            }
        t = numerator / denominator
        # not intersect to edge
        if (t <= 0) or (t >= 1):
            return {
                'type': 'v',
                'f_id': f_id
            }
        v_insert = op_v0 + t * (op_v1 - op_v0)
        return {
            'type': 've',
            'next_e_id': op_e_id,
            'v_insert': v_insert,
            'f_id': f_id
        }
    else:
        intersect_e_id = []
        intersect_v_pos = []
        for e_h in mesh.fe(f_h):
            v0, v1 = e_h2v_p(mesh, e_h)

            denominator = np.dot(v1 - v0, plane_params[:3])
            numerator = - (np.dot(v0, plane_params[:3]) + plane_params[3])

            try:
                t = numerator / denominator
                if (t <= 0) or (t >= 1):
                    continue
                v_insert = v0 + t * (v1 - v0)
                intersect_e_id.append(e_h.idx())
                intersect_v_pos.append(v_insert)
            except ZeroDivisionError:
                print('[INTERSECT]: plane & only edges, but parallel')

        if len(intersect_e_id) != 2:
            raise(f'[INTERSECT]: plane & only edges, but have {len(intersect_e_id)} intersection')
        for i in range(2):
            if np.linalg.norm(intersect_v_pos[i] - v_from_pos) < v_diff_threshold:
                return {
                    'type': 'ee',
                    'from_e_id': intersect_e_id[i],
                    'to_e_id': intersect_e_id[(i+1)%2],
                    'v_insert': intersect_v_pos[(i+1)%2],
                    'f_id': f_id
                }

'''
    vertex position relative to cylinder
    + -1: outside cylinder
    +  0: on cylinder
    +  1: inside cylinder
'''
def vert_on_cylinder(c, axis, r, v):
    v_rel = v - c
    h = np.dot(v_rel, axis)
    d = np.linalg.norm(v_rel - h * axis)

    threshold_cylinder_radiu = 5 * 1e-2
    if d < r - threshold_cylinder_radiu:
        return 1
    if d > r + threshold_cylinder_radiu:
        return -1
    return 0

def vert_to_cylinder_r_diff(c, axis, r, v):
    v_rel = v - c
    h = np.dot(v_rel, axis)
    d = np.linalg.norm(v_rel - h * axis)
    return np.abs(d - r)

def insection_edge_cylinder(c, axis, r, v0, v1):
    vec0 = v0 - c
    vec1 = v1 - v0

    dot00 = np.dot(vec0, vec0)
    dot11 = np.dot(vec1, vec1)
    dot01 = np.dot(vec0, vec1)
    dot0a = np.dot(vec0, axis)
    dot1a = np.dot(vec1, axis)

    A = dot11 - dot1a * dot1a
    B = 2 * dot01 - 2 * dot0a * dot1a
    C = dot00 - dot0a * dot0a - r * r

    eps = 1e-6
    if abs(A) < eps:
        if abs(B) < eps:
            return {
                'type': 'N'
            }
        if abs(C) < eps:
            return {
                'type': 'all'
            }
        return {
            'type': 'N'
        }
    
    delta = B * B - 4 * A * C
    if delta < 0:
        return {
            'type': 'N'
        }
    if abs(delta) < eps:
        t = -B / (2 * A)
        if 0 <= t <= 1:
            return {
                'type': 1,
                't': [t]
            }
        return {
            'type': 'N'
        }
    sqrt_delta = np.sqrt(delta)
    t0, t1 = (-B - sqrt_delta) / (2 * A), (-B + sqrt_delta) / (2 * A)
    final_t = []
    for t in [t0, t1]:
        if 0 <= t <= 1:
            final_t.append(t)
    if len(final_t) == 0:
        return {
            'type': 'N'
        }
    return {
        'type': len(final_t),
        't': final_t
    }


"""
    Sample n points on the cylinder surface along border arc between p1 and p2.

    Parameters
    ----------
    p1, p2 : (3,) array_like
        Points on cylinder border.
    c : (3,) array_like
        A point on the cylinder axis.
    axis : (3,) array_like
        Cylinder axis direction.
    r : float
        Cylinder radius.
    n : int
        Number of samples.
"""
def sample_cylinder_border(p1, p2, cylinder_params, n):
    c, axis, r = cylinder_params[:3].squeeze(), cylinder_params[3:6].squeeze(), cylinder_params[6].item()
    

    # project p1,p2 onto axis
    t1 = np.dot(p1 - c, axis)
    t2 = np.dot(p2 - c, axis)
    q1 = c + t1 * axis
    q2 = c + t2 * axis

    # radial unit vectors
    r1 = (p1 - q1) / np.linalg.norm(p1 - q1)
    r2 = (p2 - q2) / np.linalg.norm(p2 - q2)

    # angle and orientation
    dot_val = np.clip(np.dot(r1, r2), -1.0, 1.0)
    theta = np.arccos(dot_val)
    # orientation via sign of cross
    if np.dot(np.cross(r1, r2), axis) < 0:
        theta = -theta

    # Rodrigues rotation function
    def rotate(v, k, ang):
        return (v * np.cos(ang) +
                np.cross(k, v) * np.sin(ang) +
                k * np.dot(k, v) * (1 - np.cos(ang)))

    samples = []
    for i in range(n):
        alpha = i / (n - 1) if n > 1 else 0
        ang = alpha * theta
        rvec = rotate(r1, axis, ang)
        tval = (1 - alpha) * t1 + alpha * t2
        samples.append(c + tval * axis + r * rvec)
    return np.array(samples)


def insection_cylinder_f_by_vert(mesh, cylinder_params, f_h, v_from_pos, v_from_type, v_from_id):
    c, axis, r = cylinder_params[:3].squeeze(), cylinder_params[3:6].squeeze(), cylinder_params[6].item()
    f_id = f_h.idx()

    max_select_num = 100
    
    if v_from_type == 'V':
        # v_prev_id = v_id2prev_v_in_f(mesh, v_from_id, f_h)
        # v_next_id = v_id2next_v_in_f(mesh, v_from_id, f_h)
        neighbor_e_v = v_id2neighbor_e_v_in_f(mesh, v_from_id, f_h)
        v_prev_id = neighbor_e_v[1]
        v_next_id = neighbor_e_v[3]

        v_prev, v_next = mesh.point(mesh.vertex_handle(v_prev_id)), mesh.point(mesh.vertex_handle(v_next_id))

        prev_on_cylinder = vert_on_cylinder(c, axis, r, v_prev)
        next_on_cylinder = vert_on_cylinder(c, axis, r, v_next)

        if (prev_on_cylinder == 0) & (next_on_cylinder == 0):
            return {
                'type': 'coplanar'
            }
        if (prev_on_cylinder == 1) & (next_on_cylinder == 1):
            return {
                'type': 'v'
            }
        if (prev_on_cylinder == -1) & (next_on_cylinder == -1):
            return {
                'type': 'v'
            }
        if (prev_on_cylinder == 0):
            return {
                'type': 'vv',
                'next_v_id': v_prev_id,
                'f_id': f_id
            }
        if (next_on_cylinder == 0):
            return {
                'type': 'vv',
                'next_v_id': v_next_id,
                'f_id': f_id
            }
        if (prev_on_cylinder * next_on_cylinder == -1):
            op_e_id = neighbor_e_v[4]
            # e_id = v_h2op_e(mesh, mesh.vertex_handle(v_from_id), f_h)[0]
            # find v on op edge that 'on' cylinder
            candidate_v_pos = []
            candidate_diff = []
            t = np.linspace(0, 1, max_select_num + 1)
            for i in range(1, max_select_num):
                tmp_v = v_prev + t[i] * (v_next - v_prev)
                diff = vert_to_cylinder_r_diff(c, axis, r, tmp_v)
                if diff < 1e-1:
                    candidate_v_pos.append(tmp_v)
                    candidate_diff.append(diff)
            insert_v = candidate_v_pos[np.argmin(candidate_diff)]

            return {
                'type': 've',
                'next_e_id': op_e_id,
                'v_insert': insert_v,
                'f_id': f_id
            }
    else:
        # neighbor_e_v = v_id2neighbor_e_v_in_f(mesh, v_from_id, f_h)
        e_id = v_from_id
        neighbor_e_v = e_id2neighbor_e_v_in_f(mesh, e_id, f_h)

        prev_e_id = neighbor_e_v[0]
        prev_v_id = neighbor_e_v[1]
        next_e_id = neighbor_e_v[2]
        next_v_id = neighbor_e_v[3]
        op_v_id = neighbor_e_v[4]

        # prev_v_id, next_v_id = e_h2v(mesh, mesh.edge_handle(e_id))
        # op_v_id = e_h2op_v_in_f_h(mesh, mesh.edge_handle(e_id), f_h)

        v_prev, v_next, v_op = mesh.point(mesh.vertex_handle(prev_v_id)), mesh.point(mesh.vertex_handle(next_v_id)), mesh.point(mesh.vertex_handle(op_v_id))

        prev_on_cylinder = vert_on_cylinder(c, axis, r, v_prev)
        next_on_cylinder = vert_on_cylinder(c, axis, r, v_next)
        op_on_cylinder = vert_on_cylinder(c, axis, r, v_op)

        if op_on_cylinder == 0:
            return {
                'type': 'ev',
                'next_v_id': op_v_id,
                'f_id': f_id
            }
        # on prev edge
        if prev_on_cylinder * op_on_cylinder == -1:
            # find v on op edge that 'on' cylinder
            candidate_v_pos = []
            candidate_diff = []
            t = np.linspace(0, 1, max_select_num + 1)
            for i in range(1, max_select_num):
                tmp_v = v_prev + t[i] * (v_op - v_prev)
                diff = vert_to_cylinder_r_diff(c, axis, r, tmp_v)
                if diff < 1e-1:
                    candidate_v_pos.append(tmp_v)
                    candidate_diff.append(diff)
            insert_v = candidate_v_pos[np.argmin(candidate_diff)]
            return {
                'type': 'ee',
                'to_e_id': prev_e_id,
                'v_insert': insert_v,
                'f_id': f_id
            }
        elif next_on_cylinder * op_on_cylinder == -1:
            # find v on op edge that 'on' cylinder
            candidate_v_pos = []
            candidate_diff = []
            t = np.linspace(0, 1, max_select_num + 1)
            for i in range(1, max_select_num):
                tmp_v = v_next + t[i] * (v_op - v_next)
                diff = vert_to_cylinder_r_diff(c, axis, r, tmp_v)
                if diff < 1e-1:
                    candidate_v_pos.append(tmp_v)
                    candidate_diff.append(diff)
            insert_v = candidate_v_pos[np.argmin(candidate_diff)]
            return {
                'type': 'ee',
                'to_e_id': next_e_id,
                'v_insert': insert_v,
                'f_id': f_id
            }


    

def insection_cylinder_f(mesh, cylinder_params, f_h, v_from_pos = None):
    if v_from_pos is None:
        raise('not given first insect vertex')

    c, axis, r = cylinder_params[:3].squeeze(), cylinder_params[3:6].squeeze(), cylinder_params[6].item()
    
    # for i in range(3):
    #     c[i] = round(c[i], 1)
    #     axis[i] = round(axis[i], 1)
    # r = round(r, 1)


    v_diff_threshold = 1e-2
    threshold_param = 5 * 1e-2
    threshold_parallel = 1e-3
    f_id = f_h.idx()

    insection_v_id = []
    for v_h in mesh.fv(f_h):
        v_pos = mesh.point(v_h)
        # vert_on_cylinder(cylinder_params, v_pos)
        if vert_on_cylinder(c, axis, r, v_pos) == 0:
            insection_v_id.append(v_h.idx())
    
    if len(insection_v_id) == 3:
        return {
            'type': 'coplanar',
            'f_id': f_id
        }
    elif len(insection_v_id) == 2:
        for i in range(2):
            v_pos = mesh.point(mesh.vertex_handle(insection_v_id[i]))
            if np.linalg.norm(v_pos - v_from_pos) < v_diff_threshold:
                return {
                    'type': 'vv',
                    'next_v_id': insection_v_id[(i + 1) % 2],
                    'f_id': f_id
                }
    elif len(insection_v_id) == 1:
        insection_v_id = insection_v_id[0]
        
        cur_v = mesh.point(mesh.vertex_handle(insection_v_id))

        # intersect position: edge -> v 
        if np.linalg.norm(cur_v - v_from_pos) >= v_diff_threshold:
            return {
                'type': 'ev',
                'next_v_id': insection_v_id,
                'f_id': f_id
            }
        
        # intersect position: v ->
        prev_e_id, prev_v_id = v_id2prev_e_in_f(mesh, insection_v_id, f_h)
        next_e_id, next_v_id = v_id2next_e_in_f(mesh, insection_v_id, f_h)
        op_e_id, _, _ = v_h2op_e(mesh, mesh.vertex_handle(insection_v_id), f_h)
        prev_v, next_v = mesh.point(mesh.vertex_handle(prev_v_id)), mesh.point(mesh.vertex_handle(next_v_id))

        # c, axis, r = cylinder_params[:3], cylinder_params[3:6], cylinder_params[6]
        # check op edge
        op_result = insection_edge_cylinder(c, axis, r, prev_v, next_v)
        if op_result['type'] == 1:
            t = op_result['t'][0]
            
            if abs(t) < threshold_param:
                return {
                    'type': 'vv',
                    'next_v_id': prev_v_id,
                    'f_id': f_id
                }
            if abs(1 - t) < threshold_param:
                return {
                    'type': 'vv',
                    'next_v_id': next_v_id,
                    'f_id': f_id
                }
            v_insert = prev_v + t * (next_v - prev_v)
            return {
                'type': 've',
                'next_e_id': op_e_id,
                'v_insert': v_insert,
                'f_id': f_id
            }        
        
        prev_result = insection_edge_cylinder(c, axis, r, cur_v, prev_v)
        if prev_result['type'] == 2:
            t = prev_result['t'][1]
            if abs(1 - t) < threshold_param:
                return {
                    'type': 'vv',
                    'next_v_id': prev_v_id,
                    'f_id': f_id
                }
            v_insert = cur_v + t * (prev_v - cur_v)
            return {
                'type': 've',
                'next_e_id': prev_e_id,
                'v_insert': v_insert,
                'f_id': f_id
            }
        # elif (prev_result['type'] == 'N') | (prev_result['type'] == 'all'):
        #     raise(f'[INTERSECT]: cylinder & only edges, but have {prev_result["type"]} intersection on prev edge')
        
        next_result = insection_edge_cylinder(c, axis, r, cur_v, next_v)
        if next_result['type'] == 2:
            t = next_result['t'][1]
            if abs(1 - t) < threshold_param:
                return {
                    'type': 'vv',
                    'next_v_id': next_v_id,
                    'f_id': f_id
                }
            v_insert = cur_v + t * (next_v - cur_v)
            return {
                'type': 've',
                'next_e_id': next_e_id,
                'v_insert': v_insert,
                'f_id': f_id
            }
        
        # elif (next_result['type'] == 'N') | (next_result['type'] == 'all'):
        #     raise(f'[INTERSECT]: cylinder & only edges, but have {next_result["type"]} intersection on next edge')
        
        # op 0, prev 1, next 1
        return {
            'type': 'v',
            'f_id': f_id
        }
    # no vertex on cylinder
    else:
        # c, axis, r = cylinder_params[:3].flatten(), cylinder_params[3:6].flatten(), cylinder_params[6].flatten()
    
        intersect_e_id = []
        intersect_v_pos = []
        from_e_id = -1
        for e_h in mesh.fe(f_h):
            v0, v1 = e_h2v_p(mesh, e_h)

            result = insection_edge_cylinder(c, axis, r, v0, v1)
            if result['type'] == 'N':
                continue
            elif result['type'] == 'all':
                raise '[INTERSECT]: cylinder & only edges, but edge all on cylinder'
            
            t = result['t']
            for tmp_t in t:
                v_insert = v0 + tmp_t * (v1 - v0)
                if np.linalg.norm(v_insert - v_from_pos) < v_diff_threshold:
                    from_e_id = e_h.idx()
                    continue
                intersect_e_id.append(e_h.idx())
                intersect_v_pos.append(v_insert)
        
        if from_e_id == -1:
            raise f'[INTERSECT]: cylinder & only edges, but cannot find from edge'
        if len(intersect_e_id) == 0:
            return {
                'type': 'e',
                'from_e_id': from_e_id,
                'v_insert': v_from_pos,
                'f_id': f_id
            }
        return {
            'type': 'ee',
            'from_e_id': from_e_id,
            'to_e_id': intersect_e_id,
            'v_insert': intersect_v_pos,
            'f_id': f_id
        }

        
def sample_from_cylinder_on_face(
        mesh, cylinder_params, f_id, 
        v_from_pos, v_to_pos, threshold_sample_len):
    c = cylinder_params[:3].reshape(3)
    axis = cylinder_params[3:6].reshape(3)
    r = float(cylinder_params[6].item())

    # face plane param
    v_idx = [v_h.idx() for v_h in mesh.fv(mesh.face_handle(f_id))]
    v_pos = [mesh.point(mesh.vertex_handle(v_id)) for v_id in v_idx]

    face_normal = np.cross(v_pos[1] - v_pos[0], v_pos[2] - v_pos[0])
    face_normal = face_normal / np.linalg.norm(face_normal)
    face_d = - np.dot(face_normal, v_pos[0])


    # if parallel to axis, no sampling
    if abs(np.dot(face_normal, axis)) < 1e-2:
        return None
    

    # if intersect_type == 'vv':
    #     v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
    #     v_to_pos = mesh.point(mesh.vertex_handle(v_to_id))
    # elif intersect_type == 've':
    #     v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
    #     v_to_pos = mesh.point(mesh.vertex_handle(v_to_id))
    # elif intersect_type == 'ev':
    #     v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
    #     v_to_pos = mesh.point(mesh.vertex_handle(v_to_id))
    # elif intersect_type == 'ee':
    #     v_from_pos = mesh.point(mesh.vertex_handle(v_from_id))
    #     v_to_pos = mesh.point(mesh.vertex_handle(v_to_id))
    # else:
    #     raise(f'[SAMPLE]: unknown intersect type {intersect_type}')


    # project v_from, v_to on cylinder axis' orthonormal plane
    u, v = get_uv_from_normal(axis)
    
    # v_from_proj = project_point_to_2d(v_from_pos, c, u, v)
    # v_to_proj = project_point_to_2d(v_to_pos, c, u, v)
    
    def decompose(point):
        vec = point - c
        axis_len = np.dot(vec, axis)
        axis_vec = axis_len * axis
        radial_vec = vec - axis_vec
        radial_len = np.linalg.norm(radial_vec)
        if radial_len < 1e-2:
            raise ValueError('[SAMPLE]: point on axis, cannot decompose')
        radial_unit = radial_vec / radial_len
        theta = np.arctan2(np.dot(radial_unit, v), np.dot(radial_unit, u))
        return axis_len, theta
    
    v_from_axis_len, v_from_theta = decompose(v_from_pos)
    v_to_axis_len, v_to_theta = decompose(v_to_pos)

    
    dot_n_axis = np.dot(face_normal, axis)
    dot_n_u, dot_n_v = np.dot(face_normal, u), np.dot(face_normal, v)
    perp_circle = abs(dot_n_axis) < 1e-4
    def axis_len_of_theta(theta):
        if perp_circle:
            return (v_from_axis_len + v_to_axis_len) / 2.
        
        return - (np.dot(face_normal, c) + face_d + r * (np.cos(theta) * dot_n_u + np.sin(theta) * dot_n_v)) / dot_n_axis

    # get arc from v_from to v_to
    diff_theta = v_to_theta - v_from_theta
    # diff_theta = (diff_theta + np.pi) % (2 * np.pi) - np.pi

    arc_len = abs(diff_theta) * r
    if arc_len <= threshold_sample_len:
        return None
    
    # sample
    sample_num = int(arc_len / threshold_sample_len) + 1
    sample_points = []
    for i in range(sample_num):
        alpha = (i + 1) / (sample_num + 1)
        theta = v_from_theta + alpha * diff_theta
        axis_len = axis_len_of_theta(theta)
        point = c + axis_len * axis + r * (np.cos(theta) * u + np.sin(theta) * v)
        sample_points.append(point)
    sample_points = np.array(sample_points)

    return sample_points
        


def v_pos_on_f_v(mesh, f_h, v_insert):
    v_diff_threshold = 1e-2

    for v_h in mesh.fv(f_h):
        v_pos = mesh.point(v_h)
        if np.linalg.norm(v_pos - v_insert) < v_diff_threshold:
            return v_h.idx()

    return -1

def v_pos_on_f(mesh, f_h, v_insert):

    v_diff_threshold = 1e-2

    for v_h in mesh.fv(f_h):
        v_pos = mesh.point(v_h)
        if np.linalg.norm(v_pos - v_insert) < v_diff_threshold:
            return 'v', v_h.idx()
    
    for e_h in mesh.fh(f_h):
        v0_pos, v1_pos = e_h2v_p(e_h)

        


