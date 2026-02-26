import numpy as np
import igl
import meshplot as mp
import openmesh as om

def e2v(mesh, e_id):
    heh = mesh.halfedge_handle(mesh.edge_handle(e_id), 0)
    v0h = mesh.from_vertex_handle(heh)
    v1h = mesh.to_vertex_handle(heh)
    return v0h.idx(), v1h.idx()

def e2ano_v(mesh, e_id, v_id):
    v0_id, v1_id = e2v(mesh, e_id)
    if v_id == v0_id:
        return v1_id
    elif v_id == v1_id:
        return v0_id
    else:
        raise ValueError("[OM ERROR]: v_id not in e_id")

def e_pair2v(mesh, e_id1, e_id2):
    v0_id1, v1_id1 = e2v(mesh, e_id1)
    v0_id2, v1_id2 = e2v(mesh, e_id2)
    for v_id in [v0_id1, v1_id1]:
        if v_id in [v0_id2, v1_id2]:
            return v_id
    raise ValueError("[OM ERROR]: no common vertex")

def e_h2v(mesh, eh):
    heh = mesh.halfedge_handle(eh, 0)
    v0h = mesh.from_vertex_handle(heh)
    v1h = mesh.to_vertex_handle(heh)
    return v0h.idx(), v1h.idx()

def e_h2v_p(mesh, eh):
    heh = mesh.halfedge_handle(eh, 0)
    v0h = mesh.to_vertex_handle(heh)
    v1h = mesh.from_vertex_handle(heh)
    return mesh.point(v0h), mesh.point(v1h)

def e_h2f(mesh, e_h):
    he0_h, he1_h = mesh.halfedge_handle(e_h, 0), mesh.halfedge_handle(e_h, 1) 
    f0h, f1h = mesh.face_handle(he0_h), mesh.face_handle(he1_h)
    return f0h.idx(), f1h.idx()

def e_h2nf(mesh, eh):
    he0h, he1h = mesh.halfedge_handle(eh, 0), mesh.halfedge_handle(eh, 1)
    f0h, f1h = mesh.face_handle(he0h), mesh.face_handle(he1h)
    return f0h.idx(), f1h.idx()

def e2op_f(mesh, e_id, f_id):
    e_h = mesh.edge_handle(e_id)
    f0_id, f1_id = e_h2f(mesh, e_h)
    if f0_id == f_id:
        return f1_id
    elif f1_id == f_id:
        return f0_id
    else:
        raise ValueError("[OM ERROR]: e_id not in f_id")

def e_h2ov_in_face(mesh, eh, f):
    he0h, he1h = mesh.halfedge_handle(eh, 0), mesh.halfedge_handle(eh, 1)
    v0h, v1h = mesh.opposite_vh(he0h), mesh.opposite_vh(he1h)
    for v_id in [v0h.idx(), v1h.idx()]:
        if v_id in f:
            return v_id
    return -1
    raise ValueError("[OM ERROR]: cannot find oppotise v in f")

def e2ov_in_face(mesh, e, f):
    eh = mesh.edge_handle(e)
    he0h, he1h = mesh.halfedge_handle(eh, 0), mesh.halfedge_handle(eh, 1)
    v0h, v1h = mesh.opposite_vh(he0h), mesh.opposite_vh(he1h)
    for v_id in [v0h.idx(), v1h.idx()]:
        if v_id in f:
            return v_id
    raise ValueError("[OM ERROR]: cannot find oppotise v in f")



def e_h2op_v_in_f_h(mesh, eh, fh):
    tmp_v = e_h2v(mesh, eh)
    for v_h in mesh.fv(fh):
        if v_h.idx() not in tmp_v:
            return v_h.idx()
        
def e2op_v_in_f(mesh, e, f):
    tmp_v = e_h2v(mesh, mesh.edge_handle(e))
    for v_h in mesh.fv(mesh.face_handle(f)):
        if v_h.idx() not in tmp_v:
            return v_h.idx()

def e_h2v(mesh, e_h):
    he_h0 = mesh.halfedge_handle(e_h, 0)
    v0h = mesh.from_vertex_handle(he_h0)
    v1h = mesh.to_vertex_handle(he_h0)
    return v0h.idx(), v1h.idx()

def e2v(mesh, e_id):
    e_h = mesh.edge_handle(e_id)
    he_h0 = mesh.halfedge_handle(e_h, 0)
    v0h = mesh.from_vertex_handle(he_h0)
    v1h = mesh.to_vertex_handle(he_h0)
    return v0h.idx(), v1h.idx()

def v_h2op_e(mesh, v_h, f_h):
    v_id = v_h.idx()
    for e_h in mesh.fe(f_h):
        v0_id, v1_id = e_h2v(mesh, e_h)
        if v_id not in [v0_id, v1_id]:
            return e_h.idx(), v0_id, v1_id

def v_id2prev_e_in_f(mesh, v_id, f_h):
    for e_h in mesh.fe(f_h):
        v0_id, v1_id = e_h2v(mesh, e_h)
        if v_id == v1_id:
            return e_h.idx(), v0_id
    raise ValueError("[OM ERROR]: cannot find next edge in f")


def v_id2prev_v_in_f(mesh, v_id, f_h):
    for e_h in mesh.fe(f_h):
        v0_id, v1_id = e_h2v(mesh, e_h)
        if v_id == v1_id:
            return v0_id
    raise ValueError("[OM ERROR]: cannot find prev v in f")

def v_id2next_e_in_f(mesh, v_id, f_h):
    for e_h in mesh.fe(f_h):
        v0_id, v1_id = e_h2v(mesh, e_h)
        if v_id == v0_id:
            return e_h.idx(), v1_id
    raise ValueError("[OM ERROR]: cannot find next edge in f")

def v_id2neighbor_e_v_in_f(mesh, v_id, f_h):
    results = []
    op_e_id = -1
    for e_h in mesh.fe(f_h):
        v0, v1 = e_h2v(mesh, e_h)
        if v_id == v0:
            results.append(e_h.idx())
            results.append(v1)
        if v_id == v1:
            results.append(e_h.idx())
            results.append(v0)
        if v_id not in [v0, v1]:
            op_e_id = e_h.idx()
    results.append(op_e_id)
    return results

def e_id2neighbor_e_v_in_f(mesh, e_id, f_h):
    prev_e, next_e, op_v = -1, -1, -1
    prev_v, next_v = e_h2v(mesh, mesh.edge_handle(e_id))
    for e_h in mesh.fe(f_h):
        v0, v1 = e_h2v(mesh, e_h)
        if (prev_v in [v0, v1]) and (next_v not in [v0, v1]):
            prev_e = e_h.idx()
            op_v = v0 if v1 == prev_v else v1
        if (next_v in [v0, v1]) and (prev_v not in [v0, v1]):
            next_e = e_h.idx()

    results = [prev_e, prev_v, next_e, next_v, op_v]
    return results

            
def v2op_v(mesh, v0_id, v1_id, f_id):
    f_h = mesh.face_handle(f_id)
    for v_h in mesh.fv(f_h):
        if v_h.idx() not in [v0_id, v1_id]:
            return v_h.idx()
    raise ValueError("[OM ERROR]: cannot find oppotise v in f")


def v_id2next_v_in_f(mesh, v_id, f_h):
    for e_h in mesh.fe(f_h):
        v0_id, v1_id = e_h2v(mesh, e_h)
        if v_id == v0_id:
            return v1_id
    raise ValueError("[OM ERROR]: cannot find next v in f")

def v_pair2e(mesh, v0_id, v1_id):
    v0, v1 = mesh.vertex_handle(v0_id), mesh.vertex_handle(v1_id)
    for he_h in mesh.voh(v0):
        if mesh.to_vertex_handle(he_h) == v1:
            return mesh.edge_handle(he_h).idx()

def v_pair2e_h(mesh, v0_id, v1_id):
    v0, v1 = mesh.vertex_handle(v0_id), mesh.vertex_handle(v1_id)
    for he_h in mesh.voh(v0):
        if mesh.to_vertex_handle(he_h) == v1:
            return mesh.edge_handle(he_h)
        
def face_normal(v, f):
    v0, v1, v2 = v[f[0]], v[f[1]], v[f[2]]
    normal = np.cross(v1 - v0, v2 - v0)
    normal = normal / np.linalg.norm(normal)
    return normal