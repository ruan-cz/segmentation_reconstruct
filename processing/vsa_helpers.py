import os
import time

import igl
import meshplot as mp
import numpy as np

from utils.mesh_utils import *
from patch_segment.core import VSA, pre_split

def test_example(v, f):
    min_coord, max_coord = np.min(v, axis=0), np.max(v, axis=0)
    global_diagonal_len = np.linalg.norm(max_coord - min_coord)
    print(f'global diagonal {global_diagonal_len}')
    time_start = time.time()
    proxy_cluster, region_cluster, face_color = pre_split(v, f, visualize=True, visualize_part=False)
    time_end = time.time()
    print(f'pre-split time: {time_end - time_start}s')
    # write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+'_split.ply'))

    # patch2f = []
    # patch2f.append(proxy_cluster[0])
    # other_f = []
    # for i in range(1, len(proxy_cluster)):
    #     other_f.extend(proxy_cluster[i])
    # patch2f.append(other_f)
    # face_color = np.ones((f.shape[0], 3))
    # face_color[patch2f[0]] = colors[0]
    # face_color[patch2f[1]] = colors[1]
    # write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+'_split.ply'))

    return proxy_cluster, region_cluster, face_color


def direct_vsa(v, f):
    proxy_num=60
    time_start = time.time()
    vsa = VSA(v, f, proxy_num=proxy_num)
    _, proxy_cluster = vsa.iteration(visualize=True)
    
    time_end = time.time()
    print(f'direct vsa time: {time_end - time_start}s')
    # clusters = []
    # for proxy_id in proxy_cluster.keys():
    #     f_id = np.array(proxy_cluster[proxy_id])
    #     clusters.append(f_id)
    clusters = [proxy_cluster[key] for key in proxy_cluster.keys()]
    # print(len(clusters))
    face_color = np.ones((f.shape[0], 3))
    for i in range(len(clusters)):
        face_color[clusters[i]] = colors[i%500]
    write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+f'_direct_vsa_{proxy_num}.ply'))

def save_into_file(proxy_cluster, region_cluster, face_color):
    write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+'.ply'))
    
    write_proxy_cluster(proxy_cluster, os.path.join(output_path, obj_name + '_proxy_cluster.txt'))
    write_region_cluster(region_cluster, os.path.join(output_path, obj_name + '_region_cluster.txt'))

def normal_cluster(obj_path, obj_name, output_path, save_ply = True, save_cluster = True):
    v, f = igl.read_triangle_mesh(os.path.join(obj_path, obj_name+'.obj'))

    face_normal = igl.per_face_normals(v, f, np.array([1., 0., 0.]))
    face_normal = face_normal / np.linalg.norm(face_normal, axis=1).reshape(-1, 1)
    face_center = np.mean(v[f], axis=1)
    face_area = igl.doublearea(v, f) / 2
    face_color = np.ones((f.shape[0], 3), dtype=float)

    normals = []
    # cluster faces by normal
    
    def find_closest_normal_index(normal, normals):
        closest_index = -1
        closest_distance = np.inf
        for i, n in enumerate(normals):
            distance = np.linalg.norm(normal - n)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        if closest_distance > 0.1:
            closest_index = -1
        return closest_index
    
    for i in range(f.shape[0]):
        normal = face_normal[i]
        
        idx = find_closest_normal_index(normal, normals)
        if idx != -1:
            face_color[i] = colors[idx + 70]
        else:
            normals.append(normal)
            face_color[i] = colors[(len(normals) - 1) % 500 + 70]

        
    patch2f = {}
    for i in range(len(normals)):
        patch2f[i] = []
    for i in range(f.shape[0]):
        normal = face_normal[i]
        idx = find_closest_normal_index(normal, normals)
        patch2f[idx].append(i)
    if save_cluster:
        with open(os.path.join(output_path, obj_name + '_proxy_cluster.txt'), 'w') as file:
            for key in patch2f.keys():
                for val in patch2f[key]:
                    file.write(str(val) + ' ')
                file.write('\n')

    
    mp.plot(v, f, c=face_color, shading={"wireframe": False})

    if save_ply:
        write_ply_file(v, f, face_color, os.path.join(output_path, obj_name+'.ply'))
