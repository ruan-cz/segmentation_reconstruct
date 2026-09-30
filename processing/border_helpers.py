import os

import igl
import meshplot as mp
import numpy as np

from utils.mesh_utils import *

def save_raw_ply(obj_path, patch_path):
    v, f = igl.read_triangle_mesh(obj_path)
    patch2f = []
    patch_file = open(patch_path, 'r')
    lines = patch_file.readlines()
    for line in lines:
        f_list = line.split(' ')
        f_list = [int(f_id) for f_id in f_list if f_id != '' and f_id != '\n']
        patch2f.append(f_list)
    patch_file.close()
    f_colors = np.ones((f.shape[0], 3))
    for i in range(len(patch2f)):
        f_colors[patch2f[i]] = colors[i%500]
    mp.plot(v, f, f_colors, shading={"wireframe": False})
    write_ply_file(v, f, f_colors, output_ply.replace('.ply', '_raw.ply'))

def show_each():
    for i in range(len(Objects)):
        obj = Objects[i]
        v, f, patch2f = obj['v'], obj['f'], obj['patch']
        f_colors = np.ones((f.shape[0], 3))
        print(f"Object {i}, Patch {len(patch2f)}")
        for p_id in range(len(patch2f)):
            f_colors[patch2f[p_id]] = colors[p_id % 100]
        mp.plot(v, f, f_colors, shading={
            'wireframe': False
        })


def show_each_patch():
    for i in range(len(Objects)):
        obj = Objects[i]
        v, f, patch2f = obj['v'], obj['f'], obj['patch']
        for p_id in range(len(patch2f)):
            print(f"Object {i}, Patch {p_id}, {len(patch2f[p_id])}")
            f_colors = np.ones((f.shape[0], 3))
            f_colors[patch2f[p_id]] = colors[p_id % 100]
            mp.plot(v, f, f_colors, shading={
                'wireframe': False
            })
        break


def show_all(write = False):
    v_all = None
    f_all = None
    cur_v_len = 0
    cur_f_len = 0
    patch2f_all = []
    for obj in Objects:
        v, f, patch2f = obj['v'], obj['f'], obj['patch']
        if v_all is None:
            v_all = v
        else:
            v_all = np.concatenate((v_all, v), axis=0)
        if f_all is None:
            f_all = f
        else:
            f = f + cur_v_len
            f_all = np.concatenate((f_all, f), axis=0)
        
        for i in range(len(patch2f)):
            patch2f_all.append(list(np.array(patch2f[i]) + cur_f_len))

        cur_v_len = v_all.shape[0]
        cur_f_len = f_all.shape[0]
        

    f_colors = np.ones((f_all.shape[0], 3))
    for i in range(len(patch2f_all)):
        f_colors[patch2f_all[i]] = colors[i % 100]
    mp.plot(v_all, f_all, f_colors, shading={
        'wireframe': True
    })

    if write:
        write_ply_file(v_all, f_all, f_colors, 
            # f'example_results/merge/example_CAD/{obj_name}_seg.ply'
            f'example_results/merge/selected/example_result/part2surf/{obj_name}.ply'
            # f'example_results/merge/selected/example_result/our/{obj_name}.ply'
            # f'example_results/merge/cad/cad16/{obj_name}.ply'
        )


def save_all_objects():
    # folder = f'example_results/merge/selected/{obj_name}/seg_choose_2/'
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    global_color_id = 0
    for i in range(len(Objects)):
        obj = Objects[i]
        v, f, patch2f = obj['v'], obj['f'], obj['patch']
        f_colors = np.ones((f.shape[0], 3))
        for p_id in range(len(patch2f)):
            f_colors[patch2f[p_id]] = colors[global_color_id % 100]
            global_color_id += 1
        write_ply_file(v, f, f_colors,
            # f'example_results/merge/cad/{obj_name}/iter_{iteration_time}_part_{i}.ply'
            # f'example_results/merge/selected/{obj_name}/seg_choose_2/iter_{iteration_time}_part_{i}.ply'
            # f'example_results/merge/selected/example_result/our/{obj_name}.ply'
        )
