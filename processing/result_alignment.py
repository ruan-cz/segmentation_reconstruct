import numpy as np
import igl
import networkx as nx
import trimesh
from collections import defaultdict
from collections import Counter
import time
from shapely.geometry import Polygon 
import matplotlib.pyplot as plt



# read 2 meshes, resize and translate mesh2 to mesh1
def read_and_move(example_mesh_path, mesh_pathes):
    example_mesh = trimesh.load(example_mesh_path)
    meshes = [trimesh.load(p) for p in mesh_pathes]

    # resize mesh2 to match mesh1's dimensions
    for mesh in meshes:
        mesh.apply_scale(example_mesh.scale / mesh.scale)

    # translate mesh2 to mesh1's center
    # mesh2.apply_translation(mesh1.center_mass - mesh2.center_mass)
    # mesh3.apply_translation(mesh1.center_mass - mesh3.center_mass)
    # mesh4.apply_translation(mesh1.center_mass - mesh4.center_mass)
    example_mesh_v0 = example_mesh.vertices[0]
    for mesh in meshes:
        mesh_v0 = mesh.vertices[0]
        diff = example_mesh_v0 - mesh_v0
        mesh.apply_translation(diff)

    # save mesh2 to raw file
    for i, mesh in enumerate(meshes):
        mesh.export(mesh_pathes[i][:-4] + '_moved.ply')


name = '00140223_5374ce893909eddbc7536b71_trimesh_000'
read_and_move(
    f'example_results/merge/selected/example_result/our/{name}.ply',       
    # f'example_results/merge/selected/example_result/part2surf/{name}.ply',
    # f'example_results/merge/selected/example_result/samesh/{name}.ply', 
    # f'example_results/merge/selected/example_result/sdf/{name}.ply',
    [
        f'example_results/merge/selected/example_result/partfield/{name}.ply'
    ]
)
