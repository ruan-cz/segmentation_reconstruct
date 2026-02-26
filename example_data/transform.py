import trimesh
import os

def convert_obj_to_ply(path):
    ply_file = path.replace('.obj', '.ply')
    
    mesh = trimesh.load(path)
    mesh.export(ply_file)


path = 'example_data/example_CAD/2.obj'
convert_obj_to_ply(path)
