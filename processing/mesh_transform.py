import numpy as np
import trimesh
import open3d as o3d


def transform_obj_to_off(obj_path, off_path):
    """
    Transform .obj file to .off file.
    Args:
        obj_path: path to the .obj file
        off_path: path to the .off file
    """
    mesh = trimesh.load(obj_path)
    mesh.export(off_path, file_type='off')


def save_voxelgrid(voxel_grid, output_path):
    o3d.io.write_voxel_grid(output_path, voxel_grid)
    
    # voxels = voxel_grid.get_voxels()
    # indices = np.array([voxel.grid_index for voxel in voxels])
    # colors = np.array([voxel.color for voxel in voxels]) if voxel_grid.has_colors() else None

    # np.savez_compressed(
    #     output_path,
    #     grid_indices=indices,
    #     colors=colors,
    #     origin=np.array(voxel_grid.origin),
    #     voxel_size=voxel_grid.voxel_size
    # )
    

def read_voxelgrid(voxel_grid_file, visualize = False):
    # data = np.load(voxel_grid_file)
    
    # grid_indices = data['grid_indices']
    # origin = data['origin']
    # voxel_size = data['voxel_size'].item()
    
    # colors = data["colors"] if 'colors' in data else None

    # centers = origin + grid_indices * voxel_size

    # min_bound = origin
    # max_bound = origin + (grid_indices.max(axis=0) + 1) * voxel_size

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(centers)
    # if colors is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(colors)

    # reconstructed_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
    #     pcd,
    #     voxel_size=voxel_size,
    #     min_bound=min_bound,
    #     max_bound=max_bound
    # )
    voxel_grid = o3d.io.read_voxel_grid(voxel_grid_file)
    
    if visualize:
        o3d.visualization.draw_geometries([voxel_grid], width=800, height=600)

