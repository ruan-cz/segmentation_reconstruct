import numpy as np
# from meshpy.triangle import *
import triangle
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def plot_triangle(tri):
    vertices = tri['vertices']
    triangles = tri['triangles']
    plt.figure()
    for tri_idx in triangles:
        pts = vertices[tri_idx]
        poly = plt.Polygon(pts, edgecolor='k', facecolor='none', linewidth=0.7)
        plt.gca().add_patch(poly)
    plt.scatter(vertices[:,0], vertices[:,1], s=6, color='red')
    plt.axis('equal')
    plt.show()


'''
    INPUT:
    + outer_boundary: np.array(n, 2)
    + inner_boundary: list of np.array(m, 2)
    + min_angle
    OUTPUT:
    + vertices
    + triangles
'''
def triangulate_2d_plane_with_hole(outer_boundary, inner_boundary=None, min_angle=None):
    
    
    pass
    # 1.init
    # all vertices, all line segments, all inner hole points
    vertices = outer_boundary
    segments = []
    p_inner_hole = []

    len_outer_boundary = outer_boundary.shape[0]
    segments = [[i, (i + 1) % len_outer_boundary] for i in range(len_outer_boundary)]
    cur_idx = len_outer_boundary
    
    if inner_boundary is not None:
        for i in range(len(inner_boundary)):
            cur_hole = inner_boundary[i]
            len_cur_hole = cur_hole.shape[0]

            vertices = np.concatenate((vertices, cur_hole), axis=0)
            segments.extend([[cur_idx + j, cur_idx + (j + 1) % len_cur_hole] for j in range(len_cur_hole)])
            cur_idx += len_cur_hole
            
            p_inner_hole.append(np.mean(cur_hole, axis=0))

    vertices = np.array(vertices, dtype=float)
    segments = np.array(segments, dtype=np.int32)
    p_inner_hole = (
        np.array(p_inner_hole, dtype=float).reshape(-1, 2) if len(p_inner_hole) > 0 else None
    )



    # 2.triangulate
    if min_angle is not None:
        flag = f"pYq{float(min_angle)}"
    else:
        flag = "pY"

    # info = MeshInfo()
    # info.set_points(vertices)
    # info.set_facets(segments)
    # info.set_holes(p_inner_hole)

    # mesh = build(info)


    if p_inner_hole is not None:
        tri = triangle.triangulate({
            'vertices': vertices, 
            'segments': segments, 
            'holes': p_inner_hole},
        flag)
    else:
        tri = triangle.triangulate({
            'vertices': vertices, 
            'segments': segments}, 
        flag)

    print(tri['vertices'].shape)
    print(tri['triangles'].shape)
    plot_triangle(tri)

    return tri['vertices'], tri['triangles']
    return np.array(mesh.points), np.array(mesh.elements)
    


if __name__ == '__main__':
    # n1 = 30
    # n2 = 10
    # outer_boundary = np.array([[10 * np.cos(t), 10 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, n1)])
    # inner_boundary1 = np.array([[ 5 + 2 * np.cos(t),  5 + 2 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, n2)])
    # inner_boundary2 = np.array([[-5 + 2 * np.cos(t), -5 + 2 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, n2)])
    outer = np.array([(10*np.cos(t), 10*np.sin(t)) for t in np.linspace(0, 2*np.pi, 32, endpoint=False)])
    hole = [[(5 + 1.5*np.cos(t), 5 + 1.5*np.sin(t)) for t in np.linspace(0, 2*np.pi, 32, endpoint=False)]]

    
    # triangulate_2d_plane_with_hole(
    #     outer_boundary
    # )

    triangulate_2d_plane_with_hole(
        outer, hole
    )
    
    


    
    
            
            

        
