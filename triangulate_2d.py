import numpy as np
import triangle

def triangulate_2d(
    points: np.ndarray,
    max_area: float = 0.1,
    min_angle: float = 20.0
):
    A = {
        'vertices': points,
        'segments': np.array([[i, (i + 1) % points.shape[0]] for i in range(points.shape[0])])
    }

    if max_area is None:
        switches = f'p'
    else:
        switches = f'pq{min_angle}a{max_area}'
    result = triangle.triangulate(A, switches)
    v, f = result['vertices'], result['triangles']

    return v, f