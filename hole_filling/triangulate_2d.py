import numpy as np
try:
    import triangle
except ModuleNotFoundError:
    triangle = None

def triangulate_2d(
    points: np.ndarray,
    max_area: float = 0.1,
    min_angle: float = 20.0
):
    if triangle is None:
        raise ImportError("triangulate_2d requires the optional 'triangle' package")
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
