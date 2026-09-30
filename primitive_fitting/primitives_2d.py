import numpy as np
from scipy import optimize


def fit_circle(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(points) < 3:
        raise ValueError("at least three points are required to fit a circle")
    centered = points - np.mean(points, axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("circle fitting requires non-collinear points")

    init_c = np.mean(points, axis=0)

    def circle_residuals(c, points):
        x0, y0 = c
        radii = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2)
        return radii - np.mean(radii)
    
    result = optimize.least_squares(circle_residuals, init_c, args=(points,))
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("circle fitting did not converge")
    c = result.x

    radii = np.sqrt((points[:, 0] - c[0]) ** 2 + (points[:, 1] - c[1]) ** 2)
    r = np.mean(radii)

    return c, r
