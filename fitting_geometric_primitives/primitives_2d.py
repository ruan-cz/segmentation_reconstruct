import numpy as np
from scipy import optimize


def fit_circle(points):

    init_c = np.mean(points, axis=0)

    def circle_residuals(c, points):
        x0, y0 = c
        radii = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2)
        return radii - np.mean(radii)
    
    c, _ = optimize.leastsq(circle_residuals, init_c, args=(points,))

    radii = np.sqrt((points[:, 0] - c[0]) ** 2 + (points[:, 1] - c[1]) ** 2)
    r = np.mean(radii)

    return c, r
