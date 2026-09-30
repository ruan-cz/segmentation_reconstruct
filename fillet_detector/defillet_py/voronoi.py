"""Bounded 3-D Voronoi vertices from a Delaunay tetrahedralization."""
from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay


def voronoi3d(sites: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]):
    """以四面体外接球心构造 Voronoi 顶点，只保留包围盒内的球心。"""
    sites = np.asarray(sites, dtype=float)
    if len(sites) < 4:
        return np.empty((0, 3)), np.empty(0), []
    lo, hi = bounds
    span = np.linalg.norm(hi - lo) * 10.0
    corners = np.array([[x, y, z] for x in (lo[0]-span, hi[0]+span)
                        for y in (lo[1]-span, hi[1]+span)
                        for z in (lo[2]-span, hi[2]+span)])
    all_sites = np.vstack((sites, corners))
    tri = Delaunay(all_sites)
    vertices, radii, generators = [], [], []
    for tetra in tri.simplices:
        points = all_sites[tetra]
        # 外接球的等距方程相减，消去球心平方项，得到三元线性系统。
        matrix = 2.0 * (points[1:] - points[0])
        rhs = np.sum(points[1:] ** 2, axis=1) - np.sum(points[0] ** 2)
        try:
            center = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            continue
        # Match the original CGAL implementation exactly.  The C++ code adds
        # eight auxiliary box-corner sites, but only stores mappings for the
        # real input sites.  Looking up an auxiliary site in std::map therefore
        # yields the default index 0.  We preserve that behavior here instead
        # of silently discarding every tetrahedron touching an auxiliary site.
        if np.all(center >= lo) and np.all(center <= hi):
            mapped = np.where(tetra < len(sites), tetra, 0).astype(np.int64)
            vertices.append(center)
            radii.append(float(np.linalg.norm(sites[mapped] - center, axis=1).mean()))
            generators.append(mapped.tolist())
    return np.asarray(vertices), np.asarray(radii), generators
