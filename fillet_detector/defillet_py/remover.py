"""用局部拉普拉斯平滑近似去圆角，不等同于 C++ 的完整几何优化。"""

from __future__ import annotations
import numpy as np
import trimesh
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve
from .config import RemoverParameters


class Remover:
    def __init__(self, mesh: trimesh.Trimesh, parameters: RemoverParameters, labels):
        self.mesh = mesh.copy()
        self.parameters = parameters
        self.labels = np.asarray(labels, dtype=np.int32)
        if len(self.labels) != len(mesh.faces):
            raise ValueError("Label count must equal the mesh face count")
        self.defillet_mesh = None
        self.focus_mesh = None
        self.fillet_mesh = mesh.submesh(
            [self.labels == 1], append=True, repair=False
        )
        self.non_fillet_mesh = mesh.submesh(
            [self.labels == 0], append=True, repair=False
        )

    def _focus_vertices(self):
        """返回待处理面和固定顶点；只处理标签边界旁的圆角面。"""
        boundary = set()
        adjacency_data = zip(
            self.mesh.face_adjacency,
            self.mesh.face_adjacency_edges,
        )
        for (left, right), edge in adjacency_data:
            if self.labels[left] != self.labels[right]:
                boundary.update(edge.tolist())
        focus_faces = np.flatnonzero(self.labels == 1)
        if boundary:
            incident = np.any(np.isin(self.mesh.faces[focus_faces], list(boundary)), axis=1)
            focus_faces = focus_faces[incident]
        return focus_faces, boundary

    def optimize(self):
        focus_faces, fixed = self._focus_vertices()
        if len(focus_faces) == 0:
            self.defillet_mesh = self.mesh.copy()
            self.focus_mesh = trimesh.Trimesh(
                vertices=np.empty((0, 3)),
                faces=np.empty((0, 3), dtype=np.int64),
            )
            return self
        focus_vertices = np.unique(self.mesh.faces[focus_faces].ravel())
        movable = ~np.isin(focus_vertices, list(fixed))
        index = {vertex: row for row, vertex in enumerate(focus_vertices)}
        rows, cols, values = [], [], []
        adjacency = {vertex: set() for vertex in focus_vertices}
        for a, b in self.mesh.edges_unique:
            if a in adjacency and b in adjacency:
                adjacency[a].add(b)
                adjacency[b].add(a)
        for vertex in focus_vertices:
            row = index[vertex]
            neighbors = adjacency[vertex]
            rows.append(row)
            cols.append(row)
            values.append(1.0)
            if neighbors:
                weight = -1.0 / len(neighbors)
                for neighbor in neighbors:
                    rows.append(row)
                    cols.append(index[neighbor])
                    values.append(weight)
        shape = (len(focus_vertices), len(focus_vertices))
        laplacian = coo_matrix((values, (rows, cols)), shape=shape).tocsr()
        positions = self.mesh.vertices[focus_vertices].copy()
        strength = max(float(self.parameters.beta_e), 1e-8)
        normal_matrix = strength * laplacian.T @ laplacian
        system = eye(len(focus_vertices), format="csr") + normal_matrix
        # 边界坐标不更新；其对右端项的贡献在全部迭代中保持不变。
        fixed_rows = np.flatnonzero(~movable)
        fixed_contribution = normal_matrix[:, fixed_rows] @ positions[fixed_rows]
        for _ in range(max(self.parameters.num_opt_iter, 0)):
            # 三个坐标轴共用同一系数矩阵，一次求解多个右端项。
            solution = spsolve(system, positions - fixed_contribution)
            positions[movable] = solution[movable]
        self.mesh.vertices[focus_vertices] = positions
        self.defillet_mesh = self.mesh
        self.focus_mesh = self.mesh.submesh([focus_faces], append=True, repair=False)
        self.fillet_mesh = self.mesh.submesh(
            [self.labels == 1], append=True, repair=False
        )
        self.non_fillet_mesh = self.mesh.submesh(
            [self.labels == 0], append=True, repair=False
        )
        return self
