"""圆角检测：Voronoi 候选 → 相切筛选 → 半径/变化率场 → 二元图割。"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import argparse
import heapq
import multiprocessing
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import trimesh

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "defillet_py"

from .config import DetectorParameters
from .graphcut import binary_cut
from .mesh import face_data
from .voronoi import voronoi3d


@dataclass
class VoronoiVertex:
    position: np.ndarray
    radius: float
    sites: list[int]
    density: float = 0.0
    axis: np.ndarray | None = None
    corr_sites: list[int] | None = None


_OSCULATING_DATA = None


def _initialize_osculation_worker(data):
    # 每个进程只初始化一次网格数组，避免随每个候选任务重复传输。
    global _OSCULATING_DATA
    _OSCULATING_DATA = data


def _boundary_component_count(edges) -> int:
    """按共享顶点计算边集合的连通分量，重复边不影响结果。"""
    if not edges:
        return 0
    graph: dict[int, set[int]] = {}
    for first, second in edges:
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
    unseen = set(graph)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbor in graph[stack.pop()] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
    return components


def _check_osculation_arrays(task, data):
    """返回状态和相切区域：0 接受，1 不连通，2 接触退化，3 多边界。"""
    position, radius, sites = task
    if not sites:
        return 1, ()

    (
        centers,
        angles,
        angle_threshold,
        epsilon,
        neighbor_offsets,
        neighbor_faces,
        neighbor_edges,
        mesh_faces,
        adjacency,
        adjacency_edges,
    ) = data
    threshold = radius * epsilon * 0.5
    region = {int(sites[0])}
    queue = [int(sites[0])]
    while queue:
        face = queue.pop()
        start, stop = neighbor_offsets[face:face + 2]
        for position_in_neighbors in range(int(start), int(stop)):
            neighbor = int(neighbor_faces[position_in_neighbors])
            edge_index = int(neighbor_edges[position_in_neighbors])
            if neighbor in region or angles[edge_index] >= angle_threshold:
                continue
            error = abs(np.linalg.norm(position - centers[neighbor]) - radius)
            if error < threshold:
                region.add(neighbor)
                queue.append(neighbor)

    if not all(int(site) in region for site in sites):
        return 1, tuple(sorted(region))

    generator_edges = []
    for face in sites:
        triangle = mesh_faces[int(face)]
        for index in range(3):
            generator_edges.append(tuple(sorted((
                int(triangle[index]),
                int(triangle[(index + 1) % 3]),
            ))))
    if _boundary_component_count(generator_edges) < 2:
        return 2, tuple(sorted(region))

    in_region = np.zeros(len(centers), dtype=bool)
    in_region[np.fromiter(region, dtype=np.int64)] = True
    crossing = in_region[adjacency[:, 0]] != in_region[adjacency[:, 1]]
    region_boundary = [
        (int(edge[0]), int(edge[1])) for edge in adjacency_edges[crossing]
    ]
    if _boundary_component_count(region_boundary) > 1:
        return 3, tuple(sorted(region))
    return 0, tuple(sorted(region))


def _check_osculation_worker(task):
    return _check_osculation_arrays(task, _OSCULATING_DATA)


class Detector:
    """在输入网格坐标系中检测；命令行入口负责提前归一化。"""
    def __init__(self, mesh: trimesh.Trimesh, parameters: DetectorParameters):
        print("Detector: initializing...")
        self.mesh = mesh
        self.parameters = parameters
        self.centers, self.normals, self.adjacency, edge_data = face_data(mesh)
        if len(edge_data):
            self.angles, self.edge_lengths = edge_data.T
        else:
            self.angles, self.edge_lengths = np.empty(0), np.empty(0)
        self.diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        self.vertices: list[VoronoiVertex] = []
        self.radius = np.zeros(len(mesh.faces))
        self.rate = np.ones(len(mesh.faces))
        self.labels = np.zeros(len(mesh.faces), dtype=np.int32)
        self.rolling_centers = np.zeros_like(self.centers)
        self.rolling_axes = np.zeros_like(self.centers)
        self.active = np.zeros(len(self.centers), dtype=bool)
        self._cached_face_graph: list[list[tuple[int, float]]] | None = None
        self.num_threads = (
            max(os.cpu_count() or 1, 1)
            if int(parameters.num_threads) == -1
            else max(int(parameters.num_threads), 1)
        )
        self._build_face_neighbors()

    def _build_face_neighbors(self):
        """Build the same face adjacency as Easy3D, without repeated scans."""
        counts = np.bincount(self.adjacency.ravel(), minlength=len(self.centers))
        self._neighbor_offsets = np.empty(len(self.centers) + 1, dtype=np.int64)
        self._neighbor_offsets[0] = 0
        np.cumsum(counts, out=self._neighbor_offsets[1:])
        self._neighbor_faces = np.empty(2 * len(self.adjacency), dtype=np.int64)
        self._neighbor_edges = np.empty(2 * len(self.adjacency), dtype=np.int64)
        cursor = self._neighbor_offsets[:-1].copy()
        for edge_index, (left, right) in enumerate(self.adjacency):
            left = int(left)
            right = int(right)
            self._neighbor_faces[cursor[left]] = right
            self._neighbor_edges[cursor[left]] = edge_index
            cursor[left] += 1
            self._neighbor_faces[cursor[right]] = left
            self._neighbor_edges[cursor[right]] = edge_index
            cursor[right] += 1

    def _osculation_data(self):
        return (
            self.centers,
            self.angles,
            np.deg2rad(self.parameters.angle_thr),
            float(self.parameters.epsilon),
            self._neighbor_offsets,
            self._neighbor_faces,
            self._neighbor_edges,
            self.mesh.faces,
            self.adjacency,
            self.mesh.face_adjacency_edges,
        )

    def generate_voronoi_vertices(self):
        num_patches = self.parameters.num_patches
        bounds = (self.mesh.bounds[0], self.mesh.bounds[1])

        if num_patches == -1:
            positions, radii, generators = voronoi3d(self.centers, bounds)
            self.vertices = [
                VoronoiVertex(position, radius, list(map(int, sites)))
                for position, radius, sites in zip(positions, radii, generators)
            ]
            print(f"Detector: global Voronoi generated {len(self.vertices)} vertices")
            return

        if num_patches < 1:
            raise ValueError("num_patches must be -1 or a positive integer")
        if num_patches > len(self.centers):
            raise ValueError(
                f"num_patches ({num_patches}) exceeds the number of mesh faces "
                f"({len(self.centers)})"
            )

        patch_centers = self._farthest_point_sampling(num_patches)
        patch_radius = (
            0.5
            * np.pi
            * self.parameters.radius_thr
            * self.diagonal
            * 1.2
        )
        unique_generators: set[tuple[int, int, int, int]] = set()
        vertices: list[VoronoiVertex] = []
        skipped_small = 0

        for patch_number, center_face in enumerate(patch_centers, start=1):
            patch = self._crop_local_patch(int(center_face), patch_radius)
            if len(patch) < 10:
                skipped_small += 1
                continue

            positions, radii, local_generators = voronoi3d(
                self.centers[patch],
                bounds,
            )
            for position, radius, local_sites in zip(
                positions,
                radii,
                local_generators,
            ):
                global_sites = tuple(
                    sorted(int(patch[int(local_face)]) for local_face in local_sites)
                )
                if global_sites in unique_generators:
                    continue
                unique_generators.add(global_sites)
                vertices.append(VoronoiVertex(position, radius, list(global_sites)))

            if patch_number == 1 or patch_number == num_patches or patch_number % 25 == 0:
                print(
                    f"Detector: processed patch {patch_number}/{num_patches}, "
                    f"unique Voronoi vertices: {len(vertices)}"
                )

        self.vertices = vertices
        print(
            f"Detector: multi-patch Voronoi generated {len(self.vertices)} "
            f"unique vertices from {num_patches} patches "
            f"({skipped_small} patches had fewer than 10 faces)"
        )

    def _face_graph(self) -> list[list[tuple[int, float]]]:
        if self._cached_face_graph is not None:
            return self._cached_face_graph
        graph: list[list[tuple[int, float]]] = [
            [] for _ in range(len(self.centers))
        ]
        angle_threshold = np.deg2rad(self.parameters.angle_thr)
        centroid_lengths = (
            np.linalg.norm(
                self.centers[self.adjacency[:, 0]]
                - self.centers[self.adjacency[:, 1]],
                axis=1,
            )
            if len(self.adjacency)
            else np.empty(0)
        )
        for edge_index, (left, right) in enumerate(self.adjacency):
            if self.angles[edge_index] >= angle_threshold:
                continue
            distance = float(centroid_lengths[edge_index])
            graph[int(left)].append((int(right), distance))
            graph[int(right)].append((int(left), distance))
        self._cached_face_graph = graph
        return graph

    @staticmethod
    def _bounded_dijkstra(
        graph: list[list[tuple[int, float]]],
        source: int,
        maximum_distance: float | None = None,
    ) -> np.ndarray:
        distances = np.full(len(graph), np.inf)
        distances[source] = 0.0
        queue = [(0.0, source)]
        while queue:
            distance, face = heapq.heappop(queue)
            if distance != distances[face]:
                continue
            for neighbor, edge_length in graph[face]:
                candidate = distance + edge_length
                if maximum_distance is not None and candidate >= maximum_distance:
                    continue
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        return distances

    def _farthest_point_sampling(self, num_samples: int) -> np.ndarray:
        graph = self._face_graph()
        selected = np.empty(num_samples, dtype=np.int64)
        minimum_distances = np.full(len(self.centers), np.inf)
        selected[0] = int(np.random.default_rng().integers(len(self.centers)))

        for sample_index in range(num_samples):
            if sample_index > 0:
                finite = np.isfinite(minimum_distances)
                if finite.any():
                    selected[sample_index] = int(np.argmax(minimum_distances))
                else:
                    remaining = np.setdiff1d(
                        np.arange(len(self.centers)),
                        selected[:sample_index],
                        assume_unique=False,
                    )
                    selected[sample_index] = int(remaining[0])
            distances = self._bounded_dijkstra(
                graph,
                int(selected[sample_index]),
            )
            minimum_distances = np.minimum(minimum_distances, distances)
            minimum_distances[selected[:sample_index + 1]] = 0.0
        return selected

    def _crop_local_patch(self, center_face: int, maximum_distance: float) -> np.ndarray:
        # Match C++ crop_local_patch: the center is queued but is not put in
        # `vis` up front; only qualifying neighboring faces are inserted.
        graph = self._face_graph()
        visited: set[int] = set()
        queue: list[tuple[float, int]] = [(0.0, center_face)]
        while queue:
            negative_distance, face = heapq.heappop(queue)
            distance = -negative_distance
            for neighbor, edge_length in graph[face]:
                candidate = distance + edge_length
                if candidate >= maximum_distance or neighbor in visited:
                    continue
                visited.add(neighbor)
                heapq.heappush(queue, (-candidate, neighbor))
        return np.asarray(sorted(visited), dtype=np.int64)

    def filter_voronoi_vertices(self):
        threshold = self.parameters.radius_thr * self.diagonal
        self.vertices = [vertex for vertex in self.vertices if vertex.radius < threshold]
        print(
            f"Detector[FILTER]: radius threshold {threshold} "
            f"remain {len(self.vertices)} vertices"
        )
        if not self.vertices:
            return
        points = np.array([np.r_[v.position, v.radius] for v in self.vertices])
        keep = self._sor(points, self.parameters.num_sor_iter)
        self.vertices = [vertex for vertex, valid in zip(self.vertices, keep) if valid]
        print(f"Detector[FILTER]: first SOR remain {len(self.vertices)} vertices")
        
        tasks = [
            (vertex.position, vertex.radius, vertex.sites)
            for vertex in self.vertices
        ]
        data = self._osculation_data()
        if self.num_threads > 1 and len(tasks) > 1:
            chunk_size = max(len(tasks) // (self.num_threads * 8), 1)
            print(
                f"Detector[FILTER]: checking osculation with "
                f"{self.num_threads} processes"
            )
            available_methods = multiprocessing.get_all_start_methods()
            process_context = multiprocessing.get_context(
                "fork" if "fork" in available_methods else "spawn"
            )
            with ProcessPoolExecutor(
                max_workers=self.num_threads,
                mp_context=process_context,
                initializer=_initialize_osculation_worker,
                initargs=(data,),
            ) as executor:
                results = executor.map(
                    _check_osculation_worker,
                    tasks,
                    chunksize=chunk_size,
                )
                counts = self._accept_osculation_results(results)
        else:
            results = (
                _check_osculation_arrays(task, data) for task in tasks
            )
            counts = self._accept_osculation_results(results)

        if self.vertices:
            points = np.array([np.r_[v.position, v.radius] for v in self.vertices])
            keep = self._sor(points, max(self.parameters.num_sor_iter // 3, 1))
            self.vertices = [vertex for vertex, valid in zip(self.vertices, keep) if valid]
            print(f"Detector[FILTER]: second SOR remain {len(self.vertices)} vertices")
        print(
            "Detector[FILTER]: osculation filtering removed "
            f"connected={counts['connected']}, tangented={counts['tangented']}, "
            f"genus={counts['genus']}; remaining={len(self.vertices)}"
        )

    def _accept_osculation_results(self, results):
        """串行与多进程共用结果归并；在进程池关闭前消费结果迭代器。"""
        accepted = []
        counts = {"connected": 0, "tangented": 0, "genus": 0}
        reasons = {
            1: "connected",
            2: "tangented",
            3: "genus",
            4: "tangented",
        }
        for vertex, (state, region) in zip(self.vertices, results):
            if state == 0:
                vertex.corr_sites = list(region)
                accepted.append(vertex)
            elif state in reasons:
                counts[reasons[state]] += 1
        self.vertices = accepted
        return counts

    def _sor(self, points: np.ndarray, iterations: int) -> np.ndarray:
        """Match the C++ statistical-outlier removal, including repeated passes."""
        labels = np.ones(len(points), dtype=bool)
        for _ in range(max(iterations, 0)):
            active = np.flatnonzero(labels)
            if len(active) < 2:
                print(
                    f"Detector: SOR terminated early with {len(active)} "
                    "active points."
                )
                break
            count = min(max(int(self.parameters.num_sor_neighbors), 1), len(active))
            distances, _ = cKDTree(points[active]).query(
                points[active],
                k=count,
                workers=self.num_threads,
            )
            if count == 1:
                distances = distances[:, None]
            # cKDTree already returns Euclidean distances. The C++ nanoflann
            # implementation stores squared distances and takes sqrt once;
            # applying sqrt here would incorrectly produce fourth-root distances.
            mean_distances = np.maximum(distances, 0.0).mean(axis=1)
            valid = mean_distances > 0
            if not np.any(valid):
                break
            cloud_mean = float(mean_distances[valid].mean())
            if valid.sum() > 1:
                std_dev = float(mean_distances[valid].std(ddof=1))
            else:
                std_dev = 0.0
            limit = cloud_mean + float(self.parameters.num_sor_std_ratio) * std_dev
            labels[active] = valid & (mean_distances < limit)
        return labels

    def _face_neighbors(self, face: int):
        start, stop = self._neighbor_offsets[face:face + 2]
        for position in range(int(start), int(stop)):
            yield (
                int(self._neighbor_faces[position]),
                int(self._neighbor_edges[position]),
            )

    def compute_density(self):
        """在四维球参数空间估计密度，在三维邻域用 SVD 估计轴向。"""
        if not self.vertices:
            return
        data = np.array([np.r_[v.position, v.radius] for v in self.vertices])
        tree = cKDTree(data)
        k = min(max(self.parameters.num_neighbors, 1), len(data))
        distances, indices = tree.query(
            data,
            k=k,
            workers=self.num_threads,
        )
        # k=1 时 SciPy 压缩邻居维度；恢复二维以保证 SVD 接收矩阵。
        distances = distances.reshape(len(data), k)
        indices = indices.reshape(len(data), k)
        # The C++ implementation stores `parameters_.sigma` in an int local.
        scale = max(int(self.parameters.sigma), 1)
        for vertex, row_dist, row_indices in zip(self.vertices, distances, indices):
            weights = np.exp(-(row_dist ** 2) / (2 * scale ** 2))
            vertex.density = float(np.average(row_dist, weights=weights))
            neighborhood = data[row_indices, :3] - data[row_indices, :3].mean(axis=0)
            _, _, vh = np.linalg.svd(neighborhood, full_matrices=False)
            vertex.axis = vh[-1]

    def compute_fields(self):
        """选择最小密度候选球，传播半径并计算相邻面的半径变化率。"""
        if not self.vertices:
            return
        angle_threshold = np.deg2rad(self.parameters.angle_thr)
        best = {}
        occurrences = np.zeros(len(self.centers), dtype=np.int64)
        for vertex_index, vertex in enumerate(self.vertices):
            corresponding = vertex.corr_sites or vertex.sites
            for face in corresponding:
                occurrences[int(face)] += 1
                if face not in best or vertex.density < best[face][0]:
                    best[face] = (vertex.density, vertex_index)
        selected = np.full(len(self.centers), -1, dtype=np.int64)
        for face, (_, vertex_index) in best.items():
            selected[int(face)] = vertex_index
        self.active = (selected >= 0) & (occurrences > 3)  # 至少四个候选支持才启用。
        active_faces = np.flatnonzero(self.active)
        if len(active_faces):
            selected_vertices = selected[active_faces]
            self.rolling_centers[active_faces] = np.array(
                [self.vertices[index].position for index in selected_vertices]
            )
            self.rolling_axes[active_faces] = np.array(
                [self.vertices[index].axis for index in selected_vertices]
            )
            initial_radius = np.array(
                [self.vertices[index].radius for index in selected_vertices]
            )
            field = np.zeros(len(self.centers), dtype=float)
            counts = np.zeros(len(self.centers), dtype=np.int64)
            active_centers = self.rolling_centers[active_faces]
            for face, center, radius in zip(
                active_faces,
                active_centers,
                initial_radius,
            ):
                # Match C++ exactly: the seed is queued but is not inserted into
                # `vis` until reached back from a qualifying neighboring face.
                region: set[int] = set()
                queue = [int(face)]
                tolerance = radius * float(self.parameters.epsilon)
                while queue:
                    current = queue.pop()
                    for neighbor, edge_index in self._face_neighbors(current):
                        if (
                            neighbor in region
                            or self.angles[edge_index] >= angle_threshold
                        ):
                            continue
                        error = abs(np.linalg.norm(self.centers[neighbor] - center) - radius)
                        if error < tolerance:
                            region.add(neighbor)
                            queue.append(neighbor)
                region_indices = np.fromiter(region, dtype=np.int64)
                field[region_indices] += np.linalg.norm(
                    self.centers[region_indices] - center,
                    axis=1,
                )
                counts[region_indices] += 1
            self.radius = np.divide(field, counts, out=np.zeros_like(field), where=counts > 0)
        else:
            self.radius.fill(0.0)
        adjacency_ok = self.angles < angle_threshold
        valid_edges = self.adjacency[adjacency_ok]
        valid_lengths = np.maximum(
            np.linalg.norm(
                self.centers[valid_edges[:, 0]]
                - self.centers[valid_edges[:, 1]],
                axis=1,
            ),
            1e-12,
        )
        rate_sum = np.zeros_like(self.radius, dtype=float)
        neighbor_count = np.zeros(len(self.radius), dtype=np.int64)

        if len(valid_edges):
            left = valid_edges[:, 0]
            right = valid_edges[:, 1]
            edge_rates = np.abs(self.radius[left] - self.radius[right]) / valid_lengths
            np.add.at(rate_sum, left, edge_rates)
            np.add.at(rate_sum, right, edge_rates)
            np.add.at(neighbor_count, left, 1)
            np.add.at(neighbor_count, right, 1)

        self.rate.fill(1.0)
        computable = (self.radius != 0) & (neighbor_count > 0)
        self.rate[computable] = np.minimum(
            rate_sum[computable] / neighbor_count[computable],
            1.0,
        )

    def graph_cut(self):
        """变化率越小越倾向标签 1；平滑项抑制非锐边两侧的标签跳变。"""
        scale = max(len(self.radius), 1) * 2.0
        unary_zero = (1.0 - self.rate) / scale
        unary_one = self.rate / scale
        valid = self.angles < np.deg2rad(self.parameters.angle_thr)
        total = max(float(self.edge_lengths.sum()), 1e-12)
        weights = np.where(
            valid,
            self.edge_lengths * self.parameters.lamdba / total,
            0.0,
        )
        self.labels = binary_cut(unary_zero, unary_one, self.adjacency, weights)

    def apply(self):
        print("Detector: applying...")
        print("Detector: generating Voronoi vertices...")
        self.generate_voronoi_vertices()
        # print("Detector: filtering Voronoi vertices...")
        # self.filter_voronoi_vertices()
        # print("Detector: computing density...")
        # self.compute_density()
        # print("Detector: computing fields...")
        # self.compute_fields()
        # print("Detector: graph cutting...")
        # self.graph_cut()
        return self

    def voronoi_cloud(self):
        points = np.array([v.position for v in self.vertices])
        return trimesh.points.PointCloud(points if len(points) else np.empty((0, 3)))

    def rolling_ball_cloud(self):
        points = self.rolling_centers[self.active]
        return trimesh.points.PointCloud(points if len(points) else np.empty((0, 3)))


def _run_direct() -> int:
    parser = argparse.ArgumentParser(
        prog="detector.py",
        description="Run DeFillet fillet detection",
    )
    parser.add_argument("-c", "--config", required=True, help="Detector JSON configuration")
    args = parser.parse_args()
    from defillet_py.cli import detect
    result = detect(args.config)
    print(f"Results written to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_direct())
