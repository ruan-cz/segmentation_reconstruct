"""检测和去圆角的命令行入口，负责配置、坐标预处理与结果导出。"""

from __future__ import annotations
import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "defillet_py"

from .config import DetectorParameters, RemoverParameters, load_config, save_config
from .detector import Detector
from .mesh import load_mesh, normalize, save_field, save_json, save_labels, save_point_cloud
from .remover import Remover


def _result_dir(out_dir, mesh_path, mode):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    result = path / f"{Path(mesh_path).stem}_{mode}_{datetime.now():%Y%m%d_%H%M%S}"
    result.mkdir()
    return result


def _detect_one(params):
    mesh = load_mesh(params.path)
    # 检测结果使用归一化坐标，radius_thr 相对于归一化后的包围盒对角线。
    normalize(mesh)
    mesh.merge_vertices()
    detector = Detector(mesh, params).apply()
    result = _result_dir(params.out_dir, params.path, "detector")
    base = Path(params.path).stem
    voronoi_normals = None
    if detector.vertices and all(vertex.axis is not None for vertex in detector.vertices):
        voronoi_normals = np.asarray(
            [vertex.axis for vertex in detector.vertices],
            dtype=float,
        )
    save_point_cloud(
        detector.voronoi_cloud(),
        result / f"{base}_voronoi_vertices.ply",
        voronoi_normals,
    )
    save_point_cloud(
        detector.rolling_ball_cloud(),
        result / f"{base}_rolling_ball_centers.ply",
        detector.rolling_axes[detector.active],
    )
    save_field(mesh, detector.radius, result / f"{base}_radius_field.ply")
    save_field(mesh, detector.rate, result / f"{base}_rate_field.ply")
    save_labels(mesh, detector.labels, result / f"{base}_seg.ply")
    save_config(params, result / f"{base}_param.json")
    save_json(detector.labels.tolist(), result / f"{base}_seg.json")
    return result


def detect(config_path):
    params = load_config(config_path, DetectorParameters)
    input_path = Path(params.path)
    if input_path.is_dir():
        mesh_paths = sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".obj"
        )
        if not mesh_paths:
            raise FileNotFoundError(
                f"No .obj files found in detector path: {input_path}"
            )
        results = []
        for mesh_path in mesh_paths:
            results.append(_detect_one(replace(params, path=str(mesh_path))))
        return results
    return _detect_one(params)


def remove(config_path):
    params = load_config(config_path, RemoverParameters)
    mesh = load_mesh(params.path)
    labels = json.loads(Path(params.label_path).read_text(encoding="utf-8"))
    remover = Remover(mesh, params, labels).optimize()
    result = _result_dir(params.out_dir, params.path, "removal")
    base = Path(params.path).stem
    outputs = {
        "defillet": remover.defillet_mesh,
        "focus": remover.focus_mesh,
        "fillet": remover.fillet_mesh,
        "non_fillet": remover.non_fillet_mesh,
    }
    for name, value in outputs.items():
        value.export(result / f"{base}_{name}.ply")
    save_config(params, result / f"{base}_param.json")
    return result


def main():
    parser = argparse.ArgumentParser(prog="defillet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("detect", "remove"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    result = detect(args.config) if args.command == "detect" else remove(args.config)
    print(f"Results written to: {result}")


if __name__ == "__main__":
    main()
