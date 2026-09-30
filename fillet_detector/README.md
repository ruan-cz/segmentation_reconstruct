# DeFillet Python port

This folder is a standalone Python implementation of the detector and remover workflows in the DeFillet C++ project. It keeps the original JSON keys and writes the same principal PLY/JSON artifacts.

## Install

```shell
cd /Users/ruan/Desktop/MeshSegment_codeonly/fillet_detector
python3 -m venv .venv
.venv/bin/python -m pip install -r defillet_py/requirements.txt
```

## Run

直接运行检测脚本，不需要使用 Python 模块运行方式：

```shell
.venv/bin/python defillet_py/detector.py -c defillet_py/detector_config.json
```

`detector.py` 会自动处理包导入路径，因此不会出现 `No module named ...`。
检测完成后，再运行去圆角流程：

```shell
.venv/bin/python defillet_py/cli.py remove -c defillet_py/removal_config.json
```

The configurations use the original `path`, `out_dir`, and `label_path` keys. For detection, `path` may be either one OBJ file or a directory; directory paths process all direct `.obj` files in sorted order and keep the configured `out_dir`. Paths should be valid on the current operating system; the Windows paths in the sample configs may need changing on macOS/Linux.

`num_patches` controls Voronoi construction exactly as in the C++ CLI:

- `-1` computes one global 3-D Voronoi diagram.
- A positive integer performs geodesic farthest-point sampling, crops that many local
  face patches, computes a Voronoi diagram for each patch, maps local generator faces
  back to global face indices, and removes duplicate four-face generators.

The detector preserves the C++ implementation's integer conversion of `sigma` and its
local-patch queue semantics. CGAL Delaunay and GCO expansion remain represented by
SciPy Delaunay and NetworkX min-cut respectively; these replacements can differ on
degenerate geometry or tied minimum cuts.

## Algorithm notes

- `scipy.spatial.Delaunay` replaces CGAL's 3-D Delaunay/Voronoi construction.
- `scipy.spatial.cKDTree` replaces nanoflann for 4-D neighborhood queries.
- NetworkX minimum cut replaces the binary graph-cut dependency.
- NumPy/SciPy sparse matrices provide the remover's constrained smoothing solve.
- The remover uses an approximate Laplacian deformation in place of the bundled Xin-Wang exact geodesic solver.

This is intentionally isolated from the C++ build, so the original implementation remains available for result comparison.
