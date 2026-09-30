# MeshSegment CSG 重建技术文档

> 本文档描述当前 CSG 重建部分的完整技术流程：从分割后的三角网格（segmented PLY）出发，到最终生成 OpenSCAD 文件的每一步所采用的算法与技术。
> 依据 2026-09 当前代码核实整理（`csg/graph_construct.py`、`primitive_fitting.py`、`fitting_geometric_primitives/geometry_primitive.py`、`csg/csg_core/`、`csg/csg_reconstruction.py`、`csg_process.ipynb`）。
> 较旧的 `CSG_RECONSTRUCTION_PROJECT_SUMMARY.md` 未覆盖 spline extrusion、transition patch 等较新功能，本文档为准。

---

## 1. 总体流程

```text
segmented PLY（面片带 RGB segment 颜色）
  │
  ├─ [Step 1] 网格读取与 segment patch 提取        csg/graph_construct.py
  ├─ [Step 2] 单 patch primitive 初步拟合           primitive_fitting.py + geometry_primitive.py
  ├─ [Step 3] patch 邻接图构造                      csg/graph_construct.py
  │
  ├─ [Step 4] primitive candidate 生成              csg_core/candidates.py
  │     ├─ transition patch 分类（fillet/chamfer）
  │     ├─ cube candidates
  │     ├─ polygon extrusion candidates
  │     ├─ spline extrusion candidates
  │     ├─ curved candidates（cylinder/cone/sphere/torus）
  │     └─ transition curved candidates（clip 后的 cone/torus）
  │
  ├─ [Step 5] ADD / SUBTRACT 分类                   csg_core/boolean.py
  ├─ [Step 6] Boolean hypotheses + beam search 选择  csg_core/boolean.py
  ├─ [Step 7] INTERSECTION 关系推断 + CSG IR 构建    csg_core/boolean.py
  │
  ├─ [Step 8] 几何验证（surface + voxel）            csg_core/validation.py
  └─ [Step 9] OpenSCAD 代码生成                      csg_core/openscad.py
```

支持的 primitive 类型：`CUBE`（含旋转 cube）、`CYLINDER`、`CONE`、`SPHERE`、`TORUS`、`EXTRUSION`（多边形线性拉伸）、`SPLINE_EXTRUSION`（B 样条轮廓拉伸）。
支持的 Boolean 节点：`UNION`、`DIFFERENCE`、`INTERSECTION`、`EMPTY`。

整个流程不依赖神经网络，全部基于解析几何与图算法：最小二乘拟合、协方差特征分解、图分割/区域生长、包含性查询、beam search、体素近似验证。

---

## 2. Step 1：网格读取与 segment patch 提取

**文件：`csg/graph_construct.py`**

### 2.1 网格读取

- 入口 `_load_triangle_mesh()`，支持文件路径或已有 `trimesh.Trimesh` 对象。
- 使用 `trimesh.load_mesh(path, process=False)`，**刻意关闭 trimesh 的自动处理**，以保留原始 vertex/face 索引——后续每个 patch 必须用 `face_id` 回指输入模型。
- 若读取结果为 `trimesh.Scene`，用 `trimesh.util.concatenate` 合并其中所有 geometry。
- 强制校验输入必须是三角网格。

### 2.2 按 face color 提取 segment

- `_face_labels()` 读取 `mesh.visual.face_colors[:, :3]`，用 `np.unique(labels, axis=0)` 按唯一 RGB 值对面片分组，每组即一个 segment。
- `_extract_segment_patches()` 对每组面片收集全局 vertex ID，用 `np.unique(..., return_inverse=True)` 建立 patch 局部子网格（默认 `merge_vertices=False`，避免无意的拓扑改动）。
- 每个 patch 的 `metadata` 保存：
  - `patch_index`：算法内部使用的稳定顺序 ID（不依赖 RGB label 数值）；
  - `segment_label`：RGB 颜色，仅作辅助信息；
  - `face_id` / `source_face_ids`：该 patch 在输入 mesh 中的原始面片索引。
- `min_faces` 参数可过滤过小 segment。

---

## 3. Step 2：单 patch primitive 初步拟合

**文件：`csg/graph_construct.py`（`fit_patch_primitives` / `identify_patches`）、`primitive_fitting.py`、`fitting_geometric_primitives/geometry_primitive.py`**

这一步回答的问题是："单个 patch 是否接近某种解析曲面？"——注意它与后续 candidate 阶段"哪些 patch 应组合成同一个完整实体"是两层不同职责。

### 3.1 PrimitiveFitting 预计算

`primitive_fitting.PrimitiveFitting.__init__()` 基于 **libigl** 预计算：

- `igl.per_vertex_normals`：顶点法向；
- `igl.doublearea / 2.0`：面片面积；
- `igl.per_face_normals`：面片法向（零长度法向用 `np.divide(..., where=...)` 安全归一化，避免退化面导致崩溃）；
- 面片中心由 `np.mean(v[f], axis=1)` 计算。

### 3.2 各类型底层拟合器

有效 patch（有面片、顶点 ≥ 3、存在非退化面）会依次调用 6 种 fitter，**每种结果都存入 `fits` 字典**而不只保留最优者；某类拟合抛异常或返回非有限值时，仅将该类 `fit_rate` 置 0 并记录 `error`，不中断整体流程。退化 patch 直接返回 `type="invalid"`。

| 拟合类型 | 底层实现 | 核心技术 | inlier 阈值 |
|---|---|---|---|
| plane | `Plane.fit` | 协方差 PCA（`np.linalg.eig`）求法向，参数 `[a,b,c,d]` | 点到平面距离 < 1e-2 |
| cylinder | `Cylinder.fit` | 轴假设来自点集 PCA 最小特征向量 + 法向外积矩阵 `Σnnᵀ` 的 `eigh` 最小特征向量；投影到 2D 后用 **Taubin 代数圆拟合**（`scipy.linalg.eig` 广义特征值） | `|dist − r| < 1%·r` |
| sphere | `Sphere.fit` | <100 点直接 `lstsq`；≥100 点用 **RANSAC**（100 次迭代、每次采 10% 点） | 相对阈值 `5e-2·r`；半径 > 2×bbox 对角线判失败 |
| cone | `Cone.fit_on_all_points` | 先用法向方程 `n·(p−apex)=0` 对所有点 `lstsq` 解顶点；再对 (axis, θ) 做 `scipy.optimize.minimize` **SLSQP**（约束 ‖axis‖=1） | 1e-2 |
| extrusion | `Extrusion.fit` | 将面法向拟合到 Gauss 球面上的一个平面（协方差 eig），要求覆盖率 > 0.9，返回 3 维轴向量 | 5e-2 |
| spline_extrusion | 复用 `Extrusion.fit` | 当前与 extrusion 同实现；真实样条轮廓在 candidate 阶段由边界环恢复 | 同 extrusion |

注：`PrimitiveFitting.fit_torus` 目前是空壳，torus 完全在 candidate 生成阶段处理。

最终 patch 的 `type` 由 `_select_best_fit()` 决定：先取 fit rate 最大者；若多个类型 fit rate 持平（≈1.0），按 **RMS 残差**（`_fit_residual_errors()`，按 patch 尺度归一）打破平局——只有残差显著更小（< 0.5×）的类型才能胜出，干净平面因此保持 plane 类型。这对薄环面（螺纹）至关重要：锥面/柱面薄环由于绝对 inlier 阈值宽松都能拿到 plane fit 1.0，只有残差能区分锥面（cone 残差≈0）与平面近似（残差≈环高的一半）。由于 `fits` 保留了全部结果，candidate 阶段仍可读取其他类型作为 seed。

### 3.3 sphere 半径后处理

`prepare_patch_graph()` 末尾调用 `_enforce_model_fit_bounds()`：sphere 拟合半径超过 `maximum_sphere_radius_ratio`（默认 1.0）× 模型 bbox 对角线时，将该 sphere fit 置 0、记录 `rejected_reason="radius_exceeds_model_scale"`，并在原最优类型为 sphere 时改选次优。这防止近似平面的 patch 被拟合成虚假大球。

---

## 4. Step 3：patch 邻接图构造

**文件：`csg/graph_construct.py`（`_build_patch_adjacency`）**

1. `_face_patch_map()` 建立原始面片 → patch 的映射，并校验各 patch 的 `face_id` 不重叠；
2. `_mesh_edge_faces()` 枚举每个三角形的三条无向边，建立 edge → incident faces 映射；
3. 一条边的两侧面片属于不同 patch 时，形成一条 graph edge，记录：
   - `v` / `boundary_v`：原始网格上的两个端点坐标；
   - `length`：边长；
   - `dihedral_angle`：分别取两个 patch 在该边邻域内**所有邻接面法向的均值**（`_patch_normal_on_edge`），计算 `arccos(clip(|dot(n0, n1)|))`，范围 [0, π/2]——取绝对值使其与面片 winding 方向无关；
4. 同时更新两侧 node 的 `boundary_edges` 与 `neighboring_patches`。

一体化入口 `prepare_patch_graph()` 顺序执行：读取网格 → 提取 patch → 初步拟合 → 构造邻接 → 合并拟合结果与图记录，返回 `(mesh, patches, results, patch_graph)`。这是 notebook 与测试的标准预处理入口。

---

## 5. Step 4：primitive candidate 生成

**文件：`csg/csg_core/candidates.py`**

统一入口 `generate_primitive_candidates(results, graph, include_torus=True, min_seed_rate=0.55, include_spline_extrusion=True)`（类接口 `PrimitiveCandidateGenerator`），执行顺序：

```text
classify_transition_patches          # 标记圆角/倒角 patch
→ generate_cube_candidates
→ generate_extrusion_candidates
→ generate_spline_extrusion_candidates
→ generate_curved_candidates         # CYLINDER / CONE / SPHERE [/ TORUS]
→ generate_transition_curved_candidates
→ generate_planar_notch_cutters      # 圆柱弦切面盒形刀具
→ generate_pocket_floor_cutters      # 凹槽围出的中心柱刀具
→ 按 (primitive_type, patch_ids, transition) 去重
→ 按 confidence 降序、fitting error 升序排序
```

输出统一为 `PrimitiveCandidate` dataclass：`primitive_type / parameters / patch_ids / fitting_error / confidence / operation / metadata`。

### 5.1 transition patch 分类（圆角/倒角）

`classify_transition_patches()` 在生成主 candidate 前先标记过渡面：

- `FILLET_OR_CHAMFER`：低质量 sphere patch（径向方向协方差最小/最大特征值之比 < 0.03，即 surface quality 不足的条带）；
- `CHAMFER`：小 plane patch（面积 ≤ 0.35 × 邻居面积中位数，且有曲面邻居）。

这些 patch 不参与标准 candidate，而由 5.6 的 transition curved candidate 单独处理。

### 5.2 Cube 重建

Cube 重建**不是**寻找三对相对面、也不要求六个面齐全。核心流程：

```text
plane 法向聚类 → 局部正交坐标系假设 → plane 表面转换到局部坐标
→ 图兼容 region growing → 几何约束校验 → 不相交图划分
→ 从 region bounds 恢复 cube 几何
```

**局部坐标系推断（`_infer_cube_coordinate_systems`）**：

- 用法向面积加权贪心聚类（容差 12°）得到主方向簇；
- identity frame 始终作为 axis-aligned 假设参与评分（保证 axis-aligned 情形不被推翻）；
- 从两组近似正交法向构建前两轴，**Gram-Schmidt 正交化**第二轴，叉积得第三轴，det < 0 时翻转以保证右手系；
- 此外还从**原始法向对**直接构造局部 frame，要求 inlier 面积占比 ≥ 0.12（服务于多旋转块装配体），最多保留 `max_frames=8` 个 frame；
- 按所有 plane 法向与三轴的对齐程度（面积加权）计算 frame score，`_rotation_equivalent`（容差 3°）去除旋转等价 frame。

**plane 表面转换（`_local_plane_surfaces`）**：每个 plane patch 转到候选 frame，记录所属 axis family、支撑位置、局部 AABB、normal error、面积。法向用绝对对齐值，对输入 winding 稳健。

**兼容性检查（`_cube_patch_compatibility`）**：

- 必须有非零公共边界长度；
- 同方向 patch 须接近同一支撑平面（可合并，处理一个 cube 面被分割成多个 patch 的情形）；
- 不同方向 patch 须在剩余公共轴上有切向重叠；
- 二面角接近 90° 是**软性 score 惩罚**而非硬约束：`score = 2 + log1p(boundary_length) − 0.5·angle_error`。

**有效 region 校验（`_valid_cube_region`）**：region 非空、三个局部尺度非退化、每个 axis family 聚类后支撑位置 ≤ 2、每个 patch 位置必须贴近 region 包围盒边界。这允许 3/4/5 面可见的不完整 cube，同时排除"同一轴三个支撑面"或"patch 位于盒内部"等错误扩张。

**不相交图划分（`_partition_cube_surfaces`）**：对每个可用 patch 作为 seed 做 region growing，只从图邻居中加入兼容 patch，每次加入前做增量几何校验；已有 ≥2 个方向后，限制同方向 patch 对切向范围的无依据扩张；比较不同 seed 结果，优先选择 node 多且 `_cube_region_score`（综合 node 数、表面积、内部共享边界长、normal error）高者；确定后从候选集移除，对剩余图重复。输出的 partition 是严格的——不同 cube region 不共享 patch。此外，**落败的 seed trial 也作为备选 region 输出**（跳过完全被胜者覆盖的 trial）：当共面 patch 被更大的相邻 region 吸走导致正确 region 尺寸落败时（例如横梁的平齐侧面同时属于相接的斜臂），正确结构仍能以候选身份进入全局 beam search 裁决，而不是被贪心的尺寸排序永久淘汰。

**几何恢复（`_cube_candidate`）**：从 region 在局部 frame 的总 bounds 恢复 center/size/rotation；Euler 角由 `_matrix_to_euler_xyz`（arcsin/arctan2）给出，供 OpenSCAD 使用。`fitting_error = 平面到盒边界距离误差 + normal error`，`confidence = exp(−8·err) · frame_score · min(1, len/3)`。其他参数：`min_patch_count=3`、plane seed `fit_rate ≥ 0.8`、要求覆盖至少 2 个 axis family。

### 5.3 多边形拉伸（EXTRUSION）重建

处理 cube/cylinder 等无法表达的"多边形平面拉伸体"。

1. 选取高质量 plane patch（fit_rate ≥ 0.8），按 patch 图做 **BFS 连通分量拆分**；分量 < 3 个 patch 拒绝；
2. **多轴假设**（`_extrusion_axis_hypotheses`）：一个 plane 连通分量可能同时包含多个不同轴的棱柱（例如凸台侧壁与槽壁通过 cube 面连通），因此不再只取全局最优轴。轴假设来源为法向协方差**最小特征向量**（`eigh`）+ 任意两非平行法向的叉积，按（法向多样性，侧面积数，−残差）评分并按 12° 聚类去重，返回所有受支持的轴；`_extrusion_axis_from_normals` 是取最优轴的兼容包装；
3. **侧面带拆分**：对每个轴取法向 ⊥ 轴（< 15°）的 patch，再按 patch 图拆成连通侧面带（≥3 个 patch），每条带单独拟合——同一个轴向上互不连通的棱柱（凸台环与槽）各自成 candidate；
4. **轮廓恢复**（`_fit_extrusion_group`）：建立垂直于轴的 2D 正交基，投影侧面顶点到轮廓平面，用 `scipy.spatial.ConvexHull` 求二维凸轮廓，`_simplify_polygon` 删除近共线顶点（容差 1e-4×scale）；全部侧顶点投影到轴上取 `[min, max]` 得 extent；
5. **侧面-轮廓一致性**：迭代剔除顶点不落在轮廓边界上的"侧面"（>50% 顶点距边界 ≤ 2%·profile_scale 才保留），并重新计算 hull——防止"凸台所附着的 cube 面"这类位于 hull 内部的平面被当作棱柱侧壁而撑大轮廓；若**超过一半的原始侧面被剔除**则整体拒绝（螺纹这类"轮廓投影恰好是 screw 剪影"的情形，只有剪影顶/底边缘的 patch 会幸存，并非真实棱柱）；
6. **端盖识别**：法向与轴平行（cos ≥ 12°）、顶点贴近端面（≤ 3%·height）、且投影在轮廓内（`_point_in_polygon` ray casting + 近边界容许）的 patch 记为 `cap_patch_ids`；
7. **端盖对驱动棱柱**（`_cap_pair_extrusion_candidates`）：槽/凹槽常只露出两个端面加少量侧壁，侧面驱动路径（≥3 侧面）无法触发。两枚**平行（≤20°，轴取两法向符号对齐后的均值以抵消拔模）、面积相近（≥0.5×）、互不相邻、轴向站位不同、投影轮廓重叠 ≥50%、且有共同平面侧壁相邻**的 plane patch 定义一个棱柱：端盖轮廓即 profile，端盖对为 caps。侧面同样须经轮廓边界一致性过滤，轴残差同样受 `maximum_error` 门槛限制；
8. **防与 cube 重复**：矩形轮廓（原始 hull 顶点数 = 4）且关联 patch 面积占全部 plane patch 面积 > 0.8 时拒绝；此外在 `generate_primitive_candidates` 汇总后，由 `_drop_cube_covered_rectangles` 丢弃：与某个 cube candidate 的 patch 面积重叠 ≥ 0.9 的矩形 extrusion，以及**表面约一半以上落在 cube 面上（≥0.45）的端盖驱动棱柱**——大矩形实体统一交还给 cube 重建；
9. **质量门槛**：`maximum_error=0.05`，轴残差（法向与轴的平均 |dot|）超过即拒绝——剔除把"法兰顶面 + 倒角面"混作侧壁的伪棱柱；其他拒绝规则：最大侧面积占比 > 0.95 拒绝；侧面数 ≥ 10 且非矩形且 hull 顶点数 > 8 拒绝（防止吞并多特征装配体）；拟合中的 `scipy.spatial.QhullError`（退化共面点集）按失败处理。

**当前限制**：轮廓由凸包恢复，凹多边形会被填平，需要后续从边界环恢复有序轮廓。

### 5.4 样条拉伸（SPLINE_EXTRUSION）重建

针对侧表面为自由曲面的拉伸体：

- 选取 `SPLINE_EXTRUSION` 拟合率 ≥ 0.82 且解析曲面拟合率 < 0.82 的 patch 作为侧面种子；
- 按拉伸轴（8° 容差）聚类分组；
- **平直侧面段吸收**（`_grow_planar_spline_sides`）：自由轮廓中的直线段会被分割成 plane patch 而被种子规则排除；将满足 plane 拟合 ≥ 0.8、spline 拟合 ≥ 0.82、法向 ⊥ 轴（< 15°）、且轴向范围不超出组 extent ±10% 的相邻平面 patch 迭代并入侧面组，保证轮廓完整；
- `_mesh_boundary_loops` 提取 patch 边界环；`_spline_profile_loop` 只接受位于 extent 端面附近、且**二维跨度不超过侧面组跨度 1.3 倍**的环（否则退回任意环）——防止选中"凸台所附着的整面 cube 面"的外轮廓；
- `_fit_periodic_bspline` 用 `scipy.interpolate.splprep`（`per=True`，平滑量 = n·(2e-3·scale)²）拟合**周期 B 样条轮廓**，并输出拉伸曲面的控制网参数；
- **cap 轮廓包含校验**：cap patch 顶点投影到轮廓平面后，须 ≥90% 位于样条轮廓内或贴近边界（≤ 2%·profile_scale），否则从 `cap_patch_ids` 剔除——防止整面支撑面被吞为 cap 并与 cube candidate 冲突。

### 5.5 曲面 primitive 拟合（cylinder / cone / sphere / torus）

`generate_curved_candidates()`（`min_seed_rate=0.55`、`maximum_error=0.04`）先对兼容 patch 分组，再对每组做整体非线性拟合。

**采样（`_sample_patch_points`）**：按稳定 `patch_index` 合并多 patch 的顶点与顶点法向，均匀下采样限点数，可选返回每个采样点的来源 patch ID（extrusion 侧/端面法向分类依赖它）。

**分组（`_curved_patch_groups`）**：

- CYLINDER / SPHERE 用 **union-find 全对合并**，支持**不相邻**但参数一致的 patch 合并（处理解析曲面被 Boolean 切成多个不连通可见区域的情形）；
- CONE / TORUS 仅沿图邻边扩张，torus 额外生成"边对"小假设组；
- 配对容差：sphere 中心/半径差 ≤ 0.15r；cylinder 轴 12°、半径差 ≤ 4%、轴向区间间隙 ≤ max(0.005·scale, 0.10·r)——计算投影前先将两条 seed 轴的**符号对齐**，否则两段相距很远的同轴段可能因轴符号相反而虚报零间隙；cone 轴 15°、角差 8°、apex 距 ≤ 0.1·scale。

**各类型拟合技术**（全部以 `scipy.optimize.least_squares` 做联合非线性优化，残差按模型尺度归一化为 RMS）：

| 类型 | 初始化 | 优化变量 | 备注 |
|---|---|---|---|
| cylinder | 法向矩阵最小特征向量定轴；2D 投影 + 线性最小二乘拟合圆 | axis point、axis、radius（7 参数，max_nfev=400） | 点投影到轴得有限 extent |
| sphere | 线性 `lstsq` 解球心半径 | center、radius（max_nfev=300） | 见下方过滤 |
| cone | 法向切平面方程 `lstsq` 估计 apex；点协方差特征向量 × 两符号作轴假设；radial/axial 中位数 arctan 定初始角（clip [1°, 80°]） | apex、axis、angle（max_nfev=500，角界 [0.5°, 85°]） | 对 apex 反方向的点加 penalty |
| torus | 点协方差 + 法向协方差共 6 个轴假设；轴向/径向分解初始化 major/minor | center、axis、major/minor radius（max_nfev=600） | 要求 major > 1.05·minor 过滤退化 |

**Sphere 过滤**（两道）：

1. `_valid_sphere_fit`：半径有限为正、半径 ≤ 模型尺度、球心距模型中心 ≤ 2×模型尺度、bbox 与模型重叠——剔除近平面产生的超大半径假球；
2. `_sphere_surface_quality ≥ 0.03`：径向方向协方差最小/最大特征值之比，剔除 fillet 条带假球。**挽救规则**：浅球冠（如螺母球顶）的方向散布与 fillet 条带同样低，但其 torus 拟合会退化为 major_radius ≈ 0——若同一组点上不存在误差更小的有效 torus（`major > 1.05·minor`），则保留为球体，并用 `_sphere_cap_clip_bounds()` 加 `clip_bounds`：**只做轴向裁剪**——侧向球面本身跟随球顶轮廓，AABB 的方形截面会在球体并入圆台的区域灌入角部假体积；轴向边界延伸到"完全位于球冠下沿的体侧邻居 patch"（轮缘等）的位置，而不是切在球冠 patch 的下沿，使球顶实体并入下方主体。

**Cylinder/Cone 端盖附着**（`_attach_planar_caps`）：候选生成后对每个 cylinder/cone 候选收集平面端盖请求——要求 patch 自身类型为 plane、与候选侧面图相邻、法向与轴平行、顶点贴近 extent 端面且径向不超出该端半径；同一 patch 被多个候选请求时全部放弃（例如"立柱顶面与开孔端面"这类共享面保持未分配），无冲突的才并入 `patch_ids`。这使螺母底面、螺杆尖端圆盘等封闭面获得覆盖而不污染候选归属。

此外，若单 patch sphere 假设已被误差小于其一半的多 patch torus candidate 解释，则删除该 sphere。

`confidence = exp(−12·err) · mean(max(seed_rate, 0.25))`。

### 5.6 transition curved candidate（圆角/倒角母面）

`generate_transition_curved_candidates()` 对 5.1 标记的 transition patch 拟合**带裁剪边界（clip_bounds）**的 cone/torus 母面：

- `clip_bounds = patch bbox + 1e-4·scale padding`，使 candidate 只覆盖过渡区域而不无限延伸；
- cone 角度限 [3°, 78°]，torus 要求 major > 1.1·minor；
- `confidence = 0.25·exp(−10·err)`（低于标准 candidate，作为补充解释）。

### 5.7 圆柱弦切缺口刀具（planar notch cutters）

`generate_planar_notch_cutters()` 处理"圆柱被平行于其轴的平面削去一段弦"的缺口（D 形平面/盲槽）。这类缺口只露出一个小矩形面（有时加槽底面），现有生成器都无法触发：cube 需要 ≥3 个面且端盖面越界会被 `_valid_cube_region` 拒绝、extrusion 侧面驱动需要 ≥3 个侧面、端盖对驱动要求两枚全等端面。改为从圆柱参数与弦平面**解析构造盒形刀具**：

对每个 cylinder 候选，检查其侧面 patch 的图相邻 plane patch，满足全部条件才生成：

- 平面法向与圆柱轴平行（|n·axis| ≤ sin10°）；
- 平面到轴线的径向距离 < radius（是弦而非切线/外面）；
- patch 全部顶点位于圆柱内（径向 ≤ radius + 容差）且轴向不超出 cylinder extent（±5%）——排除"圆柱贴靠的大基体面"；
- patch 面积 ≤ 0.5 × 圆柱侧面积（缺口面远小于其所切削的柱面）。

刀具盒子：局部坐标系取 x' = axis × 弦面径向、y' = 弦面径向、z' = axis；y' 范围 [弦距, radius + 5%·r]（与圆柱求交恰好得到弦缺 segment）、x' 范围 [−√(r²−d²), +√(r²−d²)]、z' 取缺口面轴向范围（若抵达圆柱端面则延伸穿过）。相邻且被盒子包含（宽容差）的小 plane patch（如盲端槽底）一并并入 `patch_ids`。分类阶段由弦面法向反向 + 被圆柱完全包含判为 SUBTRACT。

### 5.8 凹槽围出的中心 pocket 刀具（pocket floor cutters）

`generate_pocket_floor_cutters()` 处理十字槽类凹陷的中心柱体：N 条槽臂棱柱（端盖对驱动棱柱，见 5.3）只覆盖槽臂区域，中心最深处的方柱（底面 patch + 四周槽壁内侧平面）不在任何棱柱轮廓内。检测：一个 plane patch（槽底）同时与 ≥2 个 extrusion 候选的侧面相邻、且这些棱柱的轴都垂直于槽底法向，则以槽底平面为底、各棱柱 cap 壁的内侧包络为侧界、棱柱顶部为顶，构造中心 cube 刀具。安全守卫：槽壁顶点必须（≥90%）位于槽底平面的同侧（空缺在槽底的一侧）；cap 壁包络必须与槽底 patch 的自身横向范围紧密一致（±30%）——防止普通大平面被误判为槽底。

### 5.9 cube 的边界平面截断（boundary-plane truncation）

`_truncate_cubes_at_boundary_planes()` 是候选生成后的后处理 pass：旋转条/臂类 cube 的可见面在相邻体的某面处齐平终止（例如穿过横梁的臂板与横梁底面齐平），但从 patch 恢复的盒子是完整长方体，其倾斜角部会越过该平面漏入空气。对每个 cube 候选，检查其成员 patch 的图相邻 plane patch 所 evidencing 的平面：

- 平面须世界轴对齐（|n| 主导分量 ≥ cos2°，clip_bounds 是 AABB 只能表达轴对齐裁剪）；
- 平面严格穿过 cube 的 AABB（在边界上的平面是 cube 自己的面，不处理）；
- cube 全部成员 patch 的顶点（容差外）**全部位于平面同侧**——若两侧都有表面证据，说明盒子合法地越过该平面（如插入另一体的 union 重叠），不裁剪。

满足则把 `clip_bounds`（初值为 cube 自身 AABB）在该轴向上收紧到平面位置。CUBE 的 `primitive_contains`/`primitive_bounds`/OpenSCAD 输出（`intersection() { cube; clip_box; }`）均支持 `clip_bounds`，与 transition 的裁剪共用 `_clip_shape`。

### 5.10 通用几何查询

供 Boolean、validation、OpenSCAD 复用：

- `primitive_bounds(candidate)`：world AABB。cube 用 `abs(rotation) @ half_size`；cylinder/cone 用轴端点加径向范围；torus 用 `major+minor` 保守包围；extrusion 把 2D polygon 放回 world 基再加轴向 extent；支持 `clip_bounds` 裁剪。
- `_primitive_normals(candidate, points, point_patch_ids)`：按参数预测表面法向（cube 取最近局部面、sphere/cylinder/torus 径向、cone 径向与轴角组合、extrusion 取最近轮廓边 2D 法向、端面用 ±axis），用于判断观测法向朝向实体外部还是空腔内部。
- `primitive_contains(operation, parameters, points, tolerance)`：点包含性查询，是 ADD/SUBTRACT 分类、cutter 支持检查、INTERSECTION 推断、体素验证的共同基础；extrusion/spline extrusion 用轴向 extent 测试 + `_point_in_polygon` 二维 ray casting。

---

## 6. Step 5：ADD / SUBTRACT 分类

**文件：`csg/csg_core/boolean.py`（`classify_boolean_operations`）**

分类是**单 candidate 的一元属性**，综合三类证据而非只看法向：

1. **Orientation score**：对 candidate 对应 patch 采样 2500 点，计算 `mean(观测法向 · 预测法向)`。接近 +1 支持 ADD；接近 −1 提示 cavity/cutter 表面；接近 0 证据弱。
2. **Containment coverage**：对所有不共享 patch、orientation 不明显反向（> −0.2）、AABB 相交的其他 candidate，用 `primitive_contains`（容差 1e-4×模型尺度）检查当前 candidate 的表面采样点被覆盖的比例。
3. **Through-hole heuristic**：candidate 在 ≥2 个轴上严格位于容器内部、且在另一个轴上同时接触容器两侧边界时，判为穿孔 cutter 几何特征。

判定规则：`containment_coverage ≥ 0.9` 且（orientation < −0.2 或 through-hole）→ SUBTRACT（score += 0.5 + through_hole）；否则倾向 ADD。要求较高覆盖率是因为输入 winding 可能翻转、开放表面无可靠内外、拟合噪声会导致法向不一致。分类结果与各项分数写入 candidate metadata（`orientation_score / containment_coverage / through_hole_score / operation_scores / operation_confidence` 等），便于 notebook 调试。

---

## 7. Step 6：Boolean hypotheses 与 beam search 选择

**文件：`csg/csg_core/boolean.py`**

### 7.1 Hypotheses 生成

`generate_boolean_hypotheses()`（`relative_score_threshold=0.35`）：若次优 operation 的分数与最优足够接近，则同一 candidate 的 ADD/SUBTRACT 两种拷贝都进入搜索——分类阶段不永久锁死，给结构搜索保留纠错能力；次优拷贝的 confidence 按 `0.5 + 0.5·score/max` 打折。

### 7.2 追加约束

`_can_append_candidate()`：

- 新 candidate 不得与已选 candidate 共享 patch；
- ADD 可直接加入；
- SUBTRACT 必须被当前已选 ADD 几何支持：采样点被 ADD 覆盖 ≥ 0.9，或 cutter bbox 被某 ADD bbox 完整包含。

这防止"减去一个与主体完全不相交的大球/柱"这类不改变几何但结构错误的结果。

### 7.3 选择评分

```text
score = patch_coverage（按 patch 面积加权）
      + 0.15 · mean_confidence
      − 0.40 · 面积加权 fitting_error
      − complexity_penalty · primitive_count
```

### 7.4 Beam search

`reconstruct_csg_tree()`（`beam_width=12`、`maximum_primitives=64`、`complexity_penalty=0.025`）：

1. 按 `covered_area · confidence` 排序 hypotheses；
2. 从空选择出发，逐层向现有状态追加一个兼容 candidate；
3. 对等价 candidate ID 集合去重；
4. 按 (coverage, score) 保留最优的 beam_width 个状态；
5. 覆盖率达 1 − 1e-9 或达到最大 primitive 数时停止；
6. 对最优选择调用 `build_csg_ir()`。

---

## 8. Step 7：INTERSECTION 推断与 CSG IR 构建

**文件：`csg/csg_core/boolean.py`**

INTERSECTION 不是单 candidate 的属性，而是**多个 ADD candidate 之间的关系**。

### 8.1 配对兼容性（`_intersection_pair_score`）

两个 ADD candidate 需同时满足：

1. 不共享 patch；
2. world AABB 存在真实正体积重叠；
3. 两组可见表面 patch 沿输入模型的实际边界相接——用 `scipy.spatial.cKDTree.query_ball_point` 检测共享边界顶点，**至少 2 个**（避免两个不相邻、仅拟合体重叠的 patch 被误判，例如两个独立球面 patch）；
4. 双向表面覆盖：A 的可见采样点大部分在 B 内、且 B 的大部分在 A 内，默认阈值 `intersection_coverage_threshold=0.85`。

### 8.2 多体 intersection（`infer_intersection_groups`）

不使用简单连通分量（A∼B、B∼C 不蕴含 A∼C），而采用 **clique 式扩张**：贪心选最优 pair 起组，新成员必须与组内**所有**已有成员两两兼容才能加入。

### 8.3 IR 构建（`build_csg_ir`）

1. 已选 candidate 分为 additive / subtractive；
2. additive 中推断 intersection groups，`_build_additive_ir` 将普通 singleton 与 intersection group 组成 base（组内 INTERSECTION、组间 UNION）；
3. transition candidate（clip 的 torus/cone）与其支撑 base primitive 组成 INTERSECTION 项并入 base——但若 clip 后的 transition 与支撑的包围盒**无正体积重叠**（例如恰好卡在两个 base 之间的轮缘带），求交会将其抹除，此时改为独立 ADD 项；
4. 多个 cutter 先组成 UNION，再生成 `DIFFERENCE(base, cutter_union)`；
5. **Additive restoration**：检查 additive candidate 表面是否被 cutter 大量覆盖（≥ 0.9），若是则将其从 base 移出，在 difference 之外重新 UNION 回去——即 `(base − cutters) + restored_parts` 而非 `base − cutters − parts`。覆盖率只在候选的**曲面侧壁点**上采样（不含附着的平面端盖点，端盖属于周围基体会稀释比例）。针对同轴 cylinder 的特殊情形（同轴 8° 内、cutter 半径更大、覆盖率 ∈ [0.5, 0.9)），还会把 cutter extent 重设为 additive candidate 端点在 cutter 轴上的投影，使本应等高的两圆柱高度对齐。

CSG IR 节点格式：

```python
# primitive 节点
{"op": "CUBE" | "CYLINDER" | ..., "parameters": {...}, "patch_ids": [...],
 "fitting_error": float, "confidence": float}
# Boolean 节点
{"op": "UNION" | "DIFFERENCE" | "INTERSECTION", "children": [...]}
# DIFFERENCE 的第一个 child 是 base，其余是 cutter
```

---

## 9. Step 8：几何验证

**文件：`csg/csg_core/validation.py`**

### 9.1 递归 occupancy

`csg_contains(node, points)` 递归计算逐点内外：

```text
UNION:         任一 child 包含
INTERSECTION:  所有 child 包含
DIFFERENCE:    base 包含且无任何 cutter 包含
primitive:     primitive_contains(...)
EMPTY:         false
```

`_node_bounds()`：UNION 取子节点包围盒的并、INTERSECTION 取交、DIFFERENCE 只用 base 的包围盒（不让 cutter 错误扩张最终范围）。

### 9.2 表面指标

`validate_csg()` 无论是否提供目标网格都输出：`surface_coverage`（被已选 candidate 覆盖的 patch 面积占比）、`missing_patch_ids`、`mean_fitting_error`。

### 9.3 体素指标

提供 `target_mesh` 时（`voxel_resolution=48`、`maximum_voxels=180000`）：

1. 取 target bounds 与 CSG bounds 的联合范围计算 pitch，体素数超限时按 `pitch *= 1.1` 循环放大；
2. 在体素中心调用 `csg_contains()` 得 CSG occupancy；
3. 目标 occupancy 用 `target_mesh.contains()`，失败时 fallback 到 `target_mesh.voxelized(pitch).fill().is_filled`；
4. 输出 `voxel_iou`、`missing_volume_ratio`、`extra_volume_ratio`、`volume_consistency`、`voxel_pitch`。

这些指标目前主要用于结果报告与人工比较，尚未进入 beam search 的目标函数。

---

## 10. Step 9：OpenSCAD 代码生成

**文件：`csg/csg_core/openscad.py`**

入口 `csg_to_openscad(node, indent=0)` / `OpenSCADExporter.generate(csg_ir)`，对 CSG IR 递归序列化：

| IR 节点 | OpenSCAD 输出 |
|---|---|
| UNION / DIFFERENCE / INTERSECTION | `union(){}` / `difference(){}` / `intersection(){}` 递归 |
| EMPTY | 注释行 |
| CUBE | `translate(center) rotate(euler_xyz_degrees) cube(size, center=true)` |
| SPHERE | `translate(center) sphere(r)` |
| CYLINDER | `translate(起点) rotate(a, v) cylinder(h, r)` |
| CONE | `translate rotate cylinder(h, r1, r2)`（半径 = `max(0, extent)·tan(angle)`） |
| TORUS | `translate rotate rotate_extrude() translate([major,0,0]) circle(r=minor)` |
| EXTRUSION / SPLINE_EXTRUSION | `multmatrix(4×4) linear_extrude(height) polygon(points)` |

**轴旋转（`_axis_rotation`）**：把 OpenSCAD 默认 Z 轴旋转到 candidate 轴——角取 `acos(clip(z·axis))`、轴取 `cross(z, axis)`，平行/反平行退化时用固定 fallback 轴。

**裁剪体（`_clip_shape`）**：带 `clip_bounds` 的 CYLINDER/CONE/TORUS（即 transition 母面）被包装为 `intersection() { 原形状; translate(center) cube(size, center=true); }`，把无限延伸的母面裁到过渡区域内。

Extrusion 的局部 X/Y 基与拉伸轴组成 4×4 变换矩阵，因此拉伸方向不局限于世界坐标轴。

---

## 11. 公共 API 与运行入口

### 11.1 `csg/csg_reconstruction.py`

纯 facade（77 行），实现全部位于 `csg/csg_core/`，保持 notebook 与旧调用方式不变。

函数式接口：`generate_primitive_candidates`、`classify_boolean_operations`、`generate_boolean_hypotheses`、`infer_intersection_groups`、`reconstruct_csg_tree`、`build_csg_ir`、`validate_csg`、`csg_contains`、`csg_to_openscad`，以及各单体生成函数（`generate_cube_candidates`、`generate_extrusion_candidates`、`generate_spline_extrusion_candidates`、`generate_curved_candidates`、`classify_transition_patches`、`generate_transition_curved_candidates`、`generate_planar_notch_cutters`）与查询函数（`primitive_bounds`、`primitive_contains`）。

类接口：`PrimitiveCandidateGenerator`、`PrimitiveFitter`、`BooleanCSGReconstructor`、`CSGValidator`、`OpenSCADExporter`。

### 11.2 标准调用流程（`csg_process.ipynb`）

```python
target_mesh, patches, patch_results, patch_graph = prepare_patch_graph(ply_path)

primitive_candidates = generate_primitive_candidates(
    patch_results, patch_graph, include_torus=True,
)
classify_boolean_operations(primitive_candidates, patch_results)

csg_ir, selected_primitives, search_report = reconstruct_csg_tree(
    primitive_candidates, patch_results,
)   # infer_intersection_groups 与 build_csg_ir 在其内部调用

validation_report = validate_csg(
    csg_ir, selected_primitives, patch_results, target_mesh=target_mesh,
)
openscad_source = csg_to_openscad(csg_ir)
```

可配置项：输入模型 `ply_path`、最小 patch 面数 `min_patch_faces`、是否启用 torus、是否运行 voxel validation、是否打印/可视化 patch（`plot_patch_graph` 依赖可选包 meshplot）、是否导出 `.scad`（默认 False，输出到 `example_results/<model>.scad`）。

---

## 12. 关键数据结构速查

**patch result**（`prepare_patch_graph` 返回的每个 result，拟合信息与图节点信息合并）：

```python
{
    "patch_index": int,            # 稳定算法 ID
    "segment_label": tuple,        # RGB，仅辅助
    "face_id": np.ndarray,         # 回指输入 mesh 的原始面片索引
    "mesh": trimesh.Trimesh, "v": np.ndarray, "f": np.ndarray,
    "type": str,                   # 单 patch 最佳类型
    "fit_rate": float, "params": np.ndarray, "parameter": np.ndarray,
    "fits": dict,                  # 所有启用类型的拟合结果
    "area": float, "boundary_edges": list, "neighboring_patches": list,
}
```

**patch graph**：`{"nodes": [...], "edges": [...]}`，每条 edge = 一条跨 patch 共享边，含 `patch0 / patch1 / v / boundary_v / length / dihedral_angle`。同一对 patch 可有多条边，candidate 阶段由 `_aggregated_graph()` 聚合（累加边界长、长度加权二面角）。

**PrimitiveCandidate**：

```python
PrimitiveCandidate(
    primitive_type,      # CUBE/CYLINDER/CONE/SPHERE/TORUS/EXTRUSION/SPLINE_EXTRUSION
    parameters,          # 类型相关的几何参数（见各节）
    patch_ids,           # 该 candidate 解释的 patch 集合
    fitting_error,       # 归一化几何残差
    confidence,          # 由残差、seed fit rate、约束程度等计算
    operation,           # ADD / SUBTRACT / UNKNOWN
    metadata,            # orientation、containment、frame score、clip_bounds 等诊断信息
)
```

---

## 13. 已知限制

1. **分割依赖**：假设输入 segmentation 大体沿解析曲面或 Boolean 边界分割；跨 primitive 的 patch 会影响后续全部环节。
2. **凹轮廓**：EXTRUSION 轮廓用凸包恢复，不支持凹多边形、带孔轮廓、多连通轮廓（spline extrusion 部分缓解了该问题）。
3. **一般仿射变换**：primitive 支持尺寸与旋转，但无统一的非均匀 scale/shear 变换优化。
4. **小曲面欠约束**：曲面 patch 很小或法向分布不足时，轴/中心/半径可能欠约束。
5. **搜索目标**：beam search 优化 patch coverage、拟合误差、置信度与复杂度，voxel IoU 未直接进入每个状态的评分，非全局几何最优。
6. **intersection 保守**：要求双向表面互包含 + 沿真实边界相接；segmentation 未保留交线时可能漏检。
7. **无精确 B-Rep Boolean**：验证基于解析包含性与体素近似，不做精确 B-Rep 布尔比较。
8. **无全局参数精化**：各 primitive 独立拟合，共轴、共面、等高等关系只靠 heuristic 修正，CSG 树确定后无联合优化。

---

## 14. 备注：与 CSG 无关的相邻代码

`cut.py`、`cut/close_cut.py`、`cut_calculate/` 属于另一条更早期的 mesh cutting/分割流水线（networkx 图切分、切割曲面求交等），CSG 流程（`csg_process.ipynb`、`csg/csg_core/`）对其无任何 import；其中 `cut/close_cut.py` 目前基本是 TODO 骨架。
