# 项目交接摘要

更新：2026-10-12；代码快照：`f0ae122`。本文件只保留跨设备续接所需状态，具体实现以当前代码为准。

## 目标与入口

目标是将 ABC 等 CAD 网格恢复为几何等价的 primitive + Boolean 表达，并研究去圆角。三段流程：

1. **Mesh segmentation**：`mesh_segmentation.ipynb`；`patch_segment/`、`patch_graph/`、`cutting/`、`hole_filling/`、`primitive_fitting/`。结合凹线切割、区域扩散和曲面拟合形成面片。
2. **Segment → CSG**：`csg_process.ipynb` → `csg/graph_construct.py:prepare_patch_graph` → `csg/csg_core/{candidates,boolean,validation,openscad}.py`。输入为按 RGB 标记面片的 PLY，输出 CSG IR、验证结果和 OpenSCAD。流程与阈值见 `csg/CSG_TECHNICAL_DOCUMENTATION.md`。
3. **DeFillet**：`fillet_detector/defillet_py/` 为检测与移除的 Python 实现；运行方式及与 C++ 版本的差异见 `fillet_detector/README.md`。

## 已完成的近期工作

- CSG 验证改为按列求交；圆锥拟合去除重复起点。`00024792` 的候选生成曾由约 17.4 秒降至 10.3 秒，仍需处理大量小面片的拟合耗时。
- 针对 `00024792` 十字槽薄壁、`00023582` 内圆角、`00025063` 双端圆角与中心孔、`00140710` 圆角误分类、`00141097` 阶梯孔薄封口、`00024448` 浅斜面、`00140982` 六角凹槽作了候选或布尔修复。数据与限制见 CSG 技术文档第 14 节；不要把 48 分辨率 IoU≈1 当作精确拓扑证明。
- `todo.md` 给出 ABC 分割一致性、可见边界的基元支撑、布尔胞腔占据及误差界的论文式分析；它是研究分析，不是逐条执行清单。

## 当前未解决问题与下一步

- CSG 技术文档仍记录 `00025611` 环面候选、`00023582` 拉伸体最终选择、`00140553` 第四个盒体、`00025183` 覆盖率阈值等回归差异；开始修复前复核当前测试结果。
- Segmentation → CSG 接口目前主要依赖面片 RGB，尚未明确传递连通分量、边界环和带符号凸凹关系。研究目标是保持现有主体区域扩散思路，同时补足过渡特征与布尔语义。
- 继续建立多分辨率 voxel IoU、双向表面误差、法向两侧占据、孔洞贯通的联合评价；按模型特征分层报告，而非只看 `surface_coverage`。

## 跨设备注意

样例配置和旧 README 中可能有设备专属绝对路径，运行前按本机位置调整。测试入口主要位于 `csg/test/` 和 `tests/`；本次仅编写交接文档，未重新运行模型测试。
