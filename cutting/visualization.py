import numpy as np
import trimesh
from trimesh.scene.cameras import Camera
# from trimesh.transformations import look_at

def look_at_matrix(eye, target, up=np.array([0, 1, 0])):
    """
    仿照 OpenGL 的 LookAt 矩阵，返回 4x4 世界到相机变换矩阵。
    参数:
        eye: 相机位置 (3,)
        target: 观察目标点 (3,)
        up: 上方向 (3,)
    返回:
        4x4 numpy.ndarray
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)

    m = np.eye(4)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -np.dot(m[:3, :3], eye)
    return m

def _orthonormal_basis_from_normal(n):
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u);   v /= (np.linalg.norm(v) + 1e-12)
    return u, v

def _make_plane_quad(plane_abcd, center, diag_len, scale=1.2):
    a, b, c, _ = plane_abcd
    n = np.array([a, b, c], dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)
    u, v = _orthonormal_basis_from_normal(n)
    half = 0.5 * diag_len * scale
    corners = np.array([
        center + (+half)*u + (+half)*v,
        center + (-half)*u + (+half)*v,
        center + (-half)*u + (-half)*v,
        center + (+half)*u + (-half)*v,
    ])
    faces = np.array([[0,1,2],[0,2,3]], dtype=int)
    return corners, faces, n

def show_mesh_cut_trimesh(
    v, f, v_cut_idx, plane_abcd,
    cut_color=(220, 20, 60, 255),    # crimson (opaque line)
    mesh_color=(204, 204, 204, 255), # smooth mesh, no wireframe
    plane_color=(255, 215, 0, 64),   # gold, semi-transparent
    window_size=(1280, 960),
    view_look=None,                  # (eye, center, up) in world; if None, auto
    fov_deg=45.0
):
    """
    使用 trimesh+pyglet 渲染：光滑网格 + 切割线 + 半透明平面（能正确“穿透”）
    参数：
        v: (N,3) float 顶点
        f: (M,3) int   三角面
        v_cut_idx: list[int] 按顺序的切割线顶点索引（若闭合可首尾相同或自行闭合）
        plane_abcd: (a,b,c,d) 平面 ax+by+cz+d=0
        view_look: (eye, center, up) 可选固定相机
    """
    v = np.asarray(v, float)
    f = np.asarray(f, int)
    a, b, c, d = plane_abcd

    # 主体网格（设置面颜色的 RGBA；无线框由 viewer 决定，这里只提供几何+颜色）
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual.face_colors = np.tile(mesh_color, (len(mesh.faces), 1))

    # 半透明平面：以 v 的质心投影到平面为中心，尺寸按包围盒对角线
    bbox_min, bbox_max = v.min(0), v.max(0)
    diag_len = np.linalg.norm(bbox_max - bbox_min)
    center = v.mean(0)
    n = np.array([a, b, c], float); n /= (np.linalg.norm(n) + 1e-12)
    center_on_plane = center - (np.dot(center, n) + d) * n

    plane_v, plane_f, plane_n = _make_plane_quad(plane_abcd, center_on_plane, diag_len, scale=1.2)
    plane_mesh = trimesh.Trimesh(vertices=plane_v, faces=plane_f, process=False)
    plane_mesh.visual.face_colors = np.tile(plane_color, (len(plane_mesh.faces), 1))

    # 切割线：用 Path3D（线框图元），或想要“粗线”可换成小半径圆柱段拼接
    cut_idx = np.asarray(v_cut_idx, int)
    cut_xyz = v[cut_idx]
    # 构造线段列表，每段形状 (2,3)
    segments = [cut_xyz[i:i+2] for i in range(len(cut_xyz)-1)]
    # 如需闭合，请取消下一行注释（前提：不是已经首尾相同）
    # segments.append(np.vstack([cut_xyz[-1], cut_xyz[0]]))
    cut_path = trimesh.load_path(segments)
    # 统一设置路径颜色（RGBA）
    cut_path.colors = np.array([cut_color] * len(cut_path.entities))

    # 组合到 Scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name='mesh')
    scene.add_geometry(plane_mesh, geom_name='cut_plane')
    scene.add_geometry(cut_path, geom_name='cut_path')

    # 背景白色（RGBA）
    scene.background = (255, 255, 255, 255)

    # 设置相机
    if view_look is None:
        # 默认：从包围盒对角看向中心；up 自动近似 y 轴方向
        center = (bbox_min + bbox_max) / 2
        extent = (bbox_max - bbox_min)
        eye = center + 1.8 * extent  # 斜向远处
        up = np.array([0.0, 1.0, 0.0])
    else:
        eye, center, up = view_look

    cam = Camera(resolution=window_size, fov=(fov_deg, fov_deg))
    scene.camera = cam
    scene.camera_transform = look_at_matrix(eye, center, up)

    # 显示（使用 pyglet viewer）
    # - 支持透明混合与深度缓冲，平面不会“整块盖住”模型
    # - 鼠标交互旋转/缩放；按 's' 可截图（默认到工作目录）
    scene.show(resolution=window_size, viewer='gl')  # 默认 viewer='gl'，即 pyglet

# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    # 一个简单立方体 + 示例平面 + 示例切割线
    v = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], float)
    f = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [1,2,6],[1,6,5],
        [2,3,7],[2,7,6],
        [3,0,4],[3,4,7]
    ], int)
    plane_abcd = (1.0, -1.0, 0., 0)
    # 假设切割线按顺序给出（如闭合边界，可令首尾相同或在代码中追加闭边）
    v_cut_idx = [0,2,4,6]

    # 可自定义相机 (eye, center, up)，否则自动
    # eye = np.array([2.5, 2.0, 2.0]); center = np.array([0.5,0.5,0.5]); up = np.array([0,1,0])
    # view = (eye, center, up)
    view = None

    show_mesh_cut_trimesh(
        v, f, v_cut_idx, plane_abcd,
        cut_color=(220, 20, 60, 255),
        mesh_color=(210, 210, 210, 255),
        plane_color=(255, 215, 0, 72),     # 调大透明度可更“虚”
        window_size=(1200, 900),
        view_look=view,
        fov_deg=40.0
    )