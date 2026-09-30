"""JSON 参数读写；忽略未知字段以兼容原版 C++ 配置。"""

from dataclasses import dataclass, asdict, fields
import json
from pathlib import Path


@dataclass
class DetectorParameters:
    """检测参数；长度在归一化网格上计算，角度单位为度。"""
    path: str = ""
    out_dir: str = ""
    epsilon: float = 0.08
    radius_thr: float = 0.04
    lamdba: float = 0.5  # 保留历史属性拼写；JSON 推荐使用 lambda。
    angle_thr: float = 30.0
    sigma: float = 5.0
    num_patches: int = -1
    num_neighbors: int = 100
    num_smooth_iter: int = 50  # 保留原配置字段；当前 Python 检测流程未使用。
    num_sor_iter: int = 2
    num_sor_neighbors: int = 50
    num_sor_std_ratio: float = 0.8
    num_threads: int = -1


@dataclass
class RemoverParameters:
    """去圆角参数；当前近似求解器仅使用 beta_e 和 num_opt_iter。"""
    path: str = ""
    label_path: str = ""
    out_dir: str = ""
    beta_e: float = 2.0
    beta_f: float = 1.0
    beta_c: float = 1.0
    angle_thr: float = 30.0
    num_opt_iter: int = 15
    num_threads: int = -1


def load_config(path, cls):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {field.name for field in fields(cls)}
    if "lamdba" in names and "lambda" in values:
        values["lamdba"] = values["lambda"]
    return cls(**{key: value for key, value in values.items() if key in names})


def save_config(params, path):
    values = asdict(params)
    if "lamdba" in values:
        values["lambda"] = values.pop("lamdba")
    Path(path).write_text(json.dumps(values, indent=4), encoding="utf-8")
