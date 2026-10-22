"""输入归一化：带列名的 mean / std。

**为什么要带列名**：Go2 那份 `normalization.npz` 里只有两个无名的 64 维数组。
重建帧布局时只能靠统计量反推（`mean[2] ≈ 9.87` 推出"这三维是含重力的加速度"、
std 的 3 周期 / 4 分组推出关节顺序），推完还不能确定 —— 直到在
`estimator_io.yaml` 里找到权威定义才敢用。那是几个小时，而代价只是当初没存 12 个字符串。

这里存进去，并且在 load 时核对列名和当前 :mod:`bpx_traction.spec` 一致。
契约改了而归一化文件没跟着重算，会在加载时立刻报错，而不是训出一个悄悄错位的模型。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import spec

#: std 的下限。某些维在数据里可能几乎不变（比如机器人一直站着时的某个关节），
#: 不夹住会把噪声放大成巨大的输入值。
MIN_STD = 1e-4


def frame_columns() -> list[str]:
    """61 个维度的名字，例如 ``imu_acc_b[0]`` / ``joint_torque[fl_knee_joint]``。

    按关节的四项用关节名而不是下标 —— 这样万一关节顺序变了，列名对不上会直接报错。
    """
    names: list[str] = []
    for term, dim in spec.TERMS:
        if term in spec.PER_JOINT_TERMS:
            names += [f"{term}[{joint}]" for joint in spec.JOINT_NAMES]
        elif term == "quat_wb":
            names += [f"{term}[{axis}]" for axis in spec.QUAT_ORDER]
        else:
            names += [f"{term}[{i}]" for i in range(dim)]
    assert len(names) == spec.FRAME_SIZE, f"列名 {len(names)} 个，帧 {spec.FRAME_SIZE} 维"
    return names


class Normalizer:
    """``(x - mean) / std``，按帧的最后一维（61）逐维归一。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray, columns: list[str] | None = None):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.maximum(np.asarray(std, dtype=np.float32).reshape(-1), MIN_STD)
        self.columns = list(columns) if columns is not None else frame_columns()
        if not (len(self.mean) == len(self.std) == len(self.columns)):
            raise ValueError(
                f"mean({len(self.mean)}) / std({len(self.std)}) / "
                f"columns({len(self.columns)}) 长度不一致"
            )

    @classmethod
    def from_history(cls, history: np.ndarray) -> "Normalizer":
        """从 (N, T, D) 的数据集统计。和参考实现一样，在 (0, 1) 两个维度上统计。"""
        history = np.asarray(history)
        if history.ndim != 3 or history.shape[-1] != spec.FRAME_SIZE:
            raise ValueError(
                f"期望 (N, T, {spec.FRAME_SIZE}) 的历史，拿到 {history.shape}"
            )
        return cls(history.mean(axis=(0, 1)), history.std(axis=(0, 1)))

    def __call__(self, history: np.ndarray) -> np.ndarray:
        return (history - self.mean) / self.std

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            columns=np.asarray(self.columns, dtype=object),
            frame_size=np.asarray(spec.FRAME_SIZE),
            history_length=np.asarray(spec.HISTORY_LENGTH),
        )
        return path

    @classmethod
    def load(cls, path: str | Path, check_columns: bool = True) -> "Normalizer":
        data = np.load(Path(path), allow_pickle=True)
        columns = [str(c) for c in data["columns"]]
        if check_columns:
            expected = frame_columns()
            if columns != expected:
                diff = [
                    f"  [{i}] 文件={got!r} 当前契约={want!r}"
                    for i, (got, want) in enumerate(zip(columns, expected))
                    if got != want
                ]
                raise ValueError(
                    f"归一化文件的列名和当前 spec 对不上（{len(columns)} vs "
                    f"{len(expected)} 维）。契约改过就要重算归一化。\n"
                    + "\n".join(diff[:10])
                )
        return cls(data["mean"], data["std"], columns)

    def describe(self, top: int = 0) -> str:
        """打出每一维的 mean / std，用来肉眼核对数据是否合理。"""
        lines = []
        for name, m, s in zip(self.columns, self.mean, self.std):
            lines.append(f"  {name:<34s} mean {m:9.4f}   std {s:8.4f}")
        if top:
            lines = lines[:top]
        return "\n".join(lines)


def sanity_check(norm: "Normalizer") -> list[str]:
    """几条能一眼看出数据错没错的检查，返回警告列表（空=没问题）。

    这些不是硬断言 —— 数据合理与否要人看。但下面几条如果触发，基本就是采集出错了。
    """
    warnings: list[str] = []
    cols = {name: i for i, name in enumerate(norm.columns)}

    # 加速度 z 应该在 ±g 附近：契约要求含重力（IsaacLab 的 ImuCfg.gravity_bias
    # 默认 (0,0,9.81)，真 IMU 也输出比力）。如果接近 0，说明用的是运动学加速度，
    # sim2real 会直接废掉。
    i = cols.get("imu_acc_b[2]")
    if i is not None and abs(abs(norm.mean[i]) - 9.81) > 3.0:
        warnings.append(
            f"imu_acc_b[2] 的均值是 {norm.mean[i]:.2f}，不在 ±9.81 附近 —— "
            f"加速度是不是没含重力？（IsaacLab 要 ImuCfg.gravity_bias=(0,0,9.81)）"
        )

    # 投影重力的模长应该 ≈ 1
    idx = [cols[f"projected_gravity_b[{k}]"] for k in range(3) if f"projected_gravity_b[{k}]" in cols]
    if len(idx) == 3:
        norm_g = float(np.linalg.norm(norm.mean[idx]))
        if not 0.8 <= norm_g <= 1.2:
            warnings.append(f"投影重力均值的模长是 {norm_g:.3f}，应该 ≈1（是不是没归一化？）")

    # 四元数模长 ≈ 1（均值的模长会小于 1，因为 yaw 在转，但不该接近 0）
    idx = [cols[f"quat_wb[{a}]"] for a in spec.QUAT_ORDER if f"quat_wb[{a}]" in cols]
    if len(idx) == 4 and float(np.linalg.norm(norm.mean[idx])) < 0.3:
        warnings.append("四元数均值的模长 <0.3，检查一下顺序和归一化")

    # 任何一维 std 贴着下限，说明那一维在数据里是常数 —— 覆盖度不够
    flat = [norm.columns[i] for i in range(len(norm.std)) if norm.std[i] <= MIN_STD * 1.001]
    if flat:
        warnings.append(f"这些维在数据里几乎不变，覆盖度不足：{flat[:8]}")

    return warnings
