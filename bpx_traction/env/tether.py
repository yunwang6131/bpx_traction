"""往机身上施加牵引力，并产出机身系的力标签。

扛力扫描和数据采集共用这一个 —— 两边必须施一模一样的力，否则"扫出来的饱和点"和
"训出来的估计器"对的不是同一个物理量。

## 为什么把力旋到机身系再施加，而不是直接用 is_global=True

`WrenchComposer.set_forces_and_torques` 的 `positions` 参数语义**随 is_global 变**
（`isaaclab/utils/warp/kernels.py` 的 `set_forces_and_torques_at_position`）：

* ``is_global=False`` → positions 是**相对 link frame 的偏移**
* ``is_global=True``  → positions 是力作用点的**世界坐标**

挂点是个固定的机身系偏移，所以走 ``is_global=False`` 这条：自己把世界系力旋进机身系，
positions 直接给常量偏移。好处是顺带就拿到了**标签**——

    label_contract: 合外力旋到当前机身坐标系

这和 Go2 早期那一代的 `label_contract: 'integrated force, current body coordinates'`
是同一个约定（见 `reference/PROVENANCE.md` §2.3）。施力和打标签用同一个张量，
不可能对不上。

## 挂点为什么要紧

挂点在质心上方 h 处时，水平力 F 会额外产生 F·h 的俯仰力矩，而**姿态信号比力本身好读**
（IMU 直接给四元数和角速度）。所以挂高一点反而更好估。

更要命的是：**仿真施力点必须和真机挂点一致**。不一致的话，"同样的力、不同挂点"会产生
不同的本体响应，估计器学到的是错的映射 —— 这是会污染整个数据集、而且在仿真里永远测
不出来的那种错。`spec.TETHER_OFFSET_CONFIRMED` 就是这个闸门。
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse

from .. import spec


class TetherApplier:
    """给每个 env 施加一个独立的世界系牵引力。

    用法::

        tether = TetherApplier(env.scene["robot"])
        for step in ...:
            force_b = tether.apply_world(forces_w)      # (N, 3) 世界系 → 返回机身系标签
            env.step(action)
        tether.clear()

    力是**持久**的：`WrenchComposer` 每个物理步都会施加，直到下次 set 或 clear。
    所以恒力场景只需设一次，变力场景每个控制步设一次。
    """

    def __init__(
        self,
        robot: Articulation,
        body_name: str | None = None,
        offset_b: tuple[float, float, float] | None = None,
        warn_unconfirmed: bool = True,
    ):
        self.robot = robot
        self.device = robot.device
        self.body_name = body_name or spec.TETHER_BODY

        body_ids, names = robot.find_bodies([self.body_name], preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(
                f"挂点刚体 {self.body_name!r} 解析出 {len(body_ids)} 个：{names}。"
                f"可选的有 {robot.body_names}"
            )
        self.body_id = int(body_ids[0])
        self._body_ids = torch.tensor([self.body_id], dtype=torch.long, device=self.device)

        offset = tuple(spec.TETHER_OFFSET_B if offset_b is None else offset_b)
        self.offset_b = offset
        self._has_offset = any(abs(v) > 1e-9 for v in offset)
        self._positions = (
            torch.tensor(offset, dtype=torch.float32, device=self.device)
            .reshape(1, 1, 3)
            .repeat(robot.num_instances, 1, 1)
            if self._has_offset
            else None
        )

        if warn_unconfirmed and not spec.TETHER_OFFSET_CONFIRMED:
            print(
                f"[traction] ⚠️ 挂点 {self.body_name}+{offset} 未经真机确认。"
                f"仿真施力点和真机挂点不一致会让标签有歧义 —— "
                f"量过之后改 spec.TETHER_OFFSET_B 和 TETHER_OFFSET_CONFIRMED。"
            )

        self._zero_torque = torch.zeros(robot.num_instances, 1, 3, device=self.device)
        #: 上一次施加的机身系力 (N, 3)，就是数据集的标签。
        self.last_force_b = torch.zeros(robot.num_instances, 3, device=self.device)
        #: 上一次施加的世界系力 (N, 3)，仅供诊断，**不要当标签**。
        self.last_force_w = torch.zeros(robot.num_instances, 3, device=self.device)

    # ----------------------------------------------------------------------
    def apply_world(self, forces_w: torch.Tensor) -> torch.Tensor:
        """施加世界系力 `(N, 3)`，返回机身系的同一个力 `(N, 3)`（= 标签）。"""
        forces_w = forces_w.to(self.device, torch.float32).reshape(-1, 3)
        if forces_w.shape[0] != self.robot.num_instances:
            raise ValueError(
                f"力的 batch 是 {forces_w.shape[0]}，环境数是 {self.robot.num_instances}"
            )

        quat_w = self.robot.data.root_quat_w            # (N, 4) wxyz，和 spec.QUAT_ORDER 一致
        forces_b = quat_apply_inverse(quat_w, forces_w)

        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_b.unsqueeze(1),               # (N, 1, 3)
            torques=self._zero_torque,
            positions=self._positions,                  # 局部偏移，is_global=False 下才是这个语义
            body_ids=self._body_ids,
            is_global=False,
        )

        self.last_force_w = forces_w.clone()
        self.last_force_b = forces_b.clone()
        return self.last_force_b

    def apply_body(self, forces_b: torch.Tensor) -> torch.Tensor:
        """直接施加机身系力。诊断用 —— 牵引绳的方向在世界里是固定的，
        所以正常采集一律走 :meth:`apply_world`。"""
        forces_b = forces_b.to(self.device, torch.float32).reshape(-1, 3)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_b.unsqueeze(1),
            torques=self._zero_torque,
            positions=self._positions,
            body_ids=self._body_ids,
            is_global=False,
        )
        self.last_force_b = forces_b.clone()
        self.last_force_w = quat_apply_inverse(
            self.robot.data.root_quat_w, forces_b
        )  # 近似：仅诊断用
        return self.last_force_b

    def clear(self) -> None:
        """撤掉外力。复位之后必须调，否则上一轮的力会跟着带过来。"""
        zeros = torch.zeros(self.robot.num_instances, 1, 3, device=self.device)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=self._zero_torque,
            body_ids=self._body_ids,
            is_global=False,
        )
        # 这里**不能**用原地的 zero_()：`apply_world` 多半是在 `torch.inference_mode()`
        # 里被调的，存下来的 last_force_* 就是 inference tensor，在推理块外面对它做原地
        # 操作会直接 RuntimeError。一律赋新张量。
        self.last_force_b = torch.zeros(self.robot.num_instances, 3, device=self.device)
        self.last_force_w = torch.zeros(self.robot.num_instances, 3, device=self.device)

    # ----------------------------------------------------------------------
    def attach_point_w(self) -> torch.Tensor:
        """挂点的世界坐标 `(N, 3)`，画图/诊断用。"""
        from isaaclab.utils.math import quat_apply

        link_pos = self.robot.data.body_pos_w[:, self.body_id]
        if not self._has_offset:
            return link_pos
        offset = torch.tensor(self.offset_b, dtype=torch.float32, device=self.device)
        quat = self.robot.data.body_quat_w[:, self.body_id]
        return link_pos + quat_apply(quat, offset.expand_as(link_pos))

    def describe(self) -> str:
        confirmed = "已确认" if spec.TETHER_OFFSET_CONFIRMED else "⚠️ 未经真机确认"
        return (
            f"挂点 {self.body_name}(body_id={self.body_id}) + {self.offset_b} m  [{confirmed}]\n"
            f"标签 = 合外力旋到当前机身系（和施加的是同一个张量）"
        )
