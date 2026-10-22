"""往机身上施加牵引力，并产出机身系的力标签。

扛力扫描和数据采集共用这一个 —— 两边必须施一模一样的力，否则"扫出来的饱和点"和
"训出来的估计器"对的不是同一个物理量。

## 为什么把力旋到机身系再施加，而不是直接用 is_global=True

`WrenchComposer.set_forces_and_torques` 的 `positions` 参数语义**随 is_global 变**
（`isaaclab/utils/warp/kernels.py` 的 `set_forces_and_torques_at_position`）：

* ``is_global=False`` → positions 是**相对 link frame 的偏移**
* ``is_global=True``  → positions 是力作用点的**世界坐标**

挂点是个机身系偏移（而且逐 env 可以不同），所以走 ``is_global=False`` 这条：自己把世界系力
旋进机身系，positions 直接给偏移。好处是顺带就拿到了**标签**——

    label_contract: 合外力旋到当前机身坐标系

这和 Go2 早期那一代的 `label_contract: 'integrated force, current body coordinates'`
是同一个约定（见 `reference/PROVENANCE.md` §2.3）。施力和打标签用同一个张量，
不可能对不上。

## 挂点是个区域，不是一个点

真机上人想从哪儿拉就从哪儿拉。所以估计器必须对挂点**不变**，做法是采数据时按 env 在
`spec.TETHER_OFFSET_RANGE_B` 里随机，而不是把它钉死在一个点上。

代价要明说：同一个力作用在不同挂点上，力臂不同 → 俯仰/横滚力矩不同 → 本体响应不同。
随机化之后"本体感知 → 外力"是个**一对多**的映射，估计器只能学到条件均值，这部分误差
消不掉，范围越大下限越高。所以它是个**该测的消融项**：固定挂点训一版、随机挂点训一版，
力精度的差就是"挂哪儿都行"这个便利的标价。

确定性评估（扛力扫描、回放、对比）要用固定的标称挂点 `spec.TETHER_OFFSET_B`，
否则不同策略之间比的不是同一件事。
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

        self.nominal_offset_b = tuple(spec.TETHER_OFFSET_B if offset_b is None else offset_b)
        #: 每个 env 的挂点偏移 (N, 3)。默认全是标称值，`randomize_offsets()` 之后才分散。
        self.offsets_b = torch.tensor(
            self.nominal_offset_b, dtype=torch.float32, device=self.device
        ).reshape(1, 3).repeat(robot.num_instances, 1)
        self._randomized = False
        self._sync_positions()

        if warn_unconfirmed and not spec.TETHER_OFFSET_CONFIRMED:
            print(
                f"[traction] ⚠️ 挂点未经真机确认。标称 {self.body_name}+"
                f"{self.nominal_offset_b}，随机范围 {spec.TETHER_OFFSET_RANGE_B}。"
                f"按真机实际能挂的位置量一遍再改 spec.TETHER_OFFSET_CONFIRMED。"
            )

        self._zero_torque = torch.zeros(robot.num_instances, 1, 3, device=self.device)
        #: 上一次施加的机身系力 (N, 3)，就是数据集的标签。
        self.last_force_b = torch.zeros(robot.num_instances, 3, device=self.device)
        #: 上一次施加的世界系力 (N, 3)，仅供诊断，**不要当标签**。
        self.last_force_w = torch.zeros(robot.num_instances, 3, device=self.device)

    # ----------------------------------------------------------------------
    def _sync_positions(self) -> None:
        """挂点全为零时省掉 positions（少一次 cross product）。"""
        self._positions = (
            self.offsets_b.reshape(-1, 1, 3)
            if bool((self.offsets_b.abs() > 1e-9).any())
            else None
        )

    def randomize_offsets(
        self,
        offset_range: tuple[tuple[float, float], ...] | None = None,
        env_ids: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """按 env 在范围里均匀重采挂点。**采训练数据时每次 reset 都要调。**

        真机上挂点不固定，估计器必须对它不变 —— 而不变是训出来的，不是假设出来的。
        代价见模块 docstring：这会把"本体感知 → 外力"变成一对多映射，留下一个消不掉的
        误差下限。跑固定挂点和随机挂点各一版、比力精度，才知道这个下限有多高。
        """
        ranges = offset_range or spec.TETHER_OFFSET_RANGE_B
        if len(ranges) != 3:
            raise ValueError(f"挂点范围要给 3 个轴，拿到 {len(ranges)}")

        if env_ids is None:
            env_ids = torch.arange(self.robot.num_instances, device=self.device)
        n = len(env_ids)
        sample = torch.rand(n, 3, device=self.device, generator=generator)
        lo = torch.tensor([r[0] for r in ranges], device=self.device)
        hi = torch.tensor([r[1] for r in ranges], device=self.device)
        self.offsets_b[env_ids] = lo + sample * (hi - lo)

        self._randomized = True
        self._sync_positions()
        return self.offsets_b

    def reset_offsets(self) -> None:
        """回到标称挂点。确定性评估（扛力扫描、策略对比）必须用这个，
        否则不同策略之间比的不是同一件事。"""
        self.offsets_b[:] = torch.tensor(
            self.nominal_offset_b, dtype=torch.float32, device=self.device
        )
        self._randomized = False
        self._sync_positions()

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
        if self._positions is None:
            return link_pos
        quat = self.robot.data.body_quat_w[:, self.body_id]
        return link_pos + quat_apply(quat, self.offsets_b)

    def describe(self) -> str:
        confirmed = "已确认" if spec.TETHER_OFFSET_CONFIRMED else "⚠️ 未经真机确认"
        if self._randomized:
            lo = self.offsets_b.min(dim=0).values.tolist()
            hi = self.offsets_b.max(dim=0).values.tolist()
            where = (f"逐 env 随机，实际跨度 "
                     f"x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] "
                     f"z[{lo[2]:+.3f},{hi[2]:+.3f}] m")
        else:
            where = f"固定标称 {self.nominal_offset_b} m"
        return (
            f"挂点 {self.body_name}(body_id={self.body_id})  {where}  [{confirmed}]\n"
            f"标签 = 合外力旋到当前机身系（和施加的是同一个张量）"
        )
