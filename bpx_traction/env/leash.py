"""绳子模型：一个虚拟牵引者，牵一根会松会紧的绳。

## 为什么不能用恒力

前面的扛力扫描用的是**理想恒力** —— 外力由脚本指定，和机器人怎么动无关。那对"测策略
扛不扛得住"是对的，但对"测牵引链路"是错的，因为它把最关键的那个回路给抹掉了：

    机器人追上人 → 绳松 → 力掉到零 → 指令掉到零 → 停下
                → 人继续走远 → 绳紧 → 力起来 → 又动

这是个**物理闭环**，力不是外生的，它取决于机器人自己走了多远。松紧循环会不会变成
来回一顿一顿的振荡，恒力场景里永远测不出来 —— 那里绳子永远是紧的。

## 模型

牵引者是个质点，按给定速度在世界里走。绳子连在它和机器人的挂点之间：

    d = ‖牵引者 − 挂点‖
    d ≤ L₀           绳松，张力 0
    d > L₀           绳紧，张力 = min(k·(d − L₀), F_max)，方向从挂点指向牵引者

`k` 是绳子的刚度。这个数影响很大：硬绳（k 大）一绷紧力就窜起来，松紧切换很陡；
弹性带（k 小）过渡平缓。真机上用什么绳，这个数就该是多少 —— 现在是估的。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LeashParams:
    """绳子和牵引者的参数。"""

    #: 绳子的静止长度 (m)。短于它绳子是松的，没有力。
    rest_length_m: float = 1.2
    #: 绳子刚度 (N/m)。⚠️ 估的值 —— 真机用什么绳就该量什么数。
    #: 150 N/m 大致是一条不太有弹性的编织绳：拉长 20 cm 产生 30 N。
    stiffness_n_per_m: float = 150.0
    #: 张力上限 (N)。人手能持续施加的力是有限的，而且超过 30 N 策略就被拖走了
    #: （扛力扫描实测，见 `spec.FORCE_DESIGN_RANGE_N`）。
    max_tension_n: float = 40.0
    #: 牵引者初始位置相对机器人的偏移 (m)，机器人朝 +x。
    start_offset_m: tuple[float, float, float] = (1.0, 0.0, 0.0)
    #: 牵引者手的高度 (m)。影响绳子的仰角，进而影响水平分量占比。
    hand_height_m: float = 0.9


class Leash:
    """逐 env 的虚拟牵引者 + 绳子。每个控制步调一次 :meth:`step`。"""

    def __init__(self, num_envs: int, device, params: LeashParams | None = None):
        self.params = params or LeashParams()
        self.device = device
        self.num_envs = num_envs
        self.handler_pos = torch.zeros(num_envs, 3, device=device)
        self.handler_vel = torch.zeros(num_envs, 3, device=device)
        #: 上一步的状态，诊断用
        self.last_tension_n = torch.zeros(num_envs, device=device)
        self.last_distance_m = torch.zeros(num_envs, device=device)
        self.last_taut = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.attached = torch.ones(num_envs, dtype=torch.bool, device=device)

    # ----------------------------------------------------------------------
    def reset(self, attach_point_w: torch.Tensor) -> None:
        """把牵引者放到机器人前方，绳子处于刚好不紧的状态。"""
        offset = torch.tensor(self.params.start_offset_m, device=self.device)
        self.handler_pos = attach_point_w + offset
        self.handler_pos[:, 2] = self.params.hand_height_m
        self.handler_vel.zero_()
        self.attached.fill_(True)
        self.last_tension_n.zero_()
        self.last_taut.fill_(False)

    def set_velocity(self, velocity_w: torch.Tensor) -> None:
        """设定牵引者的行走速度 (N, 3)，世界系。"""
        self.handler_vel = velocity_w.to(self.device, torch.float32).reshape(-1, 3)

    def detach(self, env_ids: torch.Tensor | None = None) -> None:
        """松手 / 绳子断了。之后张力恒为零 —— 用来测"释放后停不停得住"。"""
        if env_ids is None:
            self.attached.fill_(False)
        else:
            self.attached[env_ids] = False

    # ----------------------------------------------------------------------
    def step(self, attach_point_w: torch.Tensor, dt: float) -> torch.Tensor:
        """推进牵引者，返回这一步绳子施加在机器人上的世界系力 `(N, 3)`。"""
        p = self.params
        self.handler_pos = self.handler_pos + self.handler_vel * dt
        self.handler_pos[:, 2] = p.hand_height_m          # 手一直在同一高度

        delta = self.handler_pos - attach_point_w          # 挂点 → 牵引者
        distance = torch.linalg.norm(delta, dim=1)
        stretch = torch.clamp(distance - p.rest_length_m, min=0.0)
        tension = torch.clamp(stretch * p.stiffness_n_per_m, max=p.max_tension_n)
        tension = torch.where(self.attached, tension, torch.zeros_like(tension))

        direction = delta / distance.clamp_min(1e-6).unsqueeze(1)
        force_w = direction * tension.unsqueeze(1)

        self.last_distance_m = distance
        self.last_tension_n = tension
        self.last_taut = tension > 1e-6
        return force_w

    # ----------------------------------------------------------------------
    def describe(self) -> str:
        p = self.params
        return (
            f"绳子  静止长 {p.rest_length_m:.2f} m   刚度 {p.stiffness_n_per_m:.0f} N/m"
            f"（拉长 20 cm → {0.2 * p.stiffness_n_per_m:.0f} N）   张力上限 "
            f"{p.max_tension_n:.0f} N\n"
            f"      ⚠️ 刚度是估的，真机用什么绳就该量什么数"
        )
