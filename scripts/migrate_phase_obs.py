#!/usr/bin/env python3
"""把 45 维观测的 rollfix 权重迁移到 47 维(相位进观测)的 Sprint6 网络。

相位进观测在 45 维帧末尾追加 2 维 [sinφ, cosφ]，网络输入随之变宽：
  actor  base policy 第一层  [512, 53] -> [512, 55]   （base 45→47 + z 8）
  actor  base_normalizer     [1, 45]   -> [1, 47]
  critic obs_normalizer/mlp.0 110       -> 112          （policy 45→47 + priv 65）

迁移规则 = **前 45 维(policy)照搬、相位 2 维权重置 0、其后(z / privileged)原样后移**。
相位维权重为 0 ⇒ 策略初始**完全忽略**相位，等于从"会走、横滚修好"的 rollfix 出发，
训练中再逐步学会用这 2 维。encoder / enc_normalizer / 其余各层输入输出维度不变，直接
照搬。optimizer 状态因维度变了作废，丢弃让它重建。

    python scripts/migrate_phase_obs.py <rollfix.pt> <输出.pt>
"""
import sys

import torch

PHASE = 2            # 追加的相位维数
POLICY = 45          # 原 policy 帧维数
PRIV = 65            # privileged 维数


def expand_cols(w: torch.Tensor, insert_at: int, n: int) -> torch.Tensor:
    """在列维度 insert_at 处插入 n 列 0，其余列原样保留（行不变）。"""
    out = torch.zeros(w.shape[0], w.shape[1] + n, dtype=w.dtype)
    out[:, :insert_at] = w[:, :insert_at]
    out[:, insert_at + n:] = w[:, insert_at:]
    return out


def expand_norm(sd: dict, prefix: str, insert_at: int, n: int) -> None:
    """EmpiricalNormalization 的 _mean/_var/_std 扩维：mean 插 0、var/std 插 1。"""
    for key, fill in [("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)]:
        v = sd[f"{prefix}.{key}"]                        # (1, D)
        new = torch.full((1, v.shape[1] + n), fill, dtype=v.dtype)
        new[:, :insert_at] = v[:, :insert_at]
        new[:, insert_at + n:] = v[:, insert_at:]
        sd[f"{prefix}.{key}"] = new


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    # --- actor：base 第一层 [.,53]->[.,55]，相位插在 base 末尾(第 45 列) ---
    a = ckpt["actor_state_dict"]
    assert a["mlp.0.weight"].shape[1] == POLICY + 8, a["mlp.0.weight"].shape  # 45+z8=53
    a["mlp.0.weight"] = expand_cols(a["mlp.0.weight"], POLICY, PHASE)
    expand_norm(a, "base_normalizer", POLICY, PHASE)
    # encoder / enc_normalizer / mlp.2,4,6 / std_param / count 不变

    # --- critic：输入 = policy(45) + priv(65) = 110 -> 112，相位插在 policy 末尾 ---
    c = ckpt["critic_state_dict"]
    assert c["mlp.0.weight"].shape[1] == POLICY + PRIV, c["mlp.0.weight"].shape  # 110
    c["mlp.0.weight"] = expand_cols(c["mlp.0.weight"], POLICY, PHASE)
    expand_norm(c, "obs_normalizer", POLICY, PHASE)

    # optimizer 维度作废，iter 归零（新阶段从头计）
    ckpt.pop("optimizer_state_dict", None)
    ckpt["iter"] = 0

    torch.save(ckpt, dst)
    print(f"[migrate] {src}")
    print(f"       -> {dst}")
    print(f"  actor  mlp.0.weight  {tuple(a['mlp.0.weight'].shape)}  base_norm {tuple(a['base_normalizer._mean'].shape)}")
    print(f"  critic mlp.0.weight  {tuple(c['mlp.0.weight'].shape)}  obs_norm  {tuple(c['obs_normalizer._mean'].shape)}")
    print("  相位 2 维权重=0（策略初始忽略相位，训练中学会用）")


if __name__ == "__main__":
    main()
