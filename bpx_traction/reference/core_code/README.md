# 力估计器核心实现代码包

生成日期：2026-10-21（Asia/Shanghai）

本包只包含代码和接口契约，不包含 checkpoint、TorchScript/ONNX 权重、原始轨迹或运行缓存。

## 目录

- `training/`：训练、微调、蒸馏、融合和部署打包脚本。
- `inference/`：MuJoCo/MJLab 在线 rollout、数据采集和闭环分析脚本。
- `contracts/`：输入观测、帧坐标、模型 I/O 和基线契约。
- `verification/`：Torch/ONNX 导出一致性、闭环和方向矩阵验收分析。
- `acceptance/`：当前 acceptance_v2 验收分析、验证和报告生成。
- `diagnostics/`：Fz 力大小、方向和姿态误差分解。

## 当前模型接口

单帧输入 64 维，历史长度 25 帧，采样周期 20 ms；输出包含状态估计、三维外力估计和置信度。

运行代码前需要准备项目依赖、模型权重和数据路径；本包本身不包含这些大文件。
