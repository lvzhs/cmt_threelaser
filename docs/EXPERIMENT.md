# 焊接图像无监督学习实验文档

## 1. 任务目标

从焊接视频中抽取图片，训练一个不依赖人工标签的无监督模型。模型学习常见焊接图像的结构，再用重建误差定位可能异常的帧或区域。

## 2. 数据处理

- 原始视频：项目根目录下 3 个 `.mp4` 文件。
- 抽帧工具：FFmpeg。
- 抽帧频率：`5 FPS`。
- 已抽帧数量：`312` 张。
- 划分方式：随机划分为训练、验证、测试。
- 当前划分：训练 `249`，验证 `31`，测试 `32`。

注意：相邻视频帧高度相似。正式报告或生产验证时，建议按视频、工件批次或焊接工艺参数划分，减少相邻帧泄漏。

## 3. 模型方案

当前基线使用卷积自编码器：

- 输入：灰度焊接图，缩放到 `256x256`。
- 编码器：多层卷积下采样，压缩图像结构。
- 解码器：转置卷积上采样，重建输入图像。
- 损失函数：L1 重建损失。
- 异常分数：输入图与重建图的平均 L1 差异。

这个方案适合先做无标签基线，因为它简单、可解释，并且能输出像素级误差热力图。

## 4. 训练配置

- 环境：Python `.venv`。
- 框架：PyTorch。
- 设备：本次训练使用 `CPU`。
- Epoch：`20`。
- Batch size：`16`。
- Optimizer：AdamW。
- 最佳模型：`runs/autoencoder_baseline/best.pt`。

## 5. 当前结果

- 第 1 轮验证 L1：`0.1693765833`。
- 第 20 轮验证 L1：`0.0260189811`。
- 测试集阈值：第 95 分位重建误差，`0.0384631008`。
- 测试集最高异常候选：
  - `data\frames\lasercmt3.5x\lasercmt3.5x_000102.jpg`: `0.0400975086`
  - `data\frames\lasercmt3.5x\lasercmt3.5x_000004.jpg`: `0.0387192816`
  - `data\frames\lasercmt3.5x\lasercmt3.5x_000080.jpg`: `0.0382535011`

## 6. 输出目录

- 抽帧图片：`data/frames/`
- 数据划分：`data/splits/`
- 训练历史：`runs/autoencoder_baseline/history.json`
- 模型权重：`runs/autoencoder_baseline/best.pt`
- 测试分数：`runs/autoencoder_baseline/eval/scores.json`
- 热力图：`runs/autoencoder_baseline/eval/visualizations/`

## 7. 下一步优化

- 使用按视频或批次的严格划分，验证泛化能力。
- 增加 SSIM、LPIPS 或 Patch-level 误差作为异常分数。
- 尝试 SimCLR、MAE、PatchCore、Deep SVDD 等无监督/自监督方法。
- 如果能获得少量缺陷标注，用 ROC-AUC、PR-AUC、Recall@K 验证异常分数是否真正有效。
