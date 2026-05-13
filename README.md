# Weld Image Unsupervised Learning

这个项目把焊接视频切成图片，再用无监督卷积自编码器学习图像结构。训练完成后，模型用“重建误差”给每张图片打分：误差越高，说明该帧越不像训练集中学到的常见焊接画面，可作为缺陷、工况变化或异常片段的候选。

## 项目逻辑

1. `lasercmt*.mp4` 原始视频放在项目根目录。
2. `scripts/extract_frames.py` 使用 FFmpeg 按固定 FPS 抽帧到 `data/frames/`。
3. `scripts/make_splits.py` 生成 `train/val/test` 划分。
4. `scripts/train_autoencoder.py` 训练无监督自编码器，只用图片本身作为目标，不需要人工标签。
5. `scripts/evaluate_reconstruction.py` 输出每帧重建误差和可视化热力图。

## 环境配置

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果机器有 NVIDIA CUDA，可按 PyTorch 官网选择对应 CUDA 版本安装；当前 `requirements.txt` 默认会安装通用版本。

## 一键流程

```powershell
$env:PYTHONPATH="src"
python scripts/extract_frames.py --config configs/default.yaml
python scripts/make_splits.py --config configs/default.yaml
python scripts/train_autoencoder.py --config configs/default.yaml
python scripts/evaluate_reconstruction.py --checkpoint runs/autoencoder_baseline/best.pt
```

## 输出说明

- `data/frames/`: 从视频抽出的图片。
- `data/splits/train.txt`: 训练图片列表。
- `runs/autoencoder_baseline/best.pt`: 验证集误差最低的模型。
- `runs/autoencoder_baseline/history.json`: 每轮训练损失。
- `runs/autoencoder_baseline/eval/scores.json`: 每张测试图片的重建误差。
- `runs/autoencoder_baseline/eval/visualizations/`: 输入图、重建图、误差热力图。

## 配置项

主要参数在 `configs/default.yaml`：

- `data.extract_fps`: 抽帧 FPS。默认 `5`，三段约 20 秒视频会得到约 300 张图。
- `data.image_size`: 训练输入尺寸。默认 `256`。
- `train.epochs`: 训练轮数。默认 `20`。
- `train.batch_size`: 批大小。显存不足时调小。
- `eval.threshold_percentile`: 异常阈值分位数。默认用测试集重建误差的第 95 分位。

## 实验注意

这是无监督基线，不需要缺陷标签。后续如果有人工标注，可以把重建误差作为异常分数，再用 ROC-AUC、PR-AUC、召回率等指标评估。由于相邻帧高度相似，正式实验建议按视频、焊接批次或工件编号划分训练/测试，避免相邻帧泄漏导致指标虚高。
