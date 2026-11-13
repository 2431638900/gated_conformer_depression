# 基于门控多尺度 Conformer 的语音抑郁症检测

这是论文 *Hierarchical Self-supervised Speech Representation Learning for Depression Detection Based on Gated Multi-Scale Conformer Framework* 的配套代码。

模型采用冻结的 HuBERT 骨干网络、可学习的层次化层聚合、多尺度抑郁症特征提取器、门控多尺度 Conformer（6 层，宽度 256，8 个注意力头）以及统计注意力池化。跨语种迁移到中文 MODMA 时使用轻量语言适配器。

## 环境配置

建议使用 Python 3.9+ 和 CUDA GPU（原实验使用 PyTorch 2.1 与 CUDA 12）。

```bash
pip install -r requirements.txt
# 如需指定 HuBERT 模型：
#   export HUBERT_MODEL=facebook/hubert-base-ls960
# 或指向本地目录：
#   export HUBERT_MODEL=/path/to/hubert-base-ls960
```

## 准备官方 DAIC-WOZ 划分

本仓库不包含 DAIC-WOZ 或 MODMA 语音，需自行按原协议获取。

```bash
python scripts/prepare_official_split.py --daic-woz-root /path/to/DAIC-WOZ
```

脚本使用 `splits/avec2017/` 中的 AVEC 2017 官方被试列表（107 / 35 / 47），将片段写入 `data/official_split/`。完整运行后应得到 4,589 / 983 / 984 条片段。

## 训练与测试（冻结 HuBERT）

```bash
# 训练并测试
bash scripts/train_freeze.sh

# 或仅测试已有权重
python scripts/03_test_model.py --checkpoint best_model.pth
```

测试前请将训练得到的权重放在 `checkpoints/best_model.pth`。默认训练设置会冻结全部 HuBERT 参数（`--finetune-layers 0`）。

## 其他实验

```bash
python scripts/04_ablation_study.py
python scripts/05_baseline_comparison.py
python scripts/09_prepare_modma.py --modma-root /path/to/audio_lanzhou_2015
python scripts/10_cross_lingual.py
python scripts/11_train_language_adapter.py
```

与论文表格对应的实验记录在 `results/` 中。

## 默认训练设置

AdamW，学习率 \(2\times 10^{-4}\)，权重衰减 \(10^{-3}\)，8 个 epoch 预热，最多 80 个 epoch，早停耐心值 10，Focal Loss \(\gamma=2\)，MDD 类别权重 1.70，标签平滑 0.1，Mixup \(\alpha=0.2\)，SpecAugment 4\(\times\)35，梯度累积 4，梯度裁剪 1.0，EMA 0.999，dropout 0.35，DropPath 0.1，随机种子 42。

## 许可证

代码以 MIT 许可证发布。DAIC-WOZ 与 MODMA 仍遵循各自的数据使用协议。
