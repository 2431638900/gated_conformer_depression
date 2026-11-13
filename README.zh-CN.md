# 基于门控多尺度 Conformer 的语音抑郁症检测

这是论文 *Hierarchical Self-supervised Speech Representation Learning for Depression Detection Based on Gated Multi-Scale Conformer Framework* 的配套代码。本目录从原项目 `conformer_depression_v2` 整理而来，供公开代码仓使用。

主要改动：训练默认改为冻结全部 HuBERT；数据按 AVEC 2017 官方 107 / 35 / 47 划分准备，不再使用后来误跑出的 132 / 28 / 29 随机划分；去掉本机硬编码路径；并附上与论文表格对应的实验日志。

## 能复现什么，不能复现什么

论文中的 **DAIC-WOZ 测试集** 主结果（片段级 Acc 94.92%，F1 0.9169，AUC 0.9874，混淆矩阵 658/21/29/276）来自 **2026 年 5 月 18 日** 的冻结 HuBERT 实验（可训练参数 18.44M，seed=42）。把该次检查点复制为 `checkpoints/best_model.pth`，在官方测试集上评估，即可对上这些数字。

消融和层聚合表里的 **完整模型验证集** 数字（F1 0.9140，Acc 94.81%，AUC 0.9837）来自 **2026 年 4 月 1 日** 微调 HuBERT 最后 5 层的实验（53.88M）。同一次冻结实验自己的验证集是 F1 0.9264、Acc 95.52%、AUC 0.9837，论文表中没有采用。因此按默认冻结设置重新训练，不会得到 0.9140 这一行。详见 `results/EXPERIMENT_LOG.md`。

音频数据需自行按原协议获取，本仓库不包含 DAIC-WOZ 或 MODMA 语音。

## 使用步骤

```bash
pip install -r requirements.txt

# 按官方划分准备片段（完整成功时应得到 4589 / 983 / 984）
python scripts/prepare_official_split.py --daic-woz-root /path/to/DAIC-WOZ

# 冻结 HuBERT 训练并测试
bash scripts/train_freeze.sh
# 或只测试已有权重
python scripts/03_test_model.py --checkpoint best_model.pth
```

请不要运行 `scripts/legacy/01_resplit_dataset.py`，那会破坏官方划分。

原项目中 5 月 18 日的权重文件在：

`conformer_depression_v2/checkpoints/best_model_ft0_20260518_015803_w1.70_f1at050.9264.pth`

请把它复制到本目录的 `checkpoints/best_model.pth`。不要误用原文件夹里 4 月 1 日的 `best_model.pth`。
