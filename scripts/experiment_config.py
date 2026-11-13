# -*- coding: utf-8 -*-
"""
实验公共配置模块
确保所有实验（消融、对照、模块贡献度）使用与原始训练完全一致的配置。

配置来源: 02_train_model.py 中 V10.12 的 train_config 和 model_arch_config
"""

import sys
import os
import random
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import pandas as pd

from config.config import Config, config


# ===================== 路径配置 =====================
PROJECT_ROOT = Config.PROJECT_ROOT

# 训练数据路径（与02_train_model.py一致：使用分层划分数据）
STRATIFIED_DIR = Config.get_split_dir()
TRAIN_CSV = STRATIFIED_DIR / 'train' / 'augmented_train.csv'
VAL_CSV = STRATIFIED_DIR / 'val' / 'augmented_val.csv'

# MODMA 数据路径
MODMA_CSV = PROJECT_ROOT / 'data' / 'modma_processed' / 'modma_samples.csv'

# 输出路径
CHECKPOINT_DIR = Config.CHECKPOINT_DIR
LOG_DIR = Config.LOG_DIR
PLOT_DIR = Config.PLOT_DIR

# ===================== 模型架构配置 =====================
# 与 02_train_model.py V10.12 中 model_arch_config 完全一致
MODEL_ARCH = {
    'pretrain_model_type': config.PRETRAIN_MODEL_TYPE,
    'pretrain_model_name': config.PRETRAIN_MODEL_NAME,
    'pretrain_cache_dir': config.PRETRAIN_CACHE_DIR,
    'd_model': 256,                 # V10: 160→256
    'num_conformer_layers': 6,      # V10: 4→6
    'num_attention_heads': 8,       # V10: 5→8
    'conv_kernel_size': 31,         # V10: 15→31
    'feed_forward_dim': 1024,       # V10: 640→1024
    'num_classes': 2,
    'use_gating': True,
    'use_depression_extractor': True,
    'drop_path_rate': 0.1,          # V10.12: Stochastic Depth
}

# ===================== 训练超参数配置 =====================
# 与 02_train_model.py V10.12 中 train_config 完全一致
TRAIN_CONFIG = {
    'batch_size': 8,
    'audio_duration': 30,
    'sample_rate': 16000,
    'target_length': 16000 * 30,  # 480000
    'learning_rate': 2e-4,        # V10.11: 恢复2e-4
    'num_epochs': 80,             # V10.5: 60→80
    'warmup_epochs': 8,           # V10.4: 5→8
    'early_stopping_patience': 10,  # V10.11: 15→10
    'dropout': 0.35,              # V10.4: 0.45→0.35
    'weight_decay': 1e-3,         # V10.1: 5e-4→1e-3
    'label_smoothing': 0.1,       # V10.8: 0.05→0.1
    'gradient_accumulation_steps': 4,
    'max_grad_norm': 1.0,
    'pretrain_freeze': True,      # 消融实验中冻结HuBERT
    # Focal Loss
    'use_focal_loss': True,
    'focal_gamma': 2.0,           # V10.1: 1.5→2.0
}

# ===================== 随机种子 =====================
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """设置全局随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置: {seed}")


def get_device():
    """获取计算设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"计算设备: {device}")
    return device


# ===================== 数据加载 =====================
def load_train_val_data():
    """加载与训练时完全一致的训练集和验证集DataFrame"""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"训练数据不存在: {TRAIN_CSV}\n"
            f"请先运行 python scripts/prepare_official_split.py --daic-woz-root <DAIC-WOZ路径>"
        )
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"验证数据不存在: {VAL_CSV}")

    train_df = pd.read_csv(str(TRAIN_CSV))
    val_df = pd.read_csv(str(VAL_CSV))

    print(f"训练集: {len(train_df)} 样本 "
          f"(正常={sum(train_df['PHQ8_Binary']==0)}, 抑郁={sum(train_df['PHQ8_Binary']==1)})")
    print(f"验证集: {len(val_df)} 样本 "
          f"(正常={sum(val_df['PHQ8_Binary']==0)}, 抑郁={sum(val_df['PHQ8_Binary']==1)})")
    return train_df, val_df


def create_dataloaders(train_df, val_df):
    """创建与训练时一致的DataLoader"""
    from data.dataset import DepressionDataset, collate_fn
    from torch.utils.data import DataLoader

    target_length = TRAIN_CONFIG['target_length']
    sample_rate = TRAIN_CONFIG['sample_rate']
    batch_size = TRAIN_CONFIG['batch_size']

    train_dataset = DepressionDataset(
        train_df,
        sample_rate=sample_rate,
        target_length=target_length,
        mode='random_crop',
        cache_audio=True,
    )
    val_dataset = DepressionDataset(
        val_df,
        sample_rate=sample_rate,
        target_length=target_length,
        mode='pad',
        cache_audio=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=False,
    )
    return train_loader, val_loader


# ===================== 模型构建 =====================
def build_full_model(device, dropout_override=None, drop_path_override=None):
    """构建与训练时完全一致的完整模型（V10.12）"""
    from models.depression_model import DepressionDetectionModel

    dropout = dropout_override if dropout_override is not None else TRAIN_CONFIG['dropout']
    drop_path = drop_path_override if drop_path_override is not None else MODEL_ARCH['drop_path_rate']

    model = DepressionDetectionModel(
        pretrain_model_type=MODEL_ARCH['pretrain_model_type'],
        pretrain_model_name=MODEL_ARCH['pretrain_model_name'],
        pretrain_cache_dir=MODEL_ARCH['pretrain_cache_dir'],
        pretrain_freeze=TRAIN_CONFIG['pretrain_freeze'],
        pretrain_layers_to_finetune=0,
        use_depression_extractor=MODEL_ARCH['use_depression_extractor'],
        d_model=MODEL_ARCH['d_model'],
        num_conformer_layers=MODEL_ARCH['num_conformer_layers'],
        num_attention_heads=MODEL_ARCH['num_attention_heads'],
        conv_kernel_size=MODEL_ARCH['conv_kernel_size'],
        feed_forward_dim=MODEL_ARCH['feed_forward_dim'],
        dropout=dropout,
        num_classes=MODEL_ARCH['num_classes'],
        use_gating=MODEL_ARCH['use_gating'],
        drop_path_rate=drop_path,
    ).to(device)
    return model


def build_model_no_dep_extractor(device):
    """构建去掉抑郁特征提取器的模型"""
    from models.depression_model import DepressionDetectionModel

    model = DepressionDetectionModel(
        pretrain_model_type=MODEL_ARCH['pretrain_model_type'],
        pretrain_model_name=MODEL_ARCH['pretrain_model_name'],
        pretrain_cache_dir=MODEL_ARCH['pretrain_cache_dir'],
        pretrain_freeze=TRAIN_CONFIG['pretrain_freeze'],
        pretrain_layers_to_finetune=0,
        use_depression_extractor=False,
        d_model=MODEL_ARCH['d_model'],
        num_conformer_layers=MODEL_ARCH['num_conformer_layers'],
        num_attention_heads=MODEL_ARCH['num_attention_heads'],
        conv_kernel_size=MODEL_ARCH['conv_kernel_size'],
        feed_forward_dim=MODEL_ARCH['feed_forward_dim'],
        dropout=TRAIN_CONFIG['dropout'],
        num_classes=MODEL_ARCH['num_classes'],
        use_gating=MODEL_ARCH['use_gating'],
        drop_path_rate=MODEL_ARCH['drop_path_rate'],
    ).to(device)
    return model


def load_trained_model(device):
    """加载已训练好的最佳模型（V10.12 checkpoint）"""
    from models.depression_model import DepressionDetectionModel

    ckpt_path = None
    for name in ('best_model.pth', 'best_model_ft0.pth'):
        candidate = CHECKPOINT_DIR / name
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        raise FileNotFoundError(
            f"模型文件不存在: 请将论文对应的冻结检查点放到 {CHECKPOINT_DIR}/best_model.pth "
            f"(或 best_model_ft0.pth)，参见 checkpoints/README.md"
        )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    arch = ckpt.get('model_arch_config', MODEL_ARCH)
    cfg = ckpt.get('config', TRAIN_CONFIG)

    model = DepressionDetectionModel(
        pretrain_model_type=arch.get('pretrain_model_type', 'hubert'),
        pretrain_model_name=arch.get('pretrain_model_name', MODEL_ARCH['pretrain_model_name']),
        pretrain_cache_dir=arch.get('pretrain_cache_dir', MODEL_ARCH['pretrain_cache_dir']),
        pretrain_freeze=True,
        pretrain_layers_to_finetune=0,
        use_depression_extractor=arch.get('use_depression_extractor', True),
        d_model=arch.get('d_model', 256),
        num_conformer_layers=arch.get('num_conformer_layers', 6),
        num_attention_heads=arch.get('num_attention_heads', 8),
        conv_kernel_size=arch.get('conv_kernel_size', 31),
        feed_forward_dim=arch.get('feed_forward_dim', 1024),
        dropout=cfg.get('dropout', 0.35),
        num_classes=arch.get('num_classes', 2),
        use_gating=arch.get('use_gating', True),
        drop_path_rate=arch.get('drop_path_rate', 0.1),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    metrics = ckpt.get('metrics', {})
    threshold = metrics.get('threshold', 0.5)
    epoch = ckpt.get('epoch', '?')
    f1 = metrics.get('f1', 0)
    print(f"模型加载成功 | Epoch {epoch} | DAIC-WOZ F1={f1:.4f} | 阈值={threshold:.3f}")
    return model, threshold


# ===================== 训练工具 =====================
def get_criterion(use_focal=True):
    """获取与训练时一致的损失函数"""
    import torch.nn as nn
    from utils.focal_loss import FocalLoss

    if use_focal:
        # 动态计算类别权重（与02_train_model.py一致）
        train_df = pd.read_csv(str(TRAIN_CSV))
        class_counts = train_df['PHQ8_Binary'].value_counts().sort_index()
        total = len(train_df)
        raw_weights = total / (len(class_counts) * class_counts.values)
        class_weights = raw_weights / raw_weights[0]
        return FocalLoss(
            alpha=class_weights.tolist(),
            gamma=TRAIN_CONFIG['focal_gamma'],
            label_smoothing=TRAIN_CONFIG['label_smoothing'],
        )
    else:
        return nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG['label_smoothing'])


def get_optimizer_and_scheduler(model, num_epochs=None, lr_override=None):
    """获取与训练时一致的优化器和调度器"""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

    lr = lr_override if lr_override is not None else TRAIN_CONFIG['learning_rate']
    num_epochs = num_epochs or TRAIN_CONFIG['num_epochs']
    warmup_epochs = TRAIN_CONFIG['warmup_epochs']

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=TRAIN_CONFIG['weight_decay'])

    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    return optimizer, scheduler
