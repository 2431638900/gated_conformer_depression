# -*- coding: utf-8 -*-
"""
LanguageAdapter 独立训练脚本 (服务器/本地通用)

与 10_cross_lingual.py 不同：本脚本 *不做 5 折 CV*，而是训练一个可部署的
adapter 权重，保存至 checkpoints/language_adapter.pth。

用法 (在 conformer_depression_v2/ 根目录执行):
    python scripts/14_train_language_adapter.py
    python scripts/14_train_language_adapter.py --full        # 全量数据, 无 val
    python scripts/14_train_language_adapter.py --epochs 60

训练完成后将 checkpoints/language_adapter.pth (≈1MB) 下载到本地同名路径,
系统 model-service 启动时自动加载。
"""

import argparse
import copy
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# 本脚本位于 conformer_depression_v2/scripts/, 向上一级即 PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import config  # noqa: E402
from data.dataset import DepressionDataset  # noqa: E402
from models.depression_model import DepressionDetectionModel  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


BASE_CKPT = PROJECT_ROOT / 'checkpoints' / 'best_model.pth'
ADAPTER_CKPT = PROJECT_ROOT / 'checkpoints' / 'language_adapter.pth'
MODMA_CSV = PROJECT_ROOT / 'data' / 'modma_processed' / 'modma_samples.csv'

SEED = 42
ADAPTER_CONFIG = {'bottleneck_dim': 128, 'dropout': 0.1}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_base_model(device: torch.device) -> DepressionDetectionModel:
    if not BASE_CKPT.exists():
        raise FileNotFoundError(f'基础模型不存在: {BASE_CKPT}')
    print(f'[Base] 加载基础模型: {BASE_CKPT.name}')
    ckpt = torch.load(str(BASE_CKPT), map_location=device, weights_only=False)
    arch = ckpt.get('model_arch_config', {})
    cfg = ckpt.get('config', {})

    model = DepressionDetectionModel(
        pretrain_model_type=arch.get('pretrain_model_type', 'hubert'),
        pretrain_model_name=config.PRETRAIN_MODEL_NAME,
        pretrain_cache_dir=config.PRETRAIN_CACHE_DIR,
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
        adapter_config=ADAPTER_CONFIG,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.language_adapter.parameters():
        p.requires_grad = True
    model._has_trainable_pretrain = False  # noqa: SLF001

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Base] 可训练参数 (Adapter): {n_trainable/1e3:.1f}K')
    return model


def split_modma(df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = df.groupby('Original_ID')['PHQ8_Binary'].first().reset_index()
    train_sid, val_sid = train_test_split(
        subjects['Original_ID'].tolist(),
        test_size=val_ratio,
        stratify=subjects['PHQ8_Binary'].tolist(),
        random_state=SEED,
    )
    train_df = df[df['Original_ID'].isin(train_sid)].reset_index(drop=True)
    val_df = df[df['Original_ID'].isin(val_sid)].reset_index(drop=True)
    return train_df, val_df


def build_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    sr = config.SAMPLE_RATE
    tl = 30 * sr
    ds = DepressionDataset(df, sample_rate=sr, target_length=tl, mode='pad')

    def _collate(batch):
        audio = torch.stack([b['audio'] for b in batch])
        labels = torch.cat([b['label'] for b in batch])
        return {'audio': audio, 'label': labels,
                'participant_id': [b['participant_id'] for b in batch]}

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=_collate, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        with torch.cuda.amp.autocast():
            logits = model(audio, use_adapter=True)
        probs = torchF.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
    if len(set(all_labels)) < 2:
        return 0.0, 0.5, all_labels, all_probs
    preds = (np.array(all_probs) > 0.5).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return f1, auc, all_labels, all_probs


def find_optimal_threshold(labels, probs) -> Tuple[float, float]:
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.05, 0.96, 0.01):
        preds = (np.array(probs) > th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)
    return best_th, best_f1


def train(args: argparse.Namespace) -> None:
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Env] 工作目录: {PROJECT_ROOT}')
    print(f'[Env] 设备: {device}')

    if not MODMA_CSV.exists():
        raise FileNotFoundError(
            f'MODMA 数据未预处理: {MODMA_CSV}\n'
            f'请先运行: python scripts/04_prepare_modma.py'
        )

    modma_df = pd.read_csv(str(MODMA_CSV))
    print(f'[Data] MODMA 样本={len(modma_df)}, 受试者={modma_df["Original_ID"].nunique()}')

    if args.full:
        train_df, val_df = modma_df, modma_df
        print('[Data] 使用全量数据训练, val = train (仅监控)')
    else:
        train_df, val_df = split_modma(modma_df, val_ratio=args.val_ratio)
        print(f'[Data] train={len(train_df)} ({train_df["Original_ID"].nunique()}人), '
              f'val={len(val_df)} ({val_df["Original_ID"].nunique()}人)')

    train_loader = build_loader(train_df, args.batch_size, shuffle=True)
    val_loader = build_loader(val_df, args.batch_size, shuffle=False)

    model = load_base_model(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weight = torch.tensor([1.0, 1.5], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    best_threshold = 0.5
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(audio, use_adapter=True)
                loss = criterion(logits, labels) / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += float(loss.item()) * args.accum_steps

        if (step + 1) % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()

        val_f1, val_auc, val_labels, val_probs = evaluate(model, val_loader, device)
        opt_th, opt_f1 = find_optimal_threshold(val_labels, val_probs)

        elapsed = time.time() - t0
        print(f'[Epoch {epoch:3d}/{args.epochs}] '
              f'loss={epoch_loss/max(len(train_loader),1):.4f} '
              f'val_f1@0.5={val_f1:.4f} val_auc={val_auc:.4f} '
              f'val_f1@opt={opt_f1:.4f} opt_th={opt_th:.2f} '
              f'elapsed={elapsed/60:.1f}min')

        if opt_f1 > best_f1:
            best_f1 = opt_f1
            best_epoch = epoch
            best_threshold = opt_th
            best_state = copy.deepcopy(model.language_adapter.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'[Train] 早停于 Epoch {epoch} (best={best_epoch})')
                break

    if best_state is None:
        best_state = copy.deepcopy(model.language_adapter.state_dict())
        best_epoch = args.epochs
        best_threshold = 0.5

    save_payload = {
        'adapter_state_dict': best_state,
        'modma_threshold': float(best_threshold),
        'adapter_config': ADAPTER_CONFIG,
        'training_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_epoch': best_epoch,
            'best_val_f1': round(float(best_f1), 4),
            'n_train_samples': int(len(train_df)),
            'n_val_samples': int(len(val_df)),
            'epochs_total': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'accum_steps': args.accum_steps,
            'full_data': bool(args.full),
            'host': os.environ.get('HOSTNAME', os.environ.get('COMPUTERNAME', 'unknown')),
        },
    }
    ADAPTER_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_payload, str(ADAPTER_CKPT))
    size_kb = ADAPTER_CKPT.stat().st_size / 1024
    print('\n' + '=' * 60)
    print(f'[Saved] {ADAPTER_CKPT}')
    print(f'[Saved] 文件大小: {size_kb:.1f} KB')
    print(f'[Saved] Best Epoch={best_epoch} Val F1={best_f1:.4f} MODMA 阈值={best_threshold:.3f}')
    print('=' * 60)
    print('\n下一步: 将 language_adapter.pth 下载到本地同名路径后重启 model-service。')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='训练 LanguageAdapter (MODMA)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accum-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--full', action='store_true', help='使用全量 MODMA 训练, 不划分验证集')
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
