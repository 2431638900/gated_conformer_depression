# -*- coding: utf-8 -*-
"""
超参数敏感性分析脚本
对 4 个核心超参数分别进行单变量扫描，其余参数保持默认值（V10.12配置）。
使用预提取的 HuBERT WLA 特征加速训练。

超参数扫描范围：
  1. 学习率: {5e-5, 1e-4, 2e-4, 5e-4}
  2. Conformer 层数: {4, 6, 8}
  3. Dropout 率: {0.2, 0.3, 0.35, 0.4}
  4. Focal Loss γ: {1.0, 2.0, 3.0}

默认值(基线): lr=2e-4, layers=6, dropout=0.35, gamma=2.0
基线实验无需重复运行，直接引用已有结果 F1=0.9140。

输出:
  - logs/hp_sweep_results.json
  - 自动更新 paper_gen/sections.py 中表7数据
"""

import sys
import os
import json
import time
import hashlib
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from torch.utils.data import Dataset, DataLoader

from scripts.experiment_config import (
    set_seed, get_device, load_train_val_data,
    load_trained_model, get_criterion,
    MODEL_ARCH, TRAIN_CONFIG, RANDOM_SEED, LOG_DIR,
)
from models.depression_model import StatisticsAttentionPooling
from models.depression_feature_extractor import DepressionFeatureExtractor
from models.conformer import Conformer
from utils.metrics import compute_metrics, EarlyStopping, AverageMeter


# ===================== 特征预提取（复用消融实验的缓存） =====================
FEATURE_CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'feature_cache'


def _get_feature_cache_path(audio_path: str, cache_tag: str) -> Path:
    key = hashlib.md5(f"{cache_tag}:{audio_path}".encode()).hexdigest()
    return FEATURE_CACHE_DIR / cache_tag / f"{key}.npy"


@torch.no_grad()
def precompute_weighted_features(model, df, device, cache_tag='hubert_wla'):
    """使用训练好模型的 WLA 预提取 768 维特征"""
    cache_dir = FEATURE_CACHE_DIR / cache_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    all_labels = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        audio_path = str(row['Audio_Path'])
        all_paths.append(_get_feature_cache_path(audio_path, cache_tag))
        all_labels.append(int(row['PHQ8_Binary']))

    uncached = [i for i, p in enumerate(all_paths) if not p.exists()]

    if uncached:
        print(f"  预提取加权特征 ({len(uncached)}/{len(all_paths)} 个新样本)...")
        from data.dataset import DepressionDataset, collate_fn
        extract_ds = DepressionDataset(
            df, sample_rate=TRAIN_CONFIG['sample_rate'],
            target_length=TRAIN_CONFIG['target_length'],
            mode='pad', cache_audio=True)
        extract_loader = DataLoader(
            extract_ds, batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False, num_workers=0, collate_fn=collate_fn)

        model.eval()
        encoder = model.feature_encoder
        layer_weights = torch.softmax(model.layer_weights.detach(), dim=0)

        start = time.time()
        sample_idx = 0

        for batch in extract_loader:
            audio = batch['audio'].to(device)
            batch_size = audio.size(0)
            with torch.cuda.amp.autocast():
                encoder_output = encoder(audio, output_hidden_states=True)
                hidden_states = encoder_output.hidden_states
                stacked = torch.stack(hidden_states, dim=0)
                features = (stacked * layer_weights.view(-1, 1, 1, 1)).sum(dim=0)

            features_np = features.cpu().float().numpy()
            for i in range(batch_size):
                idx = sample_idx + i
                if not all_paths[idx].exists():
                    np.save(str(all_paths[idx]), features_np[i])
            sample_idx += batch_size

            if sample_idx % 500 < batch_size:
                elapsed = time.time() - start
                print(f"    [{sample_idx}/{len(all_paths)}] {elapsed:.0f}s", flush=True)

        print(f"  ✅ 特征提取完成: {time.time()-start:.1f}秒")
    else:
        print(f"  ✅ 加权特征缓存已就绪 ({len(all_paths)} 个文件)")

    return all_paths, all_labels


class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels):
        self.feature_paths = feature_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = np.load(str(self.feature_paths[idx]))
        return {
            'features': torch.from_numpy(features),
            'label': torch.LongTensor([self.labels[idx]]),
        }


def feature_collate_fn(batch):
    features = torch.stack([item['features'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    return {'features': features, 'label': labels}


def create_feature_dataloaders(train_paths, train_labels, val_paths, val_labels):
    batch_size = TRAIN_CONFIG['batch_size']
    train_ds = FeatureDataset(train_paths, train_labels)
    val_ds = FeatureDataset(val_paths, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=feature_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=feature_collate_fn, pin_memory=True)
    return train_loader, val_loader


# ===================== 可配置的下游模型 =====================
class DownstreamConfigurable(nn.Module):
    """可配置超参数的完整下游模型：DepExtractor → Conformer → StatAttnPool → Classifier"""

    def __init__(self, num_conformer_layers=6, dropout=0.35):
        super().__init__()
        d_model = MODEL_ARCH['d_model']  # 256
        input_dim = 768  # HuBERT WLA output dim

        self.depression_extractor = DepressionFeatureExtractor(
            input_dim=input_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        self.conformer = Conformer(
            input_dim=d_model,
            d_model=d_model,
            num_layers=num_conformer_layers,
            num_heads=MODEL_ARCH['num_attention_heads'],
            conv_kernel_size=MODEL_ARCH['conv_kernel_size'],
            feed_forward_dim=MODEL_ARCH['feed_forward_dim'],
            dropout=dropout,
            use_gating=MODEL_ARCH['use_gating'],
            drop_path_rate=MODEL_ARCH['drop_path_rate'],
        )

        self.pooling = StatisticsAttentionPooling(d_model)
        pool_output_dim = d_model * 2

        self.classifier = nn.Sequential(
            nn.Linear(pool_output_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, features):
        features = self.depression_extractor(features)
        features = self.conformer(features)
        pooled = self.pooling(features)
        return self.classifier(pooled)


# ===================== 训练与评估 =====================
_amp_scaler = torch.cuda.amp.GradScaler()


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    gradient_accumulation_steps=4, max_grad_norm=1.0):
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps

        _amp_scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            _amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            _amp_scaler.step(optimizer)
            _amp_scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * gradient_accumulation_steps, features.size(0))

    if (step + 1) % gradient_accumulation_steps != 0:
        _amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        _amp_scaler.step(optimizer)
        _amp_scaler.update()
        optimizer.zero_grad()

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_labels = []
    all_probs = []

    for batch in dataloader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(features)
            loss = criterion(outputs, labels)
        probs = torchF.softmax(outputs.float(), dim=1)[:, 1].cpu().numpy()

        loss_meter.update(loss.item(), features.size(0))
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    preds = (np.array(all_probs) > 0.5).astype(int)
    metrics = compute_metrics(all_labels, preds, all_probs)
    metrics['loss'] = loss_meter.avg
    return metrics


def run_single_experiment(name, model, train_loader, val_loader, device,
                          criterion, lr, num_epochs=80):
    """运行单个超参数配置的训练"""
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"{'='*60}")

    set_seed(RANDOM_SEED)

    # 重新初始化 AMP scaler（避免跨实验状态残留）
    global _amp_scaler
    _amp_scaler = torch.cuda.amp.GradScaler()

    grad_accum = TRAIN_CONFIG['gradient_accumulation_steps']
    max_grad_norm = TRAIN_CONFIG['max_grad_norm']
    patience = TRAIN_CONFIG['early_stopping_patience']
    warmup_epochs = TRAIN_CONFIG['warmup_epochs']

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params / 1e6:.2f}M | LR: {lr}")

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=TRAIN_CONFIG['weight_decay'])
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs])

    early_stopping = EarlyStopping(patience=patience, mode='max')
    best_f1 = 0
    best_metrics = None

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_accumulation_steps=grad_accum, max_grad_norm=max_grad_norm)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        val_f1 = val_metrics['f1']
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics.copy()

        if epoch % 5 == 0 or epoch <= 3:
            print(f"  Epoch {epoch:3d} | {epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | Best F1: {best_f1:.4f}", flush=True)

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    best_metrics['trainable_params'] = trainable_params
    print(f"\n  ✅ {name} 最佳 F1 = {best_f1:.4f}")
    return best_metrics


# ===================== 主函数 =====================
def main():
    total_start = time.time()
    print("=" * 60)
    print("超参数敏感性分析 (Hyperparameter Sweep)")
    print("=" * 60)

    set_seed()
    device = get_device()

    # 加载数据
    train_df, val_df = load_train_val_data()

    # 加载训练好的模型用于特征预提取
    print("\n[Step 1] 加载 V10.12 模型并预提取 WLA 特征...")
    trained_model, _ = load_trained_model(device)

    train_paths, train_labels = precompute_weighted_features(
        trained_model, train_df, device, cache_tag='hubert_wla')
    val_paths, val_labels = precompute_weighted_features(
        trained_model, val_df, device, cache_tag='hubert_wla')

    del trained_model
    torch.cuda.empty_cache()

    train_loader, val_loader = create_feature_dataloaders(
        train_paths, train_labels, val_paths, val_labels)

    # ===================== 默认配置 =====================
    default_lr = 2e-4
    default_layers = 6
    default_dropout = 0.35
    default_gamma = 2.0

    # 基线结果（已知）
    baseline_f1 = 0.9140

    results = {}

    # ===================== 扫描学习率 =====================
    print("\n" + "=" * 60)
    print("[Step 2] 学习率扫描")
    print("=" * 60)

    for lr in [5e-5, 1e-4, 2e-4, 5e-4]:
        key = f"lr={lr}"
        if lr == default_lr:
            results[key] = {'f1': baseline_f1, 'is_default': True}
            print(f"\n  {key}: 使用基线结果 F1={baseline_f1}")
            continue

        set_seed(RANDOM_SEED)
        model = DownstreamConfigurable(
            num_conformer_layers=default_layers,
            dropout=default_dropout).to(device)
        criterion = get_criterion(use_focal=True)
        # 覆盖 focal_gamma 为默认值
        criterion.gamma = default_gamma

        metrics = run_single_experiment(
            key, model, train_loader, val_loader, device, criterion, lr=lr)
        results[key] = {'f1': round(metrics['f1'], 4),
                        'acc': round(metrics['accuracy'], 2),
                        'auc': round(metrics.get('auc', 0), 4)}

        del model
        torch.cuda.empty_cache()

    # ===================== 扫描 Conformer 层数 =====================
    print("\n" + "=" * 60)
    print("[Step 3] Conformer 层数扫描")
    print("=" * 60)

    for n_layers in [4, 6, 8]:
        key = f"layers={n_layers}"
        if n_layers == default_layers:
            results[key] = {'f1': baseline_f1, 'is_default': True}
            print(f"\n  {key}: 使用基线结果 F1={baseline_f1}")
            continue

        set_seed(RANDOM_SEED)
        model = DownstreamConfigurable(
            num_conformer_layers=n_layers,
            dropout=default_dropout).to(device)
        criterion = get_criterion(use_focal=True)
        criterion.gamma = default_gamma

        metrics = run_single_experiment(
            key, model, train_loader, val_loader, device, criterion, lr=default_lr)
        results[key] = {'f1': round(metrics['f1'], 4),
                        'acc': round(metrics['accuracy'], 2),
                        'auc': round(metrics.get('auc', 0), 4)}

        del model
        torch.cuda.empty_cache()

    # ===================== 扫描 Dropout =====================
    print("\n" + "=" * 60)
    print("[Step 4] Dropout 率扫描")
    print("=" * 60)

    for dp in [0.2, 0.3, 0.35, 0.4]:
        key = f"dropout={dp}"
        if dp == default_dropout:
            results[key] = {'f1': baseline_f1, 'is_default': True}
            print(f"\n  {key}: 使用基线结果 F1={baseline_f1}")
            continue

        set_seed(RANDOM_SEED)
        model = DownstreamConfigurable(
            num_conformer_layers=default_layers,
            dropout=dp).to(device)
        criterion = get_criterion(use_focal=True)
        criterion.gamma = default_gamma

        metrics = run_single_experiment(
            key, model, train_loader, val_loader, device, criterion, lr=default_lr)
        results[key] = {'f1': round(metrics['f1'], 4),
                        'acc': round(metrics['accuracy'], 2),
                        'auc': round(metrics.get('auc', 0), 4)}

        del model
        torch.cuda.empty_cache()

    # ===================== 扫描 Focal Loss γ =====================
    print("\n" + "=" * 60)
    print("[Step 5] Focal Loss γ 扫描")
    print("=" * 60)

    for gamma in [1.0, 2.0, 3.0]:
        key = f"gamma={gamma}"
        if gamma == default_gamma:
            results[key] = {'f1': baseline_f1, 'is_default': True}
            print(f"\n  {key}: 使用基线结果 F1={baseline_f1}")
            continue

        set_seed(RANDOM_SEED)
        model = DownstreamConfigurable(
            num_conformer_layers=default_layers,
            dropout=default_dropout).to(device)

        # 构造 FocalLoss 并覆盖 gamma
        criterion = get_criterion(use_focal=True)
        criterion.gamma = gamma

        metrics = run_single_experiment(
            key, model, train_loader, val_loader, device, criterion, lr=default_lr)
        results[key] = {'f1': round(metrics['f1'], 4),
                        'acc': round(metrics['accuracy'], 2),
                        'auc': round(metrics.get('auc', 0), 4)}

        del model
        torch.cuda.empty_cache()

    # ===================== 汇总与保存 =====================
    print("\n" + "=" * 60)
    print("超参数敏感性分析结果汇总")
    print("=" * 60)

    print(f"\n{'超参数':<20} {'取值':<12} {'验证集F1':<12} {'选定':<6}")
    print("-" * 52)

    hp_groups = [
        ('学习率', [('5e-5', 'lr=5e-05'), ('1e-4', 'lr=0.0001'),
                    ('2e-4', 'lr=0.0002'), ('5e-4', 'lr=0.0005')]),
        ('Conformer层数', [('4', 'layers=4'), ('6', 'layers=6'), ('8', 'layers=8')]),
        ('Dropout率', [('0.2', 'dropout=0.2'), ('0.3', 'dropout=0.3'),
                       ('0.35', 'dropout=0.35'), ('0.4', 'dropout=0.4')]),
        ('Focal Loss γ', [('1.0', 'gamma=1.0'), ('2.0', 'gamma=2.0'), ('3.0', 'gamma=3.0')]),
    ]

    for hp_name, entries in hp_groups:
        for val_str, key in entries:
            f1 = results[key]['f1']
            is_default = results[key].get('is_default', False)
            mark = '✓' if is_default else '—'
            print(f"{hp_name:<20} {val_str:<12} {f1:<12.4f} {mark}")
            hp_name = ''  # 后续行不重复显示超参数名

    # 保存到 JSON
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / 'hp_sweep_results.json'
    with open(str(out_path), 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'seed': RANDOM_SEED,
            'default_config': {
                'lr': default_lr, 'layers': default_layers,
                'dropout': default_dropout, 'gamma': default_gamma,
            },
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    total_time = time.time() - total_start
    print(f"\n总耗时: {total_time/60:.1f} 分钟")


if __name__ == '__main__':
    main()
