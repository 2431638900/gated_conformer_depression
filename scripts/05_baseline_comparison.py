# -*- coding: utf-8 -*-
"""
对照实验脚本 V10.12 - 基线模型对比（AMP + 预提取特征加速版）
训练多种基线模型并与本文提出的模型进行性能对比。

对比模型：
1. CNN (纯卷积基线，原始音频输入)
2. BiLSTM (循环网络基线，原始音频输入)
3. BiLSTM + Attention (原始音频输入)
4. Transformer Encoder (原始音频输入)
5. HuBERT + Linear (预训练模型最简方案)
6. HuBERT + BiLSTM+Attention
7. HuBERT + Transformer
8. Ours V10.12 (从checkpoint加载)

优化：
- 模型5-7使用预提取HuBERT加权层聚合特征
- 全部模型使用AMP混合精度加速

严谨性保证：
- 所有模型使用相同的数据划分（augmented_stratified）
- 每个模型训练前重置随机种子
- HuBERT基线模型使用与本文方法相同的加权层聚合特征
- 所有模型使用相同的损失函数（FocalLoss）
- 训练超参数尽可能一致

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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from torch.utils.data import Dataset, DataLoader

from scripts.experiment_config import (
    set_seed, get_device, load_train_val_data, create_dataloaders,
    load_trained_model, get_criterion, get_optimizer_and_scheduler,
    TRAIN_CONFIG, RANDOM_SEED, PLOT_DIR, LOG_DIR,
)
from models.baseline_models import get_baseline_model
from utils.metrics import compute_metrics, EarlyStopping, AverageMeter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# ===================== 特征预提取（复用06_ablation_study_v2缓存） =====================
FEATURE_CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'feature_cache'


def _get_feature_cache_path(audio_path: str, model_type: str) -> Path:
    key = hashlib.md5(f"{model_type}:{audio_path}".encode()).hexdigest()
    return FEATURE_CACHE_DIR / model_type / f"{key}.npy"


@torch.no_grad()
def precompute_weighted_features(model, df, device, cache_tag='hubert_wla_v10'):
    """使用训练好的模型的加权层聚合来预提取特征。复用06_ablation_study_v2缓存。"""
    cache_dir = FEATURE_CACHE_DIR / cache_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_paths, all_labels = [], []
    for idx in range(len(df)):
        row = df.iloc[idx]
        all_paths.append(_get_feature_cache_path(str(row['Audio_Path']), cache_tag))
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
            bs = audio.size(0)
            with torch.cuda.amp.autocast():
                encoder_output = encoder(audio, output_hidden_states=True)
                hidden_states = encoder_output.hidden_states
                stacked = torch.stack(hidden_states, dim=0)
                features = (stacked * layer_weights.view(-1, 1, 1, 1)).sum(dim=0)
            features_np = features.cpu().float().numpy()
            for i in range(bs):
                if not all_paths[sample_idx + i].exists():
                    np.save(str(all_paths[sample_idx + i]), features_np[i])
            sample_idx += bs
            if sample_idx % 500 < bs:
                print(f"    [{sample_idx}/{len(all_paths)}] {time.time()-start:.0f}s", flush=True)
        print(f"  特征提取完成: {time.time()-start:.1f}秒")
    else:
        print(f"  加权特征缓存已就绪 ({len(all_paths)} 个文件)")
    return all_paths, all_labels


class FeatureDataset(Dataset):
    def __init__(self, feature_paths, labels):
        self.feature_paths = feature_paths
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        features = np.load(str(self.feature_paths[idx]))
        return {'features': torch.from_numpy(features),
                'label': torch.LongTensor([self.labels[idx]])}


def feature_collate_fn(batch):
    return {'features': torch.stack([b['features'] for b in batch]),
            'label': torch.cat([b['label'] for b in batch])}


def create_feature_dataloaders(train_paths, train_labels, val_paths, val_labels):
    bs = TRAIN_CONFIG['batch_size']
    train_loader = DataLoader(FeatureDataset(train_paths, train_labels),
                              batch_size=bs, shuffle=True, num_workers=0,
                              collate_fn=feature_collate_fn, pin_memory=True)
    val_loader = DataLoader(FeatureDataset(val_paths, val_labels),
                            batch_size=bs, shuffle=False, num_workers=0,
                            collate_fn=feature_collate_fn, pin_memory=True)
    return train_loader, val_loader


# ===================== HuBERT下游模型（接收预提取768-d特征） =====================
class HuBERTLinearDS(nn.Module):
    """HuBERT下游: 加权特征(768) -> AvgPool -> Linear"""
    def __init__(self, num_classes=2, dropout=0.35):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    def forward(self, features):
        return self.classifier(features.mean(dim=1))


class HuBERTBiLSTMDS(nn.Module):
    """HuBERT下游: 加权特征(768) -> BiLSTM + Attention -> classifier"""
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_w, dim=1)
        return self.classifier(context)


class HuBERTTransformerDS(nn.Module):
    """HuBERT下游: 加权特征(768) -> Transformer Encoder -> AvgPool -> classifier"""
    def __init__(self, d_model=256, num_heads=8, num_layers=4, num_classes=2, dropout=0.35):
        super().__init__()
        self.input_proj = nn.Linear(768, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
    def forward(self, features):
        x = self.input_proj(features)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


# ===================== Dummy基线 =====================
def run_dummy_baselines(train_df, val_df, n_repeats=100):
    """计算随机/多数类等Dummy基线的性能下界"""
    set_seed(RANDOM_SEED)
    val_labels = val_df['PHQ8_Binary'].values.tolist()
    n_val = len(val_labels)
    train_pos_ratio = train_df['PHQ8_Binary'].mean()
    train_neg_ratio = 1.0 - train_pos_ratio

    results = {}
    # 1. MajorityClass
    preds_maj = [0] * n_val
    probs_maj = [train_neg_ratio] * n_val
    m = compute_metrics(val_labels, preds_maj, probs_maj)
    results['MajorityClass'] = {'f1': m['f1'], 'accuracy': m['accuracy'], 'auc': m.get('auc', 0)}

    # 2. Random
    rand_f1s = []
    for i in range(n_repeats):
        np.random.seed(RANDOM_SEED + i)
        probs_rand = np.random.rand(n_val)
        preds_rand = (probs_rand < train_pos_ratio).astype(int)
        probs_out = np.where(preds_rand == 1,
                             train_pos_ratio + np.random.rand(n_val) * 0.1,
                             train_neg_ratio + np.random.rand(n_val) * 0.1)
        m = compute_metrics(val_labels, preds_rand.tolist(), np.clip(probs_out, 0, 1).tolist())
        rand_f1s.append(m['f1'])
    results['Random'] = {'f1': float(np.mean(rand_f1s)), 'accuracy': 0, 'auc': 0.5}

    # 3. PriorProbability
    probs_prior = [float(train_pos_ratio)] * n_val
    preds_prior = [1 if p > 0.5 else 0 for p in probs_prior]
    m = compute_metrics(val_labels, preds_prior, probs_prior)
    results['PriorProbability'] = {'f1': m['f1'], 'accuracy': m['accuracy'], 'auc': m.get('auc', 0)}

    print("\nDummy基线结果:")
    for name, r in results.items():
        print(f"  {name}: F1={r['f1']:.4f} Acc={r['accuracy']:.2f}% AUC={r['auc']:.4f}")
    return results


# ===================== 训练与评估函数（AMP加速） =====================
_amp_scaler = torch.cuda.amp.GradScaler()


def train_one_epoch_audio(model, dataloader, criterion, optimizer, device,
                          gradient_accumulation_steps=1, max_grad_norm=1.0):
    """原始音频输入的训练（模型1-4）"""
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        with torch.cuda.amp.autocast():
            loss = criterion(model(audio), labels)
            loss = loss / gradient_accumulation_steps
        _amp_scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            _amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            _amp_scaler.step(optimizer)
            _amp_scaler.update()
            optimizer.zero_grad()
        loss_meter.update(loss.item() * gradient_accumulation_steps, audio.size(0))
    if (step + 1) % gradient_accumulation_steps != 0:
        _amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        _amp_scaler.step(optimizer)
        _amp_scaler.update()
        optimizer.zero_grad()
    return loss_meter.avg


def train_one_epoch_feat(model, dataloader, criterion, optimizer, device,
                         gradient_accumulation_steps=1, max_grad_norm=1.0):
    """预提取特征输入的训练（模型5-7）"""
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        with torch.cuda.amp.autocast():
            loss = criterion(model(features), labels)
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
def evaluate_audio(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_labels, all_probs = [], []
    for batch in dataloader:
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(audio)
            loss = criterion(outputs, labels)
        probs = torchF.softmax(outputs.float(), dim=1)[:, 1].cpu().numpy()
        loss_meter.update(loss.item(), audio.size(0))
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
    preds = (np.array(all_probs) > 0.5).astype(int)
    metrics = compute_metrics(all_labels, preds, all_probs)
    metrics['loss'] = loss_meter.avg
    return metrics


@torch.no_grad()
def evaluate_feat(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_labels, all_probs = [], []
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


def train_and_evaluate(name, model, train_loader, val_loader, device, criterion,
                       use_features=False, seed=RANDOM_SEED):
    """训练单个模型并返回最佳验证指标"""
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print(f"{'=' * 60}")

    set_seed(seed)

    num_epochs = TRAIN_CONFIG['num_epochs']
    grad_accum = TRAIN_CONFIG['gradient_accumulation_steps']
    max_grad_norm = TRAIN_CONFIG['max_grad_norm']
    patience = TRAIN_CONFIG['early_stopping_patience']

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_p/1e6:.2f}M | Trainable: {train_p/1e6:.2f}M")

    train_fn = train_one_epoch_feat if use_features else train_one_epoch_audio
    eval_fn = evaluate_feat if use_features else evaluate_audio

    optimizer, scheduler = get_optimizer_and_scheduler(model, num_epochs)
    early_stopping = EarlyStopping(patience=patience, mode='max')
    best_f1 = 0
    best_metrics = None

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_fn(model, train_loader, criterion, optimizer, device,
                              gradient_accumulation_steps=grad_accum,
                              max_grad_norm=max_grad_norm)
        val_metrics = eval_fn(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()

        print(f"  Epoch {epoch:3d} | {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%",
              flush=True)

        early_stopping(val_metrics['f1'])
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"  {name} best: F1={best_metrics['f1']:.4f} | "
          f"Acc={best_metrics['accuracy']:.2f}% | AUC={best_metrics.get('auc',0):.4f}")
    best_metrics['trainable_params'] = train_p
    return best_metrics


# ===================== 论文级绘图 =====================
def plot_baseline_results(results, save_dir):
    """绘制基线对比图（论文级）"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11

    names = list(results.keys())
    f1s = [results[n]['f1'] for n in names]
    accs = [results[n]['accuracy'] for n in names]
    aucs = [results[n].get('auc', 0) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(names))

    for ax, vals, metric_name in zip(axes, [f1s, accs, aucs],
                                      ['F1-Score', 'Accuracy (%)', 'AUC-ROC']):
        colors = ['#FF7043'] * (len(vals) - 1) + ['#4CAF50']
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=1.0, width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, vals):
            fmt = f'{val:.2f}' if metric_name == 'Accuracy (%)' else f'{val:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    fmt, ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.suptitle('Baseline Comparison (V10.12)', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / 'baseline_comparison.png'), dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


# ===================== 主函数 =====================
def main():
    """
    基线对比主函数（合并版）
    合并自: 07_baseline_comparison + 10_dummy_baseline
    Part 1: 8个模型单次对比
    Part 2: Dummy基线（性能下界）
    """
    total_start = time.time()
    print("=" * 70)
    print("基线对比 V10.12 (含Dummy基线)")
    print("合并自: 07 + 10_dummy_baseline")
    print("=" * 70)

    set_seed()
    device = get_device()
    train_df, val_df = load_train_val_data()

    # ==================== Part 1: 单次基线对比（8个模型） ====================
    print("\n\n" + "=" * 70)
    print("Part 1: 单次基线对比（8个模型）")
    print("=" * 70)

    train_loader, val_loader = create_dataloaders(train_df, val_df)

    print("\n预提取HuBERT加权层聚合特征（复用缓存）...")
    trained_model, _ = load_trained_model(device)
    train_feat_paths, train_labels = precompute_weighted_features(
        trained_model, train_df, device, 'hubert_wla_v10')
    val_feat_paths, val_labels = precompute_weighted_features(
        trained_model, val_df, device, 'hubert_wla_v10')
    del trained_model; torch.cuda.empty_cache()
    feat_train_loader, feat_val_loader = create_feature_dataloaders(
        train_feat_paths, train_labels, val_feat_paths, val_labels)

    criterion = get_criterion(use_focal=True)
    results = {}
    dropout = TRAIN_CONFIG['dropout']

    # 1-4: 原始音频输入基线
    model = get_baseline_model('cnn', num_classes=2, dropout=dropout).to(device)
    results['CNN'] = train_and_evaluate('CNN', model, train_loader, val_loader,
                                        device, criterion, use_features=False)
    del model; torch.cuda.empty_cache()

    model = get_baseline_model('lstm', hidden_dim=256, num_layers=2,
                               num_classes=2, dropout=dropout).to(device)
    results['BiLSTM'] = train_and_evaluate('BiLSTM', model, train_loader, val_loader,
                                            device, criterion, use_features=False)
    del model; torch.cuda.empty_cache()

    model = get_baseline_model('lstm_attention', hidden_dim=256, num_layers=2,
                               num_classes=2, dropout=dropout).to(device)
    results['BiLSTM+Attn'] = train_and_evaluate(
        'BiLSTM+Attention', model, train_loader, val_loader,
        device, criterion, use_features=False)
    del model; torch.cuda.empty_cache()

    model = get_baseline_model('transformer', d_model=256, num_heads=8,
                               num_layers=4, num_classes=2, dropout=dropout).to(device)
    results['Transformer'] = train_and_evaluate(
        'Transformer', model, train_loader, val_loader,
        device, criterion, use_features=False)
    del model; torch.cuda.empty_cache()
    del train_loader, val_loader

    # 5-7: HuBERT预提取特征基线
    model = HuBERTLinearDS(num_classes=2, dropout=dropout).to(device)
    results['HuBERT+Linear'] = train_and_evaluate(
        'HuBERT+Linear', model, feat_train_loader, feat_val_loader,
        device, criterion, use_features=True)
    del model; torch.cuda.empty_cache()

    model = HuBERTBiLSTMDS(hidden_dim=256, num_layers=2,
                            num_classes=2, dropout=dropout).to(device)
    results['HuBERT+BiLSTM'] = train_and_evaluate(
        'HuBERT+BiLSTM', model, feat_train_loader, feat_val_loader,
        device, criterion, use_features=True)
    del model; torch.cuda.empty_cache()

    model = HuBERTTransformerDS(d_model=256, num_heads=8, num_layers=4,
                                 num_classes=2, dropout=dropout).to(device)
    results['HuBERT+Trans'] = train_and_evaluate(
        'HuBERT+Transformer', model, feat_train_loader, feat_val_loader,
        device, criterion, use_features=True)
    del model; torch.cuda.empty_cache()

    # 8: Ours (checkpoint)
    try:
        ours_model, _ = load_trained_model(device)
        ours_val_loader = create_dataloaders(train_df, val_df)[1]
        ours_metrics = evaluate_audio(ours_model, ours_val_loader, criterion, device)
        ours_metrics['trainable_params'] = sum(
            p.numel() for p in ours_model.parameters() if p.requires_grad)
        results['Ours'] = ours_metrics
        print(f"  Ours: F1={ours_metrics['f1']:.4f} | "
              f"Acc={ours_metrics['accuracy']:.2f}% | AUC={ours_metrics.get('auc',0):.4f}")
        del ours_model, ours_val_loader; torch.cuda.empty_cache()
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")

    # Part 1 汇总
    print("\n" + "=" * 95)
    print("Part 1: Baseline Comparison Results (V10.12)")
    print("=" * 95)
    header = (f"{'Model':<22s} {'Params':>10s} {'F1':>8s} {'Acc(%)':>8s} "
              f"{'Prec':>8s} {'Recall':>8s} {'AUC':>8s}")
    print(header)
    print("-" * 95)
    summary_rows = []
    for name, m in results.items():
        params_str = f"{m.get('trainable_params', 0)/1e6:.2f}M"
        print(f"{name:<22s} {params_str:>10s} {m['f1']:>8.4f} {m['accuracy']:>8.2f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m.get('auc', 0):>8.4f}")
        summary_rows.append({
            'model': name, 'params': m.get('trainable_params', 0),
            'f1': m['f1'], 'accuracy': m['accuracy'],
            'precision': m['precision'], 'recall': m['recall'], 'auc': m.get('auc', 0),
        })
    print("=" * 95)
    plot_baseline_results(results, PLOT_DIR)

    # ==================== Part 2: Dummy基线 ====================
    print("\n\n" + "=" * 70)
    print("Part 2: Dummy基线（性能下界）")
    print("=" * 70)
    dummy_results = run_dummy_baselines(train_df, val_df, n_repeats=100)

    # ==================== 保存所有结果 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        'timestamp': timestamp,
        'experiment': 'baseline_comparison_full_v10.12',
        'config': {
            'train_config': {k: v for k, v in TRAIN_CONFIG.items()},
            'seed': RANDOM_SEED,
        },
        'part1_single_run': {
            name: {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in m.items()}
            for name, m in results.items()
        },
        'part2_dummy': dummy_results,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOG_DIR / f'baseline_comparison_{timestamp}.json'
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    csv_path = LOG_DIR / f'baseline_comparison_{timestamp}.csv'
    pd.DataFrame(summary_rows).to_csv(str(csv_path), index=False, encoding='utf-8')

    total_time = time.time() - total_start
    print(f"\n总耗时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
    print("实验完成!")
    print(f"  Plot: {PLOT_DIR / 'baseline_comparison.png'}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == '__main__':
    main()
