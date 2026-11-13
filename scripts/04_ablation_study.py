# -*- coding: utf-8 -*-
"""
消融实验脚本 V10.12（预提取特征加速版）
系统性移除/替换模型中的各个模块，评估其对最终性能的影响。

核心优化：
  HuBERT在所有实验中都是冻结的，因此预先提取一次特征，后续实验
  直接使用特征训练下游模块，跳过HuBERT forward pass。

消融实验设计（V10.12）：
1. Full Model（完整模型）: 直接加载训练好的V10.12 checkpoint作为基准
2. w/o Depression Extractor: 去掉抑郁特征提取器，HuBERT直接接Conformer
3. w/o GatedConformer: 去掉Conformer，用2层MLP替代
4. w/o StatAttentionPooling: 用简单AvgPooling替代统计注意力池化
5. w/o DropPath: 去掉Stochastic Depth (DropPath=0)
6. w/o Focal Loss: 使用标准CrossEntropy替代Focal Loss

严谨性保证：
- 所有变体使用相同的数据划分（augmented_stratified）
- 每个变体训练前重置随机种子（确保可复现）
- 模型架构参数与02_train_model.py V10.12完全一致
- 训练超参数与原始训练一致
"""

import sys
import os
import json
import copy
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
    set_seed, get_device, load_train_val_data,
    build_full_model, load_trained_model,
    get_criterion, get_optimizer_and_scheduler,
    MODEL_ARCH, TRAIN_CONFIG, RANDOM_SEED, PLOT_DIR, LOG_DIR,
)
from models.depression_model import StatisticsAttentionPooling
from models.conformer import Conformer
from utils.metrics import compute_metrics, EarlyStopping, AverageMeter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_REPEATS = 5
SEEDS = [42, 43, 44, 45, 46]


# ===================== 特征预提取 =====================
FEATURE_CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'feature_cache'


def _get_feature_cache_path(audio_path: str, model_type: str) -> Path:
    """audio_path + model_type → 唯一的.npy缓存路径"""
    key = hashlib.md5(f"{model_type}:{audio_path}".encode()).hexdigest()
    return FEATURE_CACHE_DIR / model_type / f"{key}.npy"


def _aggregate_hidden_states(hidden_states, layer_weights, aggregation='wla'):
    stacked = torch.stack(hidden_states, dim=0)
    if aggregation == 'wla':
        return (stacked * layer_weights.view(-1, 1, 1, 1)).sum(dim=0)
    if aggregation == 'mean':
        return stacked.mean(dim=0)
    if aggregation == 'last':
        return hidden_states[-1]
    raise ValueError(f'Unsupported aggregation: {aggregation}')


@torch.no_grad()
def precompute_weighted_features(model, df, device, cache_tag='hubert_wla', aggregation='wla'):
    """
    使用训练好的模型的加权层聚合来预提取特征。
    与模型forward的前半段完全一致：HuBERT → 13层加权聚合 → (B, T, 768)
    """
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
        print(f"  预提取{aggregation}特征 ({len(uncached)}/{len(all_paths)} 个新样本)...")

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
                features = _aggregate_hidden_states(hidden_states, layer_weights, aggregation)

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
    """加载预提取的特征（代替原始音频）"""
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


# ===================== 下游模型变体（接收预提取768-d特征） =====================
class DownstreamFull(nn.Module):
    """完整下游: DepExtractor + GatedConformer + StatAttnPooling + Classifier"""
    def __init__(self, base_model):
        super().__init__()
        self.depression_extractor = base_model.depression_extractor
        self.conformer = base_model.conformer
        self.pooling = base_model.pooling
        self.classifier = base_model.classifier

    def forward(self, features):
        features = self.depression_extractor(features)
        features = self.conformer(features)
        pooled = self.pooling(features)
        return self.classifier(pooled)


class DownstreamNoDepExtractor(nn.Module):
    """去掉DepExtractor: Conformer(input_dim=768) + StatAttnPooling + Classifier"""
    def __init__(self):
        super().__init__()
        from models.conformer import Conformer
        d_model = MODEL_ARCH['d_model']
        self.conformer = Conformer(
            input_dim=768,
            d_model=d_model,
            num_layers=MODEL_ARCH['num_conformer_layers'],
            num_heads=MODEL_ARCH['num_attention_heads'],
            conv_kernel_size=MODEL_ARCH['conv_kernel_size'],
            feed_forward_dim=MODEL_ARCH['feed_forward_dim'],
            dropout=TRAIN_CONFIG['dropout'],
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
        features = self.conformer(features)
        pooled = self.pooling(features)
        return self.classifier(pooled)


class DownstreamNoConformer(nn.Module):
    """去掉Conformer: DepExtractor + 2层MLP + StatAttnPooling + Classifier"""
    def __init__(self, base_model):
        super().__init__()
        self.depression_extractor = base_model.depression_extractor
        d_model = MODEL_ARCH['d_model']
        dropout = TRAIN_CONFIG['dropout']
        self.mlp_replacement = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pooling = StatisticsAttentionPooling(d_model)
        self.classifier = base_model.classifier

    def forward(self, features):
        features = self.depression_extractor(features)
        features = self.mlp_replacement(features)
        pooled = self.pooling(features)
        return self.classifier(pooled)


class DownstreamNoStatAttnPooling(nn.Module):
    """用AvgPooling替代StatisticsAttentionPooling
    注意：分类器输入维度从 2*d_model 变为 d_model"""
    def __init__(self, base_model):
        super().__init__()
        self.depression_extractor = base_model.depression_extractor
        self.conformer = base_model.conformer
        d_model = MODEL_ARCH['d_model']
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
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
        pooled = torch.mean(features, dim=1)  # (B, d_model)
        return self.classifier(pooled)


class DownstreamStandardConformer(nn.Module):
    """标准Conformer下游 (use_gating=False, 单尺度卷积, 无门控)"""
    def __init__(self, base_model, drop_path_rate=0.0):
        super().__init__()
        self.depression_extractor = copy.deepcopy(base_model.depression_extractor)
        d_model = MODEL_ARCH['d_model']
        self.conformer = Conformer(
            input_dim=d_model,
            d_model=d_model,
            num_layers=MODEL_ARCH['num_conformer_layers'],
            num_heads=MODEL_ARCH['num_attention_heads'],
            conv_kernel_size=MODEL_ARCH['conv_kernel_size'],
            feed_forward_dim=MODEL_ARCH['feed_forward_dim'],
            dropout=TRAIN_CONFIG['dropout'],
            use_gating=False,
            drop_path_rate=drop_path_rate,
        )
        self.pooling = StatisticsAttentionPooling(d_model)
        pool_output_dim = d_model * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_output_dim),
            nn.Linear(pool_output_dim, d_model),
            nn.GELU(),
            nn.Dropout(TRAIN_CONFIG['dropout']),
            nn.Linear(d_model, MODEL_ARCH['num_classes']),
        )

    def forward(self, features):
        features = self.depression_extractor(features)
        features = self.conformer(features)
        pooled = self.pooling(features)
        return self.classifier(pooled)


class DownstreamStandardConformerMatched(nn.Module):
    """参数量匹配的标准Conformer (d_model=320使参数量接近GatedConformer)"""
    def __init__(self, base_model):
        super().__init__()
        self.depression_extractor = copy.deepcopy(base_model.depression_extractor)
        d_model_big = 320
        self.conformer = Conformer(
            input_dim=MODEL_ARCH['d_model'],
            d_model=d_model_big,
            num_layers=MODEL_ARCH['num_conformer_layers'],
            num_heads=MODEL_ARCH['num_attention_heads'],
            conv_kernel_size=MODEL_ARCH['conv_kernel_size'],
            feed_forward_dim=int(MODEL_ARCH['feed_forward_dim'] * 320 / 256),
            dropout=TRAIN_CONFIG['dropout'],
            use_gating=False,
            drop_path_rate=0.0,
        )
        self.pooling = StatisticsAttentionPooling(d_model_big)
        pool_output_dim = d_model_big * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(pool_output_dim),
            nn.Linear(pool_output_dim, d_model_big),
            nn.GELU(),
            nn.Dropout(TRAIN_CONFIG['dropout']),
            nn.Linear(d_model_big, MODEL_ARCH['num_classes']),
        )

    def forward(self, features):
        features = self.depression_extractor(features)
        features = self.conformer(features)
        pooled = self.pooling(features)
        return self.classifier(pooled)


# ===================== 训练与评估函数 =====================
_amp_scaler = torch.cuda.amp.GradScaler()


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    gradient_accumulation_steps=1, max_grad_norm=1.0):
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


# ===================== 单次消融实验 =====================
def run_ablation_variant(variant_name, model, train_loader, val_loader, device,
                         criterion, num_epochs=None, seed=RANDOM_SEED):
    """运行单个消融变体的训练和评估"""
    print(f"\n{'=' * 60}")
    print(f"消融实验: {variant_name}")
    print(f"{'=' * 60}")

    set_seed(seed)

    num_epochs = num_epochs or TRAIN_CONFIG['num_epochs']
    grad_accum = TRAIN_CONFIG['gradient_accumulation_steps']
    max_grad_norm = TRAIN_CONFIG['max_grad_norm']
    patience = TRAIN_CONFIG['early_stopping_patience']

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    optimizer, scheduler = get_optimizer_and_scheduler(model, num_epochs)
    early_stopping = EarlyStopping(patience=patience, mode='max')

    best_f1 = 0
    best_metrics = None
    best_state = None

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_accumulation_steps=grad_accum, max_grad_norm=max_grad_norm)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        val_f1 = val_metrics['f1']
        val_loss = val_metrics['loss']

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics.copy()
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:3d} | {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.2f}%", flush=True)

        early_stopping(val_f1)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_metrics['trainable_params'] = trainable_params
    print(f"\n  {variant_name} best: "
          f"F1={best_metrics['f1']:.4f} | Acc={best_metrics['accuracy']:.2f}% | "
          f"AUC={best_metrics.get('auc', 0):.4f}")

    return best_metrics


# ===================== 论文级绘图 =====================
def plot_ablation_results(results, save_dir):
    """绘制消融实验结果对比图（论文级）"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11

    names = list(results.keys())
    metrics_data = {
        'F1-Score': [results[n]['f1'] for n in names],
        'Accuracy (%)': [results[n]['accuracy'] for n in names],
        'AUC-ROC': [results[n].get('auc', 0) for n in names],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    short_names = []
    for n in names:
        if n == 'Full Model':
            short_names.append('Full Model\n(Ours)')
        else:
            short_names.append(n.replace('w/o ', 'w/o\n'))

    x = np.arange(len(names))

    for ax, (metric_name, vals) in zip(axes, metrics_data.items()):
        colors = ['#4CAF50' if i == 0 else '#FF7043' for i in range(len(vals))]
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=1.0, width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8, ha='center')
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 标注数值
        for bar, val in zip(bars, vals):
            fmt = f'{val:.2f}' if metric_name == 'Accuracy (%)' else f'{val:.4f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    fmt, ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        # 标注相对Full Model的下降
        if len(vals) > 1:
            full_val = vals[0]
            for i in range(1, len(vals)):
                delta = vals[i] - full_val
                if delta < 0:
                    ax.text(x[i], min(vals) - 0.02 * max(vals),
                            f'{delta:.4f}' if metric_name != 'Accuracy (%)' else f'{delta:.2f}',
                            ha='center', va='top', fontsize=7, color='red', fontstyle='italic')

    fig.suptitle('Ablation Study Results (V10.12)', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / 'ablation_study_results.png'), dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def _to_serializable_metrics(metrics):
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float)):
            serializable[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            serializable[key] = int(value)
        else:
            serializable[key] = value
    return serializable


def _summarize_representation_results(all_results, baseline_f1):
    print("\n" + "=" * 110)
    print("Representation Ablation Results (V10.12) — mean ± std over 5 seeds")
    print("=" * 110)
    header = (f"{'Variant':<30s} {'F1':>16s} {'Acc(%)':>16s} "
              f"{'AUC':>16s} {'n':>4s}")
    print(header)
    print("-" * 110)

    summary_rows = []
    for variant_name, data in all_results.items():
        repeats = data['repeats']
        n = data['n']
        f1_vals = [m['f1'] for m in repeats]
        acc_vals = [m['accuracy'] for m in repeats]
        auc_vals = [m.get('auc', 0.0) for m in repeats]

        f1_mean = float(np.mean(f1_vals))
        f1_std = float(np.std(f1_vals))
        acc_mean = float(np.mean(acc_vals))
        acc_std = float(np.std(acc_vals))
        auc_mean = float(np.mean(auc_vals))
        auc_std = float(np.std(auc_vals))
        delta_f1 = f1_mean - baseline_f1

        line = (f"{variant_name:<30s} {f1_mean:.4f}±{f1_std:.4f}  "
                f"{acc_mean:.2f}±{acc_std:.2f}  "
                f"{auc_mean:.4f}±{auc_std:.4f}  {n:>4d}"
                f"  (ΔF1={delta_f1:+.4f})")
        print(line)

        summary_rows.append({
            'variant': variant_name,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'delta_f1_mean': delta_f1,
            'n_repeats': n,
        })

    print("=" * 110)
    return summary_rows


def run_representation_ablation(device, train_df, val_df):
    total_start = time.time()
    print("\n" + "=" * 70)
    print("表示层聚合对照实验 V10.12")
    print("=" * 70)

    set_seed()

    print("\n步骤1: 加载训练好的主模型")
    trained_model, _ = load_trained_model(device)

    feature_sets = {}
    for agg_name, agg_label, cache_tag in [
        ('wla', 'WLA', 'hubert_wla_v10'),
        ('mean', 'Mean', 'hubert_mean_v10'),
        ('last', 'Last', 'hubert_last_v10'),
    ]:
        print(f"\n步骤2: 预提取{agg_label}特征")
        print("  --- 训练集 ---")
        train_paths, train_labels = precompute_weighted_features(
            trained_model, train_df, device, cache_tag, aggregation=agg_name)
        print("  --- 验证集 ---")
        val_paths, val_labels = precompute_weighted_features(
            trained_model, val_df, device, cache_tag, aggregation=agg_name)
        feature_sets[agg_name] = (train_paths, train_labels, val_paths, val_labels)

    criterion_focal = get_criterion(use_focal=True)

    wla_train_paths, wla_train_labels, wla_val_paths, wla_val_labels = feature_sets['wla']
    feat_train_loader, feat_val_loader = create_feature_dataloaders(
        wla_train_paths, wla_train_labels, wla_val_paths, wla_val_labels)

    print("\n步骤3: 评估WLA完整模型")
    ds_full_eval = DownstreamFull(trained_model).to(device)
    full_metrics = evaluate(ds_full_eval, feat_val_loader, criterion_focal, device)
    full_metrics['trainable_params'] = sum(
        p.numel() for p in trained_model.parameters() if p.requires_grad)
    print(f"  WLA (Full Model): F1={full_metrics['f1']:.4f} | "
          f"Acc={full_metrics['accuracy']:.2f}% | AUC={full_metrics.get('auc', 0):.4f}")
    del trained_model, ds_full_eval
    torch.cuda.empty_cache()

    baseline_name = 'WLA (Full Model)'
    all_results = {baseline_name: {'repeats': [full_metrics], 'n': 1}}

    variant_configs = [
        ('Mean Layer Aggregation', 'mean'),
        ('Last Layer Only', 'last'),
    ]

    for variant_name, agg_name in variant_configs:
        print(f"\n\n{'#' * 60}")
        print(f"# {variant_name} ({N_REPEATS}次重复)")
        print(f"{'#' * 60}")

        repeat_metrics = []
        train_paths, train_labels, val_paths, val_labels = feature_sets[agg_name]

        for rep_idx, seed in enumerate(SEEDS):
            print(f"\n  --- 第 {rep_idx + 1}/{N_REPEATS} 次 (seed={seed}) ---")
            set_seed(seed)

            feat_train_loader, feat_val_loader = create_feature_dataloaders(
                train_paths, train_labels, val_paths, val_labels)

            base = build_full_model(device)
            model = DownstreamFull(base).to(device)
            del base

            torch.cuda.empty_cache()
            metrics = run_ablation_variant(
                f'{variant_name} (seed={seed})', model,
                feat_train_loader, feat_val_loader, device,
                criterion=criterion_focal, seed=seed)
            repeat_metrics.append(metrics)
            del model
            torch.cuda.empty_cache()

        all_results[variant_name] = {'repeats': repeat_metrics, 'n': N_REPEATS}

    total_time = time.time() - total_start
    print(f"\n\n总耗时: {total_time / 60:.1f}分钟 ({total_time / 3600:.2f}小时)")

    summary_rows = _summarize_representation_results(all_results, full_metrics['f1'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_data = {
        'timestamp': timestamp,
        'experiment': 'representation_ablation_v10.12_repeated',
        'n_repeats': N_REPEATS,
        'seeds': SEEDS,
        'config': {
            'model_arch': MODEL_ARCH,
            'train_config': {k: v for k, v in TRAIN_CONFIG.items()},
            'seed': RANDOM_SEED,
        },
        'results': {},
    }

    for variant_name, data in all_results.items():
        save_data['results'][variant_name] = {
            'repeats': [_to_serializable_metrics(m) for m in data['repeats']],
            'n': data['n'],
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOG_DIR / f'representation_ablation_{timestamp}.json'
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    csv_path = LOG_DIR / f'representation_ablation_{timestamp}.csv'
    pd.DataFrame(summary_rows).to_csv(str(csv_path), index=False, encoding='utf-8')

    print("\n结果已保存:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


# ===================== 主函数 =====================
def main():
    """
    消融实验主函数（合并版）
    合并自: 06_ablation_study + 11_ablation_repeated + 12_standard_conformer_ablation
    - 8个消融变体 + 2个标准Conformer对照
    - 每个变体5次种子重复，报告 mean±std
    """
    total_start = time.time()
    print("=" * 70)
    print("消融实验 V10.12 (含标准Conformer对照 + 表示层聚合对照 + 5次种子重复)")
    print("合并自: ablation_study + representation_ablation")
    print("=" * 70)

    set_seed()
    device = get_device()
    train_df, val_df = load_train_val_data()

    # Step 1: 预提取HuBERT WLA特征
    print("\n步骤1: 预提取HuBERT加权层聚合特征")
    trained_model, _ = load_trained_model(device)
    print("  --- 训练集 ---")
    train_feat_paths, train_labels = precompute_weighted_features(
        trained_model, train_df, device, 'hubert_wla_v10')
    print("  --- 验证集 ---")
    val_feat_paths, val_labels = precompute_weighted_features(
        trained_model, val_df, device, 'hubert_wla_v10')

    # Full Model 基准（只评估一次，不受种子影响）
    print("\n\n" + "#" * 60)
    print("# Full Model (from checkpoint, 单次评估)")
    print("#" * 60)
    feat_train_loader, feat_val_loader = create_feature_dataloaders(
        train_feat_paths, train_labels, val_feat_paths, val_labels)
    criterion_focal = get_criterion(use_focal=True)
    criterion_ce = get_criterion(use_focal=False)

    ds_full_eval = DownstreamFull(trained_model).to(device)
    full_metrics = evaluate(ds_full_eval, feat_val_loader, criterion_focal, device)
    full_metrics['trainable_params'] = sum(
        p.numel() for p in trained_model.parameters() if p.requires_grad)
    print(f"  Full Model: F1={full_metrics['f1']:.4f} | "
          f"Acc={full_metrics['accuracy']:.2f}% | AUC={full_metrics.get('auc', 0):.4f}")
    del trained_model, ds_full_eval
    torch.cuda.empty_cache()

    all_results = {'Full Model': {'repeats': [full_metrics], 'n': 1}}

    # 消融变体定义: (名称, 构建key, 损失类型)
    variant_configs = [
        ('w/o DepExtractor',          'no_dep',             'focal'),
        ('w/o Conformer',             'no_conf',            'focal'),
        ('w/o StatAttnPool',          'no_pool',            'focal'),
        ('w/o DropPath',              'no_dp',              'focal'),
        ('w/o FocalLoss',             'no_fl',              'ce'),
        ('Standard Conformer',        'std_conf',           'focal'),
        ('Std Conformer (matched)',   'std_conf_matched',   'focal'),
    ]

    for variant_name, variant_key, crit_type in variant_configs:
        print(f"\n\n{'#' * 60}")
        print(f"# {variant_name} ({N_REPEATS}次重复)")
        print(f"{'#' * 60}")

        crit = criterion_focal if crit_type == 'focal' else criterion_ce
        repeat_metrics = []

        for rep_idx, seed in enumerate(SEEDS):
            print(f"\n  --- 第 {rep_idx+1}/{N_REPEATS} 次 (seed={seed}) ---")
            set_seed(seed)

            feat_train_loader, feat_val_loader = create_feature_dataloaders(
                train_feat_paths, train_labels, val_feat_paths, val_labels)

            if variant_key == 'no_dep':
                model = DownstreamNoDepExtractor().to(device)
            elif variant_key == 'no_conf':
                base = build_full_model(device)
                model = DownstreamNoConformer(base).to(device)
                del base
            elif variant_key == 'no_pool':
                base = build_full_model(device)
                model = DownstreamNoStatAttnPooling(base).to(device)
                del base
            elif variant_key == 'no_dp':
                base = build_full_model(device, drop_path_override=0.0)
                model = DownstreamFull(base).to(device)
                del base
            elif variant_key == 'no_fl':
                base = build_full_model(device)
                model = DownstreamFull(base).to(device)
                del base
            elif variant_key == 'std_conf':
                base = build_full_model(device)
                model = DownstreamStandardConformer(base).to(device)
                del base
            elif variant_key == 'std_conf_matched':
                base = build_full_model(device)
                model = DownstreamStandardConformerMatched(base).to(device)
                del base

            torch.cuda.empty_cache()
            metrics = run_ablation_variant(
                f'{variant_name} (seed={seed})', model,
                feat_train_loader, feat_val_loader, device, criterion=crit,
                seed=seed)
            repeat_metrics.append(metrics)
            del model
            torch.cuda.empty_cache()

        all_results[variant_name] = {'repeats': repeat_metrics, 'n': N_REPEATS}

    # ===================== 汇总结果 =====================
    total_time = time.time() - total_start
    print(f"\n\n总耗时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")

    print("\n" + "=" * 110)
    print("Ablation Study Results (V10.12) — mean ± std over 5 seeds")
    print("=" * 110)
    header = (f"{'Variant':<30s} {'F1':>16s} {'Acc(%)':>16s} "
              f"{'AUC':>16s} {'n':>4s}")
    print(header)
    print("-" * 110)

    summary_rows = []
    full_f1_mean = full_metrics['f1']

    for variant_name, data in all_results.items():
        repeats = data['repeats']
        n = data['n']
        f1s = [m['f1'] for m in repeats]
        accs = [m['accuracy'] for m in repeats]
        aucs = [m.get('auc', 0) for m in repeats]

        f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))
        acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
        auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))

        if n == 1:
            line = (f"{variant_name:<30s} {f1_mean:>8.4f}         "
                    f"{acc_mean:>8.2f}         {auc_mean:>8.4f}         {n:>4d}")
        else:
            delta_f1 = f1_mean - full_f1_mean
            line = (f"{variant_name:<30s} {f1_mean:.4f}±{f1_std:.4f}  "
                    f"{acc_mean:.2f}±{acc_std:.2f}  "
                    f"{auc_mean:.4f}±{auc_std:.4f}  {n:>4d}"
                    f"  (ΔF1={delta_f1:+.4f})")
        print(line)

        summary_rows.append({
            'variant': variant_name,
            'f1_mean': f1_mean, 'f1_std': f1_std,
            'accuracy_mean': acc_mean, 'accuracy_std': acc_std,
            'auc_mean': auc_mean, 'auc_std': auc_std,
            'delta_f1_mean': f1_mean - full_f1_mean,
            'n_repeats': n,
        })
    print("=" * 110)

    # 绘图（仅绘制核心消融变体，不含Standard Conformer）
    core_results = {name: data['repeats'][0] for name, data in all_results.items()
                    if name not in ['Standard Conformer', 'Std Conformer (matched)', 'Mean Layer Aggregation', 'Last Layer Only']}
    plot_ablation_results(core_results, PLOT_DIR)

    # 保存JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        'timestamp': timestamp,
        'experiment': 'ablation_study_v10.12_repeated',
        'n_repeats': N_REPEATS,
        'seeds': SEEDS,
        'config': {
            'model_arch': MODEL_ARCH,
            'train_config': {k: v for k, v in TRAIN_CONFIG.items()},
            'seed': RANDOM_SEED,
        },
        'results': {},
    }
    for variant_name, data in all_results.items():
        save_data['results'][variant_name] = {
            'repeats': [
                {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                 for k, v in m.items()}
                for m in data['repeats']
            ],
            'n': data['n'],
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOG_DIR / f'ablation_study_{timestamp}.json'
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    csv_path = LOG_DIR / f'ablation_study_{timestamp}.csv'
    pd.DataFrame(summary_rows).to_csv(str(csv_path), index=False, encoding='utf-8')

    print("\n结果已保存:")
    print(f"  Plot: {PLOT_DIR / 'ablation_study_results.png'}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    run_representation_ablation(device, train_df, val_df)


if __name__ == '__main__':
    main()
