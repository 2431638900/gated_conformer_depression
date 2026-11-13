# -*- coding: utf-8 -*-
"""
模块贡献度验证实验 V10.12（预提取特征加速版）
通过逐步添加模块的方式，量化每个模块对最终性能的增量贡献。

核心优化：HuBERT冻结特征预提取一次，后续实验复用，跳过HuBERT forward pass。

实验设计（递增式）：
  M1: HuBERT(WLA) + AvgPool + Linear                                    (最简基线)
  M2: HuBERT(WLA) + GatedConformer + AvgPool                            (+ Conformer)
  M3: HuBERT(WLA) + DepExtractor + GatedConformer + AvgPool             (+ 抑郁特征提取器)
  M4: HuBERT(WLA) + DepExtractor + GatedConformer + StatAttnPool        (+ 统计注意力池化)
  M5: M4 + Focal Loss                                                    (+ Focal Loss = 完整模型)

严谨性保证：
- 所有阶段使用相同的数据划分（augmented_stratified）
- 每个阶段训练前重置随机种子
- 模型架构参数与02_train_model.py V10.12一致
- 训练超参数一致
- M1-M4使用CrossEntropy，M5使用FocalLoss（体现FocalLoss贡献）
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
    build_full_model, build_model_no_dep_extractor,
    get_criterion, get_optimizer_and_scheduler,
    MODEL_ARCH, TRAIN_CONFIG, RANDOM_SEED, PLOT_DIR, LOG_DIR,
)
from utils.metrics import compute_metrics, EarlyStopping, AverageMeter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===================== 特征预提取（复用 06_ablation_study_v2 的缓存） =====================
FEATURE_CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'feature_cache'


def _get_feature_cache_path(audio_path: str, model_type: str) -> Path:
    key = hashlib.md5(f"{model_type}:{audio_path}".encode()).hexdigest()
    return FEATURE_CACHE_DIR / model_type / f"{key}.npy"


@torch.no_grad()
def precompute_weighted_features(model, df, device, cache_tag='hubert_wla_v10'):
    """使用训练好的模型的加权层聚合来预提取特征。复用缓存。"""
    cache_dir = FEATURE_CACHE_DIR / cache_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_paths, all_labels = [], []
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


# ===================== 下游模型变体（接收预提取768-d特征） =====================
class M1_DS_Linear(nn.Module):
    """最简基线: 预提取特征(768) -> AvgPool -> Linear"""
    def __init__(self, dropout=0.35):
        super().__init__()
        d_model = MODEL_ARCH['d_model']
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, MODEL_ARCH['num_classes']),
        )
    def forward(self, features):
        return self.classifier(features.mean(dim=1))


class M2_DS_ConformerAvgPool(nn.Module):
    """+ Conformer: 特征(768) -> Conformer -> AvgPool -> classifier"""
    def __init__(self, base_model):
        super().__init__()
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
        out = self.conformer(features)
        return self.classifier(out.mean(dim=1))


class M3_DS_DepExtractorAvgPool(nn.Module):
    """+ DepExtractor: 特征 -> DepExtractor -> Conformer -> AvgPool -> classifier"""
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
        out = self.conformer(features)
        return self.classifier(out.mean(dim=1))


class M4_DS_FullWithStatAttnPool(nn.Module):
    """+ StatAttnPool: 特征 -> DepExtractor -> Conformer -> StatAttnPool -> classifier"""
    def __init__(self, base_model):
        super().__init__()
        self.depression_extractor = base_model.depression_extractor
        self.conformer = base_model.conformer
        self.pooling = base_model.pooling
        self.classifier = base_model.classifier
    def forward(self, features):
        features = self.depression_extractor(features)
        out = self.conformer(features)
        pooled = self.pooling(out)
        return self.classifier(pooled)


# ===================== 训练与评估（AMP加速） =====================
_amp_scaler = torch.cuda.amp.GradScaler()


def train_one_epoch(model, loader, criterion, optimizer, device,
                    gradient_accumulation_steps=1, max_grad_norm=1.0):
    model.train()
    meter = AverageMeter()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
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

        meter.update(loss.item() * gradient_accumulation_steps, features.size(0))

    if (step + 1) % gradient_accumulation_steps != 0:
        _amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        _amp_scaler.step(optimizer)
        _amp_scaler.update()
        optimizer.zero_grad()

    return meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    all_labels, all_probs = [], []
    for batch in loader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        with torch.cuda.amp.autocast():
            out = model(features)
            loss = criterion(out, labels)
        meter.update(loss.item(), features.size(0))
        probs = torchF.softmax(out.float(), dim=1)[:, 1].cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
    preds = (np.array(all_probs) > 0.5).astype(int)
    metrics = compute_metrics(all_labels, preds, all_probs)
    metrics['loss'] = meter.avg
    return metrics


def run_stage(name, model, train_loader, val_loader, device, criterion):
    """训练单个阶段"""
    print(f"\n{'=' * 50}")
    print(f"Stage: {name}")
    print(f"{'=' * 50}")

    set_seed(RANDOM_SEED)

    num_epochs = TRAIN_CONFIG['num_epochs']
    grad_accum = TRAIN_CONFIG['gradient_accumulation_steps']
    max_grad_norm = TRAIN_CONFIG['max_grad_norm']
    patience = TRAIN_CONFIG['early_stopping_patience']

    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {train_p/1e6:.2f}M")

    optimizer, scheduler = get_optimizer_and_scheduler(model, num_epochs)
    es = EarlyStopping(patience=patience, mode='max')
    best_f1, best_metrics, best_state = 0, None, None

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                 gradient_accumulation_steps=grad_accum,
                                 max_grad_norm=max_grad_norm)
        v_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        if v_metrics['f1'] > best_f1:
            best_f1 = v_metrics['f1']
            best_metrics = v_metrics.copy()
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Ep {epoch:3d} | {epoch_time:.1f}s | TLoss {t_loss:.4f} | "
              f"VF1 {v_metrics['f1']:.4f} | VAcc {v_metrics['accuracy']:.2f}%",
              flush=True)

        es(v_metrics['f1'])
        if es.early_stop:
            print(f"  Early stop @ ep {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_metrics['trainable_params'] = train_p
    print(f"  {name}: F1={best_metrics['f1']:.4f} | Acc={best_metrics['accuracy']:.2f}%")
    return best_metrics


# ===================== 论文级绘图 =====================
def plot_module_contribution(summary, save_dir):
    """绘制模块贡献度图（论文级，双Y轴）"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11

    stage_names = [s['stage'].split(': ')[1] if ': ' in s['stage'] else s['stage']
                   for s in summary]
    f1s = [s['f1'] for s in summary]
    deltas = [s['delta_f1'] for s in summary]
    x = np.arange(len(stage_names))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # F1柱状图
    bar_colors = ['#4FC3F7', '#42A5F5', '#5C6BC0', '#26A69A', '#4CAF50']
    bars = ax1.bar(x, f1s, color=bar_colors[:len(f1s)],
                   edgecolor='white', linewidth=1.0, label='F1-Score', width=0.55)
    ax1.set_ylabel('F1-Score', fontsize=12, color='#333')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_names, rotation=25, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    ax1.spines['top'].set_visible(False)

    for bar, val in zip(bars, f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # dF1折线图（双Y轴）
    ax2 = ax1.twinx()
    ax2.plot(x, deltas, 'o-', color='#D32F2F', linewidth=2.5, markersize=9,
             label='$\\Delta$F1', zorder=5)
    ax2.set_ylabel('$\\Delta$F1 (Incremental)', fontsize=12, color='#D32F2F')
    ax2.spines['top'].set_visible(False)

    for xi, d in zip(x, deltas):
        ax2.annotate(f'{d:+.4f}', (xi, d), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, color='#D32F2F',
                     fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               framealpha=0.9, fontsize=10)

    fig.suptitle('Module Contribution Analysis (V10.12)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / 'module_contribution.png'), dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


# ===================== 主函数 =====================
def main():
    total_start = time.time()
    print("=" * 60)
    print("Module Contribution Experiment V10.12 - 预提取特征加速版")
    print("Config aligned with 02_train_model.py V10.12")
    print("=" * 60)

    set_seed()
    device = get_device()

    train_df, val_df = load_train_val_data()

    # 预提取HuBERT加权特征（复用缓存）
    print("\n" + "=" * 60)
    print("预提取HuBERT加权层聚合特征（复用缓存）")
    print("=" * 60)
    from scripts.experiment_config import load_trained_model
    trained_model, _ = load_trained_model(device)
    print("  --- 训练集 ---")
    train_feat_paths, train_labels = precompute_weighted_features(
        trained_model, train_df, device, 'hubert_wla_v10')
    print("  --- 验证集 ---")
    val_feat_paths, val_labels = precompute_weighted_features(
        trained_model, val_df, device, 'hubert_wla_v10')
    del trained_model
    torch.cuda.empty_cache()

    feat_train_loader, feat_val_loader = create_feature_dataloaders(
        train_feat_paths, train_labels, val_feat_paths, val_labels)
    print(f"  特征DataLoader就绪: train={len(feat_train_loader)}步, val={len(feat_val_loader)}步")

    criterion_ce = get_criterion(use_focal=False)
    criterion_focal = get_criterion(use_focal=True)

    results = {}

    # M1: HuBERT(WLA) + Linear (CE loss)
    print("\n\n" + "#" * 50)
    print("# M1: HuBERT(WLA) + AvgPool + Linear")
    print("#" * 50)
    m1 = M1_DS_Linear(dropout=TRAIN_CONFIG['dropout']).to(device)
    results['M1: HuBERT+Linear'] = run_stage(
        'M1: HuBERT+Linear', m1, feat_train_loader, feat_val_loader, device,
        criterion=criterion_ce)
    del m1
    torch.cuda.empty_cache()

    # M2: + GatedConformer (CE loss)
    print("\n\n" + "#" * 50)
    print("# M2: + GatedConformer")
    print("#" * 50)
    base = build_model_no_dep_extractor(device)
    m2 = M2_DS_ConformerAvgPool(base).to(device)
    del base
    torch.cuda.empty_cache()
    results['M2: +Conformer'] = run_stage(
        'M2: +Conformer', m2, feat_train_loader, feat_val_loader, device,
        criterion=criterion_ce)
    del m2
    torch.cuda.empty_cache()

    # M3: + DepExtractor (CE loss)
    print("\n\n" + "#" * 50)
    print("# M3: + DepExtractor")
    print("#" * 50)
    base = build_full_model(device)
    m3 = M3_DS_DepExtractorAvgPool(base).to(device)
    del base
    torch.cuda.empty_cache()
    results['M3: +DepExtractor'] = run_stage(
        'M3: +DepExtractor', m3, feat_train_loader, feat_val_loader, device,
        criterion=criterion_ce)
    del m3
    torch.cuda.empty_cache()

    # M4: + StatisticsAttentionPooling (CE loss)
    print("\n\n" + "#" * 50)
    print("# M4: + StatAttentionPooling")
    print("#" * 50)
    base = build_full_model(device)
    m4 = M4_DS_FullWithStatAttnPool(base).to(device)
    del base
    torch.cuda.empty_cache()
    results['M4: +StatAttnPool'] = run_stage(
        'M4: +StatAttnPool', m4, feat_train_loader, feat_val_loader, device,
        criterion=criterion_ce)
    del m4
    torch.cuda.empty_cache()

    # M5: + Focal Loss = Full Model
    print("\n\n" + "#" * 50)
    print("# M5: + Focal Loss = Full Model")
    print("#" * 50)
    base = build_full_model(device)
    m5 = M4_DS_FullWithStatAttnPool(base).to(device)
    del base
    torch.cuda.empty_cache()
    results['M5: Full Model'] = run_stage(
        'M5: Full Model', m5, feat_train_loader, feat_val_loader, device,
        criterion=criterion_focal)
    del m5
    torch.cuda.empty_cache()

    # ===================== Summary =====================
    total_time = time.time() - total_start
    print(f"\n\n总耗时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
    print("\n" + "=" * 95)
    print("Module Contribution Summary (V10.12)")
    print("=" * 95)
    header = (f"{'Stage':<25s} {'F1':>8s} {'dF1':>8s} {'Acc(%)':>8s} "
              f"{'Recall':>8s} {'AUC':>8s} {'Params':>10s}")
    print(header)
    print("-" * 95)

    summary = []
    prev_f1 = 0
    for name, m in results.items():
        delta = m['f1'] - prev_f1
        params_str = f"{m.get('trainable_params', 0)/1e6:.2f}M"
        print(f"{name:<25s} {m['f1']:>8.4f} {delta:>+8.4f} {m['accuracy']:>8.2f} "
              f"{m['recall']:>8.4f} {m.get('auc', 0):>8.4f} {params_str:>10s}")
        summary.append({
            'stage': name, 'f1': m['f1'], 'delta_f1': delta,
            'accuracy': m['accuracy'], 'precision': m['precision'],
            'recall': m['recall'], 'auc': m.get('auc', 0),
            'params': m.get('trainable_params', 0),
        })
        prev_f1 = m['f1']
    print("=" * 95)

    # Plot
    plot_module_contribution(summary, PLOT_DIR)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        'timestamp': timestamp,
        'experiment': 'module_contribution_v10.12',
        'config': {
            'model_arch': MODEL_ARCH,
            'train_config': {k: v for k, v in TRAIN_CONFIG.items()},
            'seed': RANDOM_SEED,
        },
        'results': {},
    }
    for name, m in results.items():
        save_data['results'][name] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in m.items()
        }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOG_DIR / f'module_contribution_{timestamp}.json'
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    csv_path = LOG_DIR / f'module_contribution_{timestamp}.csv'
    pd.DataFrame(summary).to_csv(str(csv_path), index=False, encoding='utf-8')

    print("\nModule contribution experiment (V10.12) complete!")
    print(f"  Plot: {PLOT_DIR / 'module_contribution.png'}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == '__main__':
    main()
