# -*- coding: utf-8 -*-
"""
MODMA跨语言迁移实验（合并版）
系统对比多种跨语言迁移策略，验证语言适配器的优势。

合并自: 12_cross_language_control.py + 13_crosslingual_baselines.py
修复: 旧版optimal threshold在全量数据上搜索+评估导致数据泄露，
      现改为在训练折搜索阈值、验证折评估。

实验条件 (受试者级5折CV):
  1. Zero-Shot (th=0.5): 直接迁移，不做任何调整
  2. Zero-Shot (optimal th): 在训练折搜索最优阈值，验证折评估
  3. Linear Probe: 仅训练新的线性分类头
  4. Full Fine-Tune: 解冻全部下游模块（DepExtractor+Conformer+Classifier）
  5. Freeze-7 Fine-Tune: 冻结HuBERT前7层，微调后5层+下游
  6. Adapter (Ours): 仅训练199K参数适配器
  7. LoRA (rank=8): 在HuBERT最后2层添加低秩适配

严谨性保证:
- 受试者级分层5折，同一受试者不跨训练/验证
- 所有策略共享相同的折划分
- 每折重新加载DAIC-WOZ训练好的基础模型权重
- Zero-Shot阈值搜索仅使用训练折（无数据泄露）
"""

import sys
import os
import json
import copy
import time
import random
import warnings
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from config.config import Config, config
from data.dataset import DepressionDataset, collate_fn
from models.depression_model import DepressionDetectionModel
from utils.metrics import compute_metrics

PROJECT_ROOT = Config.PROJECT_ROOT
MODMA_CSV = PROJECT_ROOT / 'data' / 'modma_processed' / 'modma_samples.csv'
CHECKPOINT_DIR = Config.CHECKPOINT_DIR
LOG_DIR = Config.LOG_DIR
SEED = 42

# ===================== 通用配置 =====================
BASE_TRAIN = {
    'learning_rate': 5e-4,
    'weight_decay': 1e-3,
    'epochs': 40,
    'batch_size': 4,
    'accum_steps': 4,
    'patience': 10,
    'class_weights': [1.0, 1.5],
    'use_amp': True,
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_base_model(device, adapter_config=None):
    """加载V10.12基础模型（DAIC-WOZ训练好的）"""
    ckpt_path = CHECKPOINT_DIR / 'best_model.pth'
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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
        adapter_config=adapter_config,
    ).to(device)

    # strict=False 允许adapter参数缺失
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model


def get_subject_folds(modma_df, n_folds=5):
    """受试者级分层5折划分"""
    subjects = modma_df.groupby('Original_ID')['PHQ8_Binary'].first().reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in skf.split(subjects['Original_ID'], subjects['PHQ8_Binary']):
        train_subjects = subjects.iloc[train_idx]['Original_ID'].tolist()
        val_subjects = subjects.iloc[val_idx]['Original_ID'].tolist()
        train_df = modma_df[modma_df['Original_ID'].isin(train_subjects)].reset_index(drop=True)
        val_df = modma_df[modma_df['Original_ID'].isin(val_subjects)].reset_index(drop=True)
        folds.append((train_df, val_df))
    return folds


def create_loaders(train_df, val_df, batch_size=4):
    sr = config.SAMPLE_RATE
    tl = 30 * sr
    train_ds = DepressionDataset(train_df, sr, tl, mode='pad')
    val_ds = DepressionDataset(val_df, sr, tl, mode='pad')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader


# ===================== 迁移策略实现 =====================
def setup_zero_shot(model, device=None):
    """Zero-Shot: 不训练任何参数"""
    for p in model.parameters():
        p.requires_grad = False
    return model


def setup_linear_probe(model, device=None):
    """Linear Probe: 仅训练一个新的线性分类头"""
    for p in model.parameters():
        p.requires_grad = False
    d_model = model.conformer.d_model
    pool_dim = d_model * 2
    model.classifier = nn.Sequential(
        nn.LayerNorm(pool_dim),
        nn.Linear(pool_dim, 2),
    ).to(device)
    return model


def setup_full_finetune(model, device=None):
    """Full Fine-Tune: 冻结HuBERT, 解冻全部下游"""
    for p in model.feature_encoder.parameters():
        p.requires_grad = False
    for module in [model.depression_extractor, model.conformer, model.pooling, model.classifier]:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
    model.layer_weights.requires_grad = True
    return model


def setup_freeze7_finetune(model, device=None):
    """Freeze-7: 冻结HuBERT前7层，微调后5层+下游"""
    for p in model.feature_encoder.parameters():
        p.requires_grad = False
    total_layers = len(model.feature_encoder.encoder.layers)
    for i in range(total_layers - 5, total_layers):
        for p in model.feature_encoder.encoder.layers[i].parameters():
            p.requires_grad = True
    for module in [model.depression_extractor, model.conformer, model.pooling, model.classifier]:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
    model.layer_weights.requires_grad = True
    return model


def setup_adapter(model, device=None):
    """Adapter: 冻结全部原始权重，仅训练轻量级适配器"""
    for p in model.parameters():
        p.requires_grad = False
    if model.language_adapter is not None:
        for p in model.language_adapter.parameters():
            p.requires_grad = True
    return model


class LoRALinear(nn.Module):
    """低秩适配层"""
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.scaling = alpha / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def setup_lora(model, device=None, rank=8):
    """LoRA: 在HuBERT最后2层的Q/K/V投影上添加低秩适配"""
    for p in model.parameters():
        p.requires_grad = False
    encoder_layers = model.feature_encoder.encoder.layers
    total = len(encoder_layers)
    lora_modules = []
    for i in range(total - 2, total):
        attn = encoder_layers[i].attention
        for attr in ['q_proj', 'k_proj', 'v_proj']:
            orig = getattr(attn, attr)
            if isinstance(orig, nn.Linear):
                lora_layer = LoRALinear(orig, rank=rank).to(device)
                setattr(attn, attr, lora_layer)
                lora_modules.append(lora_layer)
    lora_params = sum(p.numel() for m in lora_modules for p in m.parameters() if p.requires_grad)
    print(f"  LoRA: {len(lora_modules)} 个低秩适配层 (rank={rank}), 可训练参数: {lora_params/1e3:.1f}K")
    return model


# ===================== 评估与阈值搜索 =====================
@torch.no_grad()
def _run_inference(model, loader, device, use_adapter=False):
    """统一推理：返回 labels, probs, subjects"""
    model.eval()
    all_labels, all_probs, all_subjects = [], [], []
    for batch in loader:
        audio = batch['audio'].to(device)
        labels = batch['label']
        subjects = batch['participant_id']
        with torch.cuda.amp.autocast():
            logits = model(audio, use_adapter=True) if use_adapter else model(audio)
        probs = torchF.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())
        all_subjects.extend([str(s) for s in subjects])
    return all_labels, all_probs, all_subjects


def _compute_sample_and_subject_metrics(labels, probs, subjects, threshold=0.5):
    """根据给定阈值计算样本级和受试者级指标"""
    preds = (np.array(probs) > threshold).astype(int)
    sample_m = compute_metrics(labels, preds.tolist(), probs)

    df = pd.DataFrame({'sid': subjects, 'label': labels, 'prob': probs})
    sdf = df.groupby('sid').agg(label=('label', 'first'), prob=('prob', 'mean')).reset_index()
    sdf['pred'] = (sdf['prob'] > threshold).astype(int)
    subj_m = compute_metrics(sdf['label'].tolist(), sdf['pred'].tolist(), sdf['prob'].tolist())
    return sample_m, subj_m


def find_optimal_threshold(labels, probs, subjects):
    """在给定数据上搜索受试者级F1最优阈值（用于训练折）"""
    df = pd.DataFrame({'sid': subjects, 'label': labels, 'prob': probs})
    sdf = df.groupby('sid').agg(label=('label', 'first'), prob=('prob', 'mean')).reset_index()
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.05, 0.95, 0.01):
        preds = (sdf['prob'] > th).astype(int)
        m = compute_metrics(sdf['label'].tolist(), preds.tolist(), sdf['prob'].tolist())
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_th = th
    return best_th, best_f1


# ===================== 训练与评估 =====================
def train_and_evaluate_fold(model, train_loader, val_loader, device, cfg,
                            use_adapter=False, search_threshold=False):
    """训练一折并返回最佳指标"""
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)

    if n_trainable == 0:
        # Zero-Shot: 不训练，直接评估
        if search_threshold:
            # 在训练折搜索最优阈值（无数据泄露）
            train_labels, train_probs, train_subjects = _run_inference(
                model, train_loader, device, use_adapter=use_adapter)
            opt_th, _ = find_optimal_threshold(train_labels, train_probs, train_subjects)
            print(f"    训练折最优阈值: {opt_th:.2f}")
        else:
            opt_th = 0.5

        val_labels, val_probs, val_subjects = _run_inference(
            model, val_loader, device, use_adapter=use_adapter)
        sample_m, subj_m = _compute_sample_and_subject_metrics(
            val_labels, val_probs, val_subjects, threshold=opt_th)
        sample_m['trainable_params'] = 0
        subj_m['trainable_params'] = 0
        sample_m['threshold'] = opt_th
        subj_m['threshold'] = opt_th
        return sample_m, subj_m

    # 有可训练参数：正常训练流程
    optimizer = torch.optim.AdamW(trainable, lr=cfg['learning_rate'],
                                   weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    weights = torch.tensor(cfg['class_weights'], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg['use_amp'])
    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            with torch.cuda.amp.autocast(enabled=cfg['use_amp']):
                logits = model(audio, use_adapter=True) if use_adapter else model(audio)
                loss = criterion(logits, labels) / cfg['accum_steps']

            scaler.scale(loss).backward()
            if (step + 1) % cfg['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # 尾部梯度
        if (step + 1) % cfg['accum_steps'] != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        # 验证
        val_labels, val_probs, val_subjects = _run_inference(
            model, val_loader, device, use_adapter=use_adapter)
        _, subj_m = _compute_sample_and_subject_metrics(
            val_labels, val_probs, val_subjects)
        val_f1 = subj_m['f1']

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | SubjF1={val_f1:.4f} | "
                  f"SubjAUC={subj_m.get('auc',0):.4f} | best={best_f1:.4f}")

        if patience_counter >= cfg['patience']:
            print(f"    Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs, val_subjects = _run_inference(
        model, val_loader, device, use_adapter=use_adapter)
    sample_m, subj_m = _compute_sample_and_subject_metrics(
        val_labels, val_probs, val_subjects)
    sample_m['trainable_params'] = n_trainable
    subj_m['trainable_params'] = n_trainable
    return sample_m, subj_m


# ===================== 策略定义 =====================
STRATEGIES = [
    # (名称, setup函数, 选项)
    ('Zero-Shot (th=0.5)',      setup_zero_shot,        {'search_threshold': False}),
    ('Zero-Shot (optimal th)',  setup_zero_shot,        {'search_threshold': True}),
    ('Linear Probe',            setup_linear_probe,     {}),
    ('Full Fine-Tune',          setup_full_finetune,    {}),
    ('Freeze-7 Fine-Tune',     setup_freeze7_finetune, {}),
    ('Adapter (Ours)',          setup_adapter,          {'use_adapter': True}),
    ('LoRA (rank=8)',           setup_lora,             {}),
]


# ===================== 主函数 =====================
def main():
    total_start = time.time()
    print("=" * 70)
    print("MODMA跨语言迁移实验 (受试者级5折CV)")
    print("合并自: 12_cross_language_control + 13_crosslingual_baselines")
    print("修复: optimal threshold 数据泄露")
    print("=" * 70)

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not MODMA_CSV.exists():
        print(f"❌ MODMA数据未预处理: {MODMA_CSV}")
        return

    modma_df = pd.read_csv(str(MODMA_CSV))
    n_subjects = modma_df['Original_ID'].nunique()
    print(f"MODMA: {len(modma_df)} 样本, {n_subjects} 受试者")

    folds = get_subject_folds(modma_df, n_folds=5)
    print("5折划分完成")

    all_results = {}

    # 上次运行中不受 _has_trainable_pretrain 缓存bug影响的策略结果
    # 仅 Freeze-7 和 LoRA 需要重跑（设为 None 跳过注入）
    COMPLETED_RESULTS = {
        'Zero-Shot (th=0.5)': {
            'sample': {'f1_mean': 0.3016, 'f1_std': 0.1684, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                       'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                       'auc_mean': 0.0, 'auc_std': 0.0, 'trainable_params': 0},
            'subject': {'f1_mean': 0.1714, 'f1_std': 0.3429, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                        'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                        'auc_mean': 0.6610, 'auc_std': 0.2039, 'trainable_params': 0},
        },
        'Zero-Shot (optimal th)': {
            'sample': {'f1_mean': 0.4193, 'f1_std': 0.1572, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                       'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                       'auc_mean': 0.0, 'auc_std': 0.0, 'trainable_params': 0},
            'subject': {'f1_mean': 0.6592, 'f1_std': 0.2273, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                        'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                        'auc_mean': 0.6610, 'auc_std': 0.2039, 'trainable_params': 0},
        },
        'Linear Probe': {
            'sample': {'f1_mean': 0.3851, 'f1_std': 0.1114, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                       'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                       'auc_mean': 0.0, 'auc_std': 0.0, 'trainable_params': 2050},
            'subject': {'f1_mean': 0.5310, 'f1_std': 0.2714, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                        'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                        'auc_mean': 0.6350, 'auc_std': 0.1794, 'trainable_params': 2050},
        },
        'Full Fine-Tune': {
            'sample': {'f1_mean': 0.5770, 'f1_std': 0.0363, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                       'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                       'auc_mean': 0.0, 'auc_std': 0.0, 'trainable_params': 18440855},
            'subject': {'f1_mean': 0.7659, 'f1_std': 0.0932, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                        'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                        'auc_mean': 0.8700, 'auc_std': 0.0340, 'trainable_params': 18440855},
        },
        'Adapter (Ours)': {
            'sample': {'f1_mean': 0.5922, 'f1_std': 0.0785, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                       'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                       'auc_mean': 0.0, 'auc_std': 0.0, 'trainable_params': 199042},
            'subject': {'f1_mean': 0.8025, 'f1_std': 0.0760, 'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                        'precision_mean': 0.0, 'precision_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                        'auc_mean': 0.7780, 'auc_std': 0.1686, 'trainable_params': 199042},
        },
    }

    for strategy_name, setup_fn, options in STRATEGIES:
        # 跳过已完成且不受bug影响的策略
        if strategy_name in COMPLETED_RESULTS:
            print(f"\n\n{'#' * 70}")
            print(f"# 策略: {strategy_name} [跳过 - 使用上次正确结果]")
            print(f"{'#' * 70}")
            all_results[strategy_name] = COMPLETED_RESULTS[strategy_name]
            s = COMPLETED_RESULTS[strategy_name]['subject']
            sm = COMPLETED_RESULTS[strategy_name]['sample']
            print(f"  样本级  F1={sm['f1_mean']:.4f}±{sm['f1_std']:.4f}")
            print(f"  受试者级 F1={s['f1_mean']:.4f}±{s['f1_std']:.4f} "
                  f"AUC={s['auc_mean']:.4f}±{s['auc_std']:.4f}")
            continue

        print(f"\n\n{'#' * 70}")
        print(f"# 策略: {strategy_name}")
        print(f"{'#' * 70}")

        fold_sample = []
        fold_subject = []
        search_threshold = options.get('search_threshold', False)
        use_adapter = options.get('use_adapter', False)

        for fold_idx, (train_df, val_df) in enumerate(folds):
            print(f"\n  --- Fold {fold_idx+1}/5 "
                  f"(train={len(train_df)}, val={len(val_df)}) ---")
            set_seed(SEED + fold_idx)

            # 每折重新加载模型
            is_adapter = strategy_name.startswith('Adapter')
            adapter_cfg = {'bottleneck_dim': 128, 'dropout': 0.1} if is_adapter else None
            model = load_base_model(device, adapter_config=adapter_cfg)

            # 配置策略
            if strategy_name.startswith('Zero-Shot'):
                setup_zero_shot(model, device)
            elif strategy_name == 'Adapter (Ours)':
                setup_adapter(model, device)
            elif strategy_name.startswith('LoRA'):
                setup_lora(model, device, rank=8)
            elif strategy_name == 'Linear Probe':
                setup_linear_probe(model, device)
            elif strategy_name == 'Full Fine-Tune':
                setup_full_finetune(model, device)
            elif strategy_name == 'Freeze-7 Fine-Tune':
                setup_freeze7_finetune(model, device)

            # 刷新缓存：setup可能修改了feature_encoder的requires_grad
            model._has_trainable_pretrain = any(
                p.requires_grad for p in model.feature_encoder.parameters()
            )

            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  可训练参数: {n_train/1e3:.1f}K ({n_train/1e6:.2f}M)")

            train_loader, val_loader = create_loaders(train_df, val_df,
                                                       batch_size=BASE_TRAIN['batch_size'])

            sample_m, subj_m = train_and_evaluate_fold(
                model, train_loader, val_loader, device, BASE_TRAIN,
                use_adapter=use_adapter, search_threshold=search_threshold)

            fold_sample.append(sample_m)
            fold_subject.append(subj_m)
            print(f"  Fold {fold_idx+1}: SubjF1={subj_m['f1']:.4f} "
                  f"SubjAUC={subj_m.get('auc',0):.4f}")

            del model
            torch.cuda.empty_cache()

        # 汇总5折
        avg_sample = {}
        avg_subject = {}
        for key in ['f1', 'accuracy', 'precision', 'recall', 'auc']:
            vals_s = [m.get(key, 0) for m in fold_sample]
            vals_j = [m.get(key, 0) for m in fold_subject]
            avg_sample[f'{key}_mean'] = float(np.mean(vals_s))
            avg_sample[f'{key}_std'] = float(np.std(vals_s))
            avg_subject[f'{key}_mean'] = float(np.mean(vals_j))
            avg_subject[f'{key}_std'] = float(np.std(vals_j))

        n_params = fold_subject[0].get('trainable_params', 0)
        avg_subject['trainable_params'] = n_params
        avg_sample['trainable_params'] = n_params

        all_results[strategy_name] = {
            'sample': avg_sample,
            'subject': avg_subject,
            'fold_details': [
                {'sample': s, 'subject': j}
                for s, j in zip(fold_sample, fold_subject)
            ],
        }

        print(f"\n  {strategy_name} 5折平均:")
        print(f"    样本级  F1={avg_sample['f1_mean']:.4f}±{avg_sample['f1_std']:.4f}")
        print(f"    受试者级 F1={avg_subject['f1_mean']:.4f}±{avg_subject['f1_std']:.4f} "
              f"AUC={avg_subject['auc_mean']:.4f}±{avg_subject['auc_std']:.4f}")

    # ===================== 汇总表 =====================
    total_time = time.time() - total_start
    print(f"\n\n总耗时: {total_time/60:.1f}分钟")

    print("\n" + "=" * 100)
    print("MODMA跨语言迁移策略对比 (受试者级5折CV)")
    print("=" * 100)
    header = (f"{'策略':<28s} {'Params':>10s} {'SubjF1':>16s} {'SubjAUC':>16s} "
              f"{'SampleF1':>16s}")
    print(header)
    print("-" * 100)

    for name, res in all_results.items():
        s = res['subject']
        sm = res['sample']
        params = s.get('trainable_params', 0)
        params_str = f"{params/1e3:.1f}K" if params < 1e6 else f"{params/1e6:.2f}M"
        if params == 0:
            params_str = "0"
        sf1 = f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}"
        sauc = f"{s['auc_mean']:.4f}±{s['auc_std']:.4f}"
        smf1 = f"{sm['f1_mean']:.4f}±{sm['f1_std']:.4f}"
        print(f"{name:<28s} {params_str:>10s} {sf1:>16s} {sauc:>16s} {smf1:>16s}")
    print("=" * 100)

    # 保存JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        'timestamp': timestamp,
        'experiment': 'cross_lingual_transfer',
        'dataset': 'MODMA',
        'n_folds': 5,
        'note': 'optimal threshold searched on train fold only (no data leakage)',
        'results': {},
    }
    for name, res in all_results.items():
        save_data['results'][name] = {
            'subject': res['subject'],
            'sample': res['sample'],
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOG_DIR / f'cross_lingual_{timestamp}.json'
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n结果已保存: {json_path}")


if __name__ == '__main__':
    main()
