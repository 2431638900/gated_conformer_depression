# -*- coding: utf-8 -*-
"""
可解释性分析脚本 - 证明模型学习到了正确的抑郁特征

5种可解释性方法:
1. HuBERT层权重可视化 — 哪些声学层对抑郁检测最重要
2. 注意力池化热力图 — 模型关注语音的哪些时间段
3. t-SNE特征空间可视化 — 抑郁/健康样本是否可分
4. Grad-CAM时间归因 — 哪些时间帧对预测贡献最大
5. 跨尺度注意力权重 — 不同卷积尺度(3/7/15/31/63)的贡献

使用: python scripts/09_interpretability.py
"""

import sys
import platform
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

IS_LINUX = platform.system() == 'Linux'

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*not initialized from the model checkpoint.*')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config
from data.dataset import DepressionDataset, collate_fn
from models.depression_model import DepressionDetectionModel

# ============================================================
# matplotlib 配置
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 论文级全局样式
if IS_LINUX:
    rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
else:
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 13
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# 配色方案
COLOR_MDD = '#D32F2F'   # 深红
COLOR_HC  = '#1976D2'   # 深蓝
COLOR_MDD_LIGHT = '#FFCDD2'
COLOR_HC_LIGHT  = '#BBDEFB'

SAVE_DIR = project_root / 'plots' / 'interpretability'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. HuBERT 层权重可视化
# ============================================================
def plot_hubert_layer_weights(model, save_dir):
    """可视化HuBERT 13层可学习聚合权重 — 论文级"""
    print("\n[1/5] HuBERT 层权重可视化...")

    weights = torch.softmax(model.layer_weights.detach().cpu(), dim=0).numpy()
    num_layers = len(weights)
    layer_names = ['CNN'] + [f'L{i}' for i in range(1, num_layers)]

    # 按层级分组配色
    group_colors = (
        ['#5C6BC0']          # CNN: 靛蓝
        + ['#26A69A'] * 4    # L1-4: 青绿
        + ['#FFA726'] * 4    # L5-8: 琥珀
        + ['#EF5350'] * 4    # L9-12: 红
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(num_layers), weights * 100, color=group_colors,
                  edgecolor='white', linewidth=1.2, width=0.75)

    # 在每根柱上标注百分比
    for i, (bar, w) in enumerate(zip(bars, weights)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'{w*100:.1f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if i == np.argmax(weights) else 'normal',
                color='#D32F2F' if i == np.argmax(weights) else '#555')

    # 最高权重柱加粗边框
    max_idx = np.argmax(weights)
    bars[max_idx].set_edgecolor('#D32F2F')
    bars[max_idx].set_linewidth(2.5)

    ax.set_xticks(range(num_layers))
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('Aggregation Weight (%)')
    ax.set_title('Learned HuBERT Layer Weights for Depression Detection')
    ax.set_ylim(0, max(weights) * 100 * 1.25)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    # 语义分区背景 + 图例
    from matplotlib.patches import Patch
    ax.axvspan(-0.5, 0.5, alpha=0.06, color='#5C6BC0')
    ax.axvspan(0.5, 4.5, alpha=0.06, color='#26A69A')
    ax.axvspan(4.5, 8.5, alpha=0.06, color='#FFA726')
    ax.axvspan(8.5, 12.5, alpha=0.06, color='#EF5350')
    legend_elements = [
        Patch(facecolor='#5C6BC0', alpha=0.5, label='Acoustic (CNN)'),
        Patch(facecolor='#26A69A', alpha=0.5, label='Low-level (L1-4)'),
        Patch(facecolor='#FFA726', alpha=0.5, label='Mid-level (L5-8)'),
        Patch(facecolor='#EF5350', alpha=0.5, label='High-level (L9-12)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
             edgecolor='#ccc', fontsize=10)

    plt.tight_layout()
    path = save_dir / 'hubert_layer_weights.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 保存: {path}")
    print(f"  最重要的层: {layer_names[max_idx]} (weight={weights[max_idx]:.4f})")
    return weights


# ============================================================
# 2. 注意力池化热力图 (时间维度)
# ============================================================
def extract_attention_weights(model, dataloader, device, max_samples=200):
    """提取StatisticsAttentionPooling的时间注意力权重"""
    print("\n[2/5] 注意力池化热力图...")

    model.eval()
    attn_weights_all = {'MDD': [], 'HC': []}
    collected = 0

    # Hook to capture attention weights
    attention_maps = []

    def hook_fn(module, input, output):
        # StatisticsAttentionPooling.forward: input[0] = x (batch, seq_len, d_model)
        x = input[0]
        attn_w = module.attention(x)  # (batch, seq_len, 1)
        attn_w = torch.softmax(attn_w, dim=1).detach().cpu()
        attention_maps.append(attn_w)

    handle = model.pooling.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  提取注意力权重"):
            if collected >= max_samples:
                break
            audio = batch['audio'].to(device)
            labels = batch['label'].numpy()
            _ = model(audio)

            attn = attention_maps[-1].squeeze(-1).numpy()  # (batch, seq_len)
            for i in range(len(labels)):
                if collected >= max_samples:
                    break
                key = 'MDD' if labels[i] == 1 else 'HC'
                attn_weights_all[key].append(attn[i])
                collected += 1

    handle.remove()

    # 统一长度 (取最短)
    min_len = min(
        min(len(a) for a in attn_weights_all['MDD']) if attn_weights_all['MDD'] else 999,
        min(len(a) for a in attn_weights_all['HC']) if attn_weights_all['HC'] else 999
    )
    mdd_attn = np.array([a[:min_len] for a in attn_weights_all['MDD']])
    hc_attn = np.array([a[:min_len] for a in attn_weights_all['HC']])

    return mdd_attn, hc_attn, min_len


def _smooth(arr, window=15):
    """简单移动平均平滑"""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def plot_attention_heatmap(mdd_attn, hc_attn, seq_len, save_dir, audio_duration=30):
    """绘制MDD vs HC 的时间注意力对比 — 论文级"""
    time_axis = np.linspace(0, audio_duration, seq_len)

    mdd_mean = _smooth(mdd_attn.mean(axis=0))
    hc_mean = _smooth(hc_attn.mean(axis=0))
    mdd_std = _smooth(mdd_attn.std(axis=0))
    hc_std = _smooth(hc_attn.std(axis=0))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                             gridspec_kw={'height_ratios': [1, 1, 1.3], 'hspace': 0.35})

    # 共享 colorbar 范围
    vmax = max(mdd_attn[:20].max(), hc_attn[:20].max())

    # MDD heatmap
    im0 = axes[0].imshow(mdd_attn[:20], aspect='auto', cmap='Reds',
                         extent=[0, audio_duration, 20, 0], vmin=0, vmax=vmax,
                         interpolation='bilinear')
    axes[0].set_ylabel('Sample Index')
    axes[0].set_title(f'(a) Temporal Attention — MDD (n={len(mdd_attn)})', fontweight='bold')
    axes[0].set_xlim(0, audio_duration)
    fig.colorbar(im0, ax=axes[0], shrink=0.8, pad=0.02, label='Weight')

    # HC heatmap
    im1 = axes[1].imshow(hc_attn[:20], aspect='auto', cmap='Blues',
                         extent=[0, audio_duration, 20, 0], vmin=0, vmax=vmax,
                         interpolation='bilinear')
    axes[1].set_ylabel('Sample Index')
    axes[1].set_title(f'(b) Temporal Attention — HC (n={len(hc_attn)})', fontweight='bold')
    axes[1].set_xlim(0, audio_duration)
    fig.colorbar(im1, ax=axes[1], shrink=0.8, pad=0.02, label='Weight')

    # Mean comparison (smoothed)
    axes[2].plot(time_axis, mdd_mean, color=COLOR_MDD, linewidth=2.2, label='MDD (mean)')
    axes[2].fill_between(time_axis, mdd_mean - mdd_std, mdd_mean + mdd_std,
                         color=COLOR_MDD, alpha=0.12)
    axes[2].plot(time_axis, hc_mean, color=COLOR_HC, linewidth=2.2, label='HC (mean)')
    axes[2].fill_between(time_axis, hc_mean - hc_std, hc_mean + hc_std,
                         color=COLOR_HC, alpha=0.12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Attention Weight')
    axes[2].set_title('(c) Mean Temporal Attention Comparison', fontweight='bold')
    axes[2].legend(loc='upper right', framealpha=0.9)
    axes[2].set_xlim(0, audio_duration)
    axes[2].grid(alpha=0.2, linestyle='--')

    plt.tight_layout()
    path = save_dir / 'attention_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 保存: {path}")

    diff = mdd_attn.mean(axis=0) - hc_attn.mean(axis=0)
    peak_time = time_axis[np.argmax(np.abs(diff))]
    print(f"  MDD vs HC 注意力差异最大时刻: {peak_time:.1f}s")


# ============================================================
# 3. t-SNE / UMAP 特征空间可视化
# ============================================================
def extract_features_at_layers(model, dataloader, device, max_samples=300):
    """提取多层特征: HuBERT原始 → DepExtractor → Conformer → Pooling"""
    print("\n[3/5] t-SNE 特征空间可视化...")

    model.eval()
    features = {
        'hubert': [], 'dep_extractor': [], 'conformer': [], 'pooling': []
    }
    labels_all = []
    collected = 0

    # 注册hooks
    hooks = []
    layer_outputs = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if name == 'pooling':
                layer_outputs[name] = output.detach().cpu()
            else:
                # 时序特征取均值池化
                layer_outputs[name] = output.detach().cpu().mean(dim=1)
        return hook_fn

    hooks.append(model.depression_extractor.register_forward_hook(make_hook('dep_extractor')))
    hooks.append(model.conformer.register_forward_hook(make_hook('conformer')))
    hooks.append(model.pooling.register_forward_hook(make_hook('pooling')))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  提取多层特征"):
            if collected >= max_samples:
                break
            audio = batch['audio'].to(device)
            batch_labels = batch['label'].numpy()

            # 手动获取HuBERT聚合输出
            encoder_output = model.feature_encoder(audio, output_hidden_states=True)
            hidden_states = encoder_output.hidden_states
            stacked = torch.stack(hidden_states, dim=0)
            weights = torch.softmax(model.layer_weights, dim=0)
            hubert_feat = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
            features['hubert'].append(hubert_feat.cpu().mean(dim=1))  # avg pool

            # 完整前向传播（触发hooks）
            _ = model(audio)

            for key in ['dep_extractor', 'conformer', 'pooling']:
                if key in layer_outputs:
                    features[key].append(layer_outputs[key])

            labels_all.extend(batch_labels.tolist())
            collected += len(batch_labels)

    for h in hooks:
        h.remove()

    # 拼接
    result = {}
    for key in features:
        if features[key]:
            result[key] = torch.cat(features[key], dim=0).numpy()[:max_samples]
    labels_all = np.array(labels_all)[:max_samples]

    return result, labels_all


def plot_tsne(features_dict, labels, save_dir):
    """绘制多层t-SNE对比图 — 论文级 2×2网格"""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    layer_names = {
        'hubert': 'HuBERT Output (768-d)',
        'dep_extractor': 'Depression Extractor (256-d)',
        'conformer': 'Conformer Output (256-d)',
        'pooling': 'Attention Pooling (512-d)'
    }
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    available = [k for k in ['hubert', 'dep_extractor', 'conformer', 'pooling']
                 if k in features_dict]
    n = len(available)

    # 计算 Silhouette Score
    ss_scores = {}
    for key in available:
        feat = features_dict[key]
        if len(np.unique(labels)) > 1:
            ss_scores[key] = silhouette_score(feat, labels, sample_size=min(500, len(feat)))

    # 2×2 布局
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for idx, key in enumerate(available):
        feat = features_dict[key]
        perplexity = min(30, len(feat) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     n_iter=1500, learning_rate='auto', init='pca')
        emb = tsne.fit_transform(feat)

        ax = axes[idx]
        hc_mask = labels == 0
        mdd_mask = labels == 1

        # 先画半透明小点
        ax.scatter(emb[hc_mask, 0], emb[hc_mask, 1],
                   c=COLOR_HC, alpha=0.4, s=20, label='HC', edgecolors='none')
        ax.scatter(emb[mdd_mask, 0], emb[mdd_mask, 1],
                   c=COLOR_MDD, alpha=0.4, s=20, label='MDD', edgecolors='none')

        # 添加密度等高线
        from scipy.stats import gaussian_kde
        for mask, color in [(hc_mask, COLOR_HC), (mdd_mask, COLOR_MDD)]:
            pts = emb[mask]
            if len(pts) > 10:
                try:
                    kde = gaussian_kde(pts.T)
                    xmin, xmax = pts[:, 0].min() - 5, pts[:, 0].max() + 5
                    ymin, ymax = pts[:, 1].min() - 5, pts[:, 1].max() + 5
                    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 80),
                                         np.linspace(ymin, ymax, 80))
                    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    ax.contour(xx, yy, zz, levels=3, colors=color, alpha=0.5,
                               linewidths=1.0)
                except Exception:
                    pass

        ss = ss_scores.get(key, 0)
        title = f"{subplot_labels[idx]} {layer_names.get(key, key)}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Silhouette score 标注
        ax.text(0.02, 0.98, f'SS = {ss:.3f}', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#aaa', alpha=0.9))

        # 右下角图例
        ax.legend(loc='lower right', markerscale=1.5, framealpha=0.9,
                  edgecolor='#ccc')

    # 隐藏多余子图
    for idx in range(n, 4):
        axes[idx].set_visible(False)

    fig.suptitle('t-SNE Feature Visualization Across Model Layers',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = save_dir / 'tsne_features.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 保存: {path}")

    print("  各层特征可分性 (Silhouette Score, 越大越好):")
    for key in available:
        print(f"    {layer_names.get(key, key)}: {ss_scores.get(key, 0):.4f}")


# ============================================================
# 4. Grad-CAM 时间归因
# ============================================================
def compute_gradcam(model, dataloader, device, max_samples=100):
    """计算Grad-CAM时间归因图"""
    print("\n[4/5] Grad-CAM 时间归因...")

    model.eval()
    gradcam_all = {'MDD': [], 'HC': []}
    collected = 0

    # Hook 目标: conformer 输出
    activations = {}
    gradients = {}

    def fwd_hook(module, input, output):
        activations['conformer'] = output

    def bwd_hook(module, grad_input, grad_output):
        gradients['conformer'] = grad_output[0]

    fwd_handle = model.conformer.register_forward_hook(fwd_hook)
    bwd_handle = model.conformer.register_full_backward_hook(bwd_hook)

    for batch in tqdm(dataloader, desc="  计算Grad-CAM"):
        if collected >= max_samples:
            break
        audio = batch['audio'].to(device).requires_grad_(False)
        labels = batch['label'].numpy()

        # 逐样本计算 (需要梯度)
        for i in range(len(labels)):
            if collected >= max_samples:
                break
            single_audio = audio[i:i+1]
            single_audio.requires_grad_(False)

            model.zero_grad()
            logits = model(single_audio)
            # 对预测类的logit求导
            pred_class = logits.argmax(dim=1).item()
            target_score = logits[0, pred_class]
            target_score.backward()

            # Grad-CAM: 梯度加权激活
            act = activations['conformer'].detach()  # (1, seq_len, d_model)
            grad = gradients['conformer'].detach()   # (1, seq_len, d_model)

            # 全局平均池化梯度 → 通道权重
            channel_weights = grad.mean(dim=1, keepdim=True)  # (1, 1, d_model)
            cam = (act * channel_weights).sum(dim=2).squeeze(0)  # (seq_len,)
            cam = F.relu(cam).cpu().numpy()

            # 归一化
            if cam.max() > 0:
                cam = cam / cam.max()

            key = 'MDD' if labels[i] == 1 else 'HC'
            gradcam_all[key].append(cam)
            collected += 1

    fwd_handle.remove()
    bwd_handle.remove()

    return gradcam_all


def plot_gradcam(gradcam_dict, save_dir, audio_duration=30):
    """绘制Grad-CAM时间归因对比 — 论文级"""
    all_cams = gradcam_dict['MDD'] + gradcam_dict['HC']
    if not all_cams:
        print("  ⚠️ 无Grad-CAM数据")
        return
    min_len = min(len(c) for c in all_cams)
    # 截掉最后2%帧以去除边界效应
    trim = max(1, int(min_len * 0.02))
    effective_len = min_len - trim
    time_axis = np.linspace(0, audio_duration * effective_len / min_len, effective_len)

    mdd_cams = np.array([c[:effective_len] for c in gradcam_dict['MDD']])
    hc_cams = np.array([c[:effective_len] for c in gradcam_dict['HC']])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'hspace': 0.35})

    # (a) 单样本示例 (前5个)，平滑
    for i in range(min(5, len(mdd_cams))):
        axes[0].plot(time_axis, _smooth(mdd_cams[i], 25), alpha=0.45,
                     color=COLOR_MDD, linewidth=1.0)
    for i in range(min(5, len(hc_cams))):
        axes[0].plot(time_axis, _smooth(hc_cams[i], 25), alpha=0.45,
                     color=COLOR_HC, linewidth=1.0)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_MDD, linewidth=2, label='MDD'),
        Line2D([0], [0], color=COLOR_HC, linewidth=2, label='HC'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    axes[0].set_title('(a) Grad-CAM Individual Examples', fontweight='bold')
    axes[0].set_ylabel('Activation')
    axes[0].set_xlim(0, time_axis[-1])
    axes[0].grid(alpha=0.2, linestyle='--')

    # (b) 均值对比（平滑）
    if len(mdd_cams) > 0:
        mdd_mean = _smooth(mdd_cams.mean(axis=0), 25)
        mdd_std = _smooth(mdd_cams.std(axis=0), 25)
        axes[1].plot(time_axis, mdd_mean, color=COLOR_MDD, linewidth=2.2,
                     label=f'MDD (n={len(mdd_cams)})')
        axes[1].fill_between(time_axis, mdd_mean - mdd_std, mdd_mean + mdd_std,
                             color=COLOR_MDD, alpha=0.12)
    if len(hc_cams) > 0:
        hc_mean = _smooth(hc_cams.mean(axis=0), 25)
        hc_std = _smooth(hc_cams.std(axis=0), 25)
        axes[1].plot(time_axis, hc_mean, color=COLOR_HC, linewidth=2.2,
                     label=f'HC (n={len(hc_cams)})')
        axes[1].fill_between(time_axis, hc_mean - hc_std, hc_mean + hc_std,
                             color=COLOR_HC, alpha=0.12)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Mean Grad-CAM')
    axes[1].set_title('(b) Grad-CAM Temporal Attribution — MDD vs HC', fontweight='bold')
    axes[1].legend(loc='upper right', framealpha=0.9)
    axes[1].set_xlim(0, time_axis[-1])
    axes[1].grid(alpha=0.2, linestyle='--')

    plt.tight_layout()
    path = save_dir / 'gradcam_temporal.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 保存: {path}")


# ============================================================
# 5. 跨尺度注意力权重
# ============================================================
def extract_cross_scale_weights(model, dataloader, device, max_samples=200):
    """提取CrossScaleAttention的门控权重"""
    print("\n[5/5] 跨尺度注意力权重...")

    model.eval()
    gate_weights_all = {'MDD': [], 'HC': []}
    collected = 0

    # Hook CrossScaleAttention
    gate_outputs = []

    def hook_fn(module, input, output):
        # output = gated_features (list of tensors)
        # 需要从 module 内部获取 gate 信号
        # 重新计算 gate signals
        scale_features = input[0]  # list of (batch, ch_i, seq_len)
        gates = []
        for i, feat in enumerate(scale_features):
            pooled = feat.mean(dim=-1)
            projected = module.scale_projs[i](pooled)
            gates.append(projected)

        scale_seq = torch.stack(gates, dim=1)
        attn_out, attn_w = module.cross_attn(scale_seq, scale_seq, scale_seq)
        scale_enhanced = module.norm(scale_seq + attn_out)

        # 提取每个尺度的门控值 (sigmoid)
        gate_values = []
        for i, feat in enumerate(scale_features):
            gate = module.gate_projs[i](scale_enhanced[:, i, :])
            gate = torch.sigmoid(gate).mean(dim=1)  # 平均通道维度 → 标量
            gate_values.append(gate.detach().cpu())

        # (batch, 5)
        gate_mat = torch.stack(gate_values, dim=1).numpy()
        gate_outputs.append(gate_mat)

    handle = model.depression_extractor.cross_scale_attention.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  提取跨尺度门控"):
            if collected >= max_samples:
                break
            audio = batch['audio'].to(device)
            labels = batch['label'].numpy()
            _ = model(audio)

            gates = gate_outputs[-1]  # (batch, 5)
            for i in range(len(labels)):
                if collected >= max_samples:
                    break
                key = 'MDD' if labels[i] == 1 else 'HC'
                gate_weights_all[key].append(gates[i])
                collected += 1

    handle.remove()
    return gate_weights_all


def plot_cross_scale(gate_dict, save_dir):
    """绘制跨尺度注意力权重对比 — 论文级"""
    scale_names = ['Micro\n(k=3)', 'Short\n(k=7)', 'Medium\n(k=15)',
                   'Long\n(k=31)', 'Extra-Long\n(k=63)']

    mdd_gates = np.array(gate_dict['MDD'])
    hc_gates = np.array(gate_dict['HC'])

    mdd_mean = mdd_gates.mean(axis=0)
    hc_mean = hc_gates.mean(axis=0)
    mdd_std = mdd_gates.std(axis=0)
    hc_std = hc_gates.std(axis=0)
    diff = mdd_mean - hc_mean

    x = np.arange(5)
    width = 0.32

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})

    # 上图：门控值柱状图（放大y轴到有效范围）
    ax1.bar(x - width/2, mdd_mean, width, yerr=mdd_std, capsize=5,
            label=f'MDD (n={len(mdd_gates)})', color=COLOR_MDD, alpha=0.85,
            error_kw={'linewidth': 1.5})
    ax1.bar(x + width/2, hc_mean, width, yerr=hc_std, capsize=5,
            label=f'HC (n={len(hc_gates)})', color=COLOR_HC, alpha=0.85,
            error_kw={'linewidth': 1.5})

    ax1.set_xticks(x)
    ax1.set_xticklabels(scale_names)
    ax1.set_ylabel('Gate Value (sigmoid)')
    ax1.set_title('(a) Cross-Scale Attention Gate Values', fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    # 放大y轴以显示差异
    y_min = min(mdd_mean.min(), hc_mean.min()) - max(mdd_std.max(), hc_std.max()) - 0.01
    y_max = max(mdd_mean.max(), hc_mean.max()) + max(mdd_std.max(), hc_std.max()) + 0.01
    ax1.set_ylim(max(0, y_min), y_max)

    # 下图：MDD - HC 差异图
    colors_diff = [COLOR_MDD if d > 0 else COLOR_HC for d in diff]
    bars = ax2.bar(x, diff * 1000, width=0.5, color=colors_diff, alpha=0.85,
                   edgecolor='white', linewidth=1)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scale_names)
    ax2.set_ylabel('Δ Gate (MDD − HC) × 10⁻³')
    ax2.set_title('(b) Gate Value Difference (MDD − HC)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.2, linestyle='--')

    # 标注每个差异值
    for i, (bar, d) in enumerate(zip(bars, diff)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{d*1000:+.1f}', ha='center',
                 va='bottom' if d > 0 else 'top',
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = save_dir / 'cross_scale_attention.png'
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 保存: {path}")

    print("  各尺度门控均值:")
    for i, name in enumerate(scale_names):
        name_clean = name.replace('\n', ' ')
        print(f"    {name_clean}: MDD={mdd_mean[i]:.4f}  HC={hc_mean[i]:.4f}  Δ={diff[i]:+.4f}")


# ============================================================
# 主函数
# ============================================================
def load_model_and_data(device):
    """加载模型和测试数据"""
    # 加载checkpoint
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    arch = checkpoint.get('model_arch_config', {})

    audio_duration = checkpoint.get('config', {}).get('audio_duration', config.AUDIO_DURATION)
    target_length = int(audio_duration * config.SAMPLE_RATE)

    # 构建模型
    model = DepressionDetectionModel(
        pretrain_model_type=arch.get('pretrain_model_type', config.PRETRAIN_MODEL_TYPE),
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
        dropout=checkpoint.get('config', {}).get('dropout', 0.35),
        num_classes=arch.get('num_classes', 2),
        use_gating=arch.get('use_gating', True),
        drop_path_rate=arch.get('drop_path_rate', 0.0),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    val_f1 = checkpoint.get('metrics', {}).get('f1', 0)
    print(f"✅ 模型加载: Epoch {epoch}, Val F1={val_f1:.4f}")

    # 加载测试数据
    stratified_dir = config.get_split_dir()
    test_csv = stratified_dir / 'test' / 'augmented_test.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"测试数据不存在: {test_csv}")

    test_df = pd.read_csv(test_csv)
    test_dataset = DepressionDataset(
        test_df,
        sample_rate=config.SAMPLE_RATE,
        target_length=target_length,
        mode='pad'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8 if IS_LINUX else 4,
        shuffle=False,
        num_workers=4 if IS_LINUX else 0,
        collate_fn=collate_fn,
        pin_memory=IS_LINUX,
    )

    n_mdd = (test_df['PHQ8_Binary'] == 1).sum() if 'PHQ8_Binary' in test_df else '?'
    n_hc = (test_df['PHQ8_Binary'] == 0).sum() if 'PHQ8_Binary' in test_df else '?'
    print(f"✅ 测试集: {len(test_df)} 样本 (MDD={n_mdd}, HC={n_hc})")

    return model, test_loader, device, audio_duration


def main():
    print("=" * 60)
    print("可解释性分析 — 证明模型学习到正确的抑郁特征")
    print("=" * 60)
    print(f"Platform: {'Linux' if IS_LINUX else 'Windows'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, test_loader, device, audio_duration = load_model_and_data(device)

    save_dir = SAVE_DIR
    print(f"\n图表输出目录: {save_dir}")

    # ========================================
    # 1. HuBERT 层权重
    # ========================================
    plot_hubert_layer_weights(model, save_dir)

    # ========================================
    # 2. 注意力池化热力图
    # ========================================
    mdd_attn, hc_attn, seq_len = extract_attention_weights(
        model, test_loader, device, max_samples=200)
    plot_attention_heatmap(mdd_attn, hc_attn, seq_len, save_dir, audio_duration)

    # ========================================
    # 3. t-SNE 特征空间可视化
    # ========================================
    features_dict, labels = extract_features_at_layers(
        model, test_loader, device, max_samples=300)
    plot_tsne(features_dict, labels, save_dir)

    # ========================================
    # 4. Grad-CAM 时间归因
    # ========================================
    gradcam_dict = compute_gradcam(model, test_loader, device, max_samples=100)
    plot_gradcam(gradcam_dict, save_dir, audio_duration)

    # ========================================
    # 5. 跨尺度注意力权重
    # ========================================
    gate_dict = extract_cross_scale_weights(
        model, test_loader, device, max_samples=200)
    plot_cross_scale(gate_dict, save_dir)

    # ========================================
    # 汇总
    # ========================================
    print("\n" + "=" * 60)
    print("可解释性分析完成")
    print("=" * 60)
    print(f"所有图表保存在: {save_dir}")
    print("  1. hubert_layer_weights.png  — HuBERT层权重")
    print("  2. attention_heatmap.png     — 时间注意力热力图")
    print("  3. tsne_features.png         — t-SNE特征可视化")
    print("  4. gradcam_temporal.png      — Grad-CAM时间归因")
    print("  5. cross_scale_attention.png — 跨尺度注意力")


if __name__ == '__main__':
    main()
