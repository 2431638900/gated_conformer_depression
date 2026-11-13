"""
评估指标计算
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    balanced_accuracy_score
)
import torch


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
    
    Returns:
        指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # 如果有概率，计算AUC
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0
    
    return metrics


def compute_multiclass_metrics(y_true, y_pred, num_classes=4):
    """
    计算多分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数
    
    Returns:
        指标字典
    """
    # 每个类别的指标
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # 计算support（每个类别的样本数）
    cm = confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    # 保留旧格式以兼容
    for i in range(num_classes):
        metrics[f'class_{i}_precision'] = per_class_precision[i]
        metrics[f'class_{i}_recall'] = per_class_recall[i]
        metrics[f'class_{i}_f1'] = per_class_f1[i]
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """打印指标"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"\n混淆矩阵:")
            cm = np.array(value)
            print(cm)
        elif isinstance(value, float):
            if 'accuracy' in key:
                print(f"{key:25s}: {value:.2f}%")
            elif 'auc' in key:
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")
    
    print(f"{'='*60}\n")


class EarlyStopping:
    """Early Stopping工具 - 支持监控loss(越小越好)或f1(越大越好)"""
    
    def __init__(self, patience=7, verbose=False, delta=0, mode='max'):
        """
        Args:
            patience: 容忍的epoch数
            verbose: 是否打印信息
            delta: 最小改善量
            mode: 'min'=监控loss(越小越好), 'max'=监控f1/acc(越大越好)
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = None
    
    def __call__(self, value, model=None):
        if self.mode == 'max':
            score = value
        else:
            score = -value
        
        if self.best_score is None:
            self.best_score = score
            self.best_value = value
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_value = value
            self.counter = 0


class AverageMeter:
    """计算和存储平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # 测试
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 0, 1, 1, 1]
    
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, "Binary Classification Metrics")
