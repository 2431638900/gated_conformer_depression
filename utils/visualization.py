"""
可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json


def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Loss曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 + Recall曲线
    axes[0, 2].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
    if 'val_recall' in history:
        axes[0, 2].plot(history['val_recall'], label='Val Recall', linewidth=2, color='red', linestyle='--')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('Validation F1 & Recall')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # AUC曲线（如果有）
    if 'val_auc' in history and len(history['val_auc']) > 0:
        axes[1, 0].plot(history['val_auc'], label='Val AUC', linewidth=2, color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        # 没有AUC时显示F1单独曲线
        axes[1, 0].plot(history['val_f1'], label='Val F1', linewidth=2, color='green')
        best_epoch = np.argmax(history['val_f1'])
        best_f1 = history['val_f1'][best_epoch]
        axes[1, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best F1: {best_f1:.4f} (Epoch {best_epoch+1})')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 (Best Marked)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 1].plot(history['learning_rate'], linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 过拟合监控：Train-Val Loss差距
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    gap = val_loss - train_loss
    axes[1, 2].plot(gap, linewidth=2, color='crimson')
    axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 2].fill_between(range(len(gap)), gap, alpha=0.2, color='crimson')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Val Loss - Train Loss')
    axes[1, 2].set_title('Overfitting Monitor (Gap)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm, class_names=['Normal', 'Depressed'], save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热图
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    
    # 添加原始数量
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j+0.5, i+0.7, f'({cm[i, j]})',
                    ha='center', va='center', fontsize=10, color='gray')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ablation_comparison(results_file, save_path=None):
    """
    绘制消融实验对比图
    
    Args:
        results_file: 结果JSON文件
        save_path: 保存路径
    """
    # 加载结果
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 提取数据
    exp_names = [r['experiment'] for r in results]
    f1_scores = [r['best_val_f1'] for r in results]
    accuracies = [r['best_val_acc'] for r in results]
    
    # 简化实验名称
    exp_names_short = [name.split('_')[0] + '_' + name.split('_')[1] for name in exp_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1对比
    bars1 = axes[0].bar(range(len(exp_names)), f1_scores, color='skyblue', edgecolor='navy')
    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('Ablation Study: F1 Score Comparison')
    axes[0].set_xticks(range(len(exp_names)))
    axes[0].set_xticklabels(exp_names_short, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, f1_scores)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Accuracy对比
    bars2 = axes[1].bar(range(len(exp_names)), accuracies, color='lightcoral', edgecolor='darkred')
    axes[1].set_xlabel('Experiment')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Ablation Study: Accuracy Comparison')
    axes[1].set_xticks(range(len(exp_names)))
    axes[1].set_xticklabels(exp_names_short, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars2, accuracies)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"消融实验对比图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(metrics_list, labels, save_path=None):
    """
    绘制多个模型的指标对比雷达图
    
    Args:
        metrics_list: 指标列表
        labels: 模型标签
        save_path: 保存路径
    """
    # 指标名称
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    # 创建雷达图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # 绘制每个模型
    colors = ['b', 'r', 'g', 'orange', 'purple']
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        values = [
            metrics.get('accuracy', 0) / 100,
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('auc', 0)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Model Performance Comparison', y=1.08)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比雷达图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 测试
    print("可视化工具模块")
    
    # 测试训练历史绘制
    dummy_history = {
        'train_loss': np.random.rand(20) * 0.5 + 0.3,
        'val_loss': np.random.rand(20) * 0.5 + 0.4,
        'train_acc': np.random.rand(20) * 10 + 85,
        'val_acc': np.random.rand(20) * 10 + 80,
        'val_f1': np.random.rand(20) * 0.2 + 0.7,
        'learning_rate': [1e-4 * (0.95 ** i) for i in range(20)]
    }
    
    plot_training_history(dummy_history, 'test_history.png')
    
    # 测试混淆矩阵
    cm = np.array([[45, 5], [8, 22]])
    plot_confusion_matrix(cm, save_path='test_cm.png')
    
    print("✅ 测试完成！")
