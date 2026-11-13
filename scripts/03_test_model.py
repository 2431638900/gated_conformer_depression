"""
测试脚本 V10.10 - 在测试集上评估模型
自动适配 Windows / Linux，从checkpoint读取模型架构
"""

import sys
import platform
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import warnings

IS_LINUX = platform.system() == 'Linux'

# 过滤警告信息
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*pytree.*')
warnings.filterwarnings('ignore', message='.*not initialized from the model checkpoint.*')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config
from data.dataset import DepressionDataset, collate_fn
from models.depression_model import DepressionDetectionModel
from utils.metrics import compute_metrics, print_metrics
from utils.logger import setup_logger
from utils.visualization import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def test_model(model, test_loader, device, logger, threshold=0.5):
    """
    测试模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        logger: 日志记录器
        threshold: 决策阈值（默认0.5，与验证/测试协议一致）
    
    Returns:
        预测结果和指标
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_ids = []
    
    logger.info("开始测试...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            audio = batch['audio'].to(device)
            
            # 如果有标签
            if 'label' in batch:
                labels = batch['label'].to(device)
                all_labels.extend(labels.cpu().numpy())
            
            participant_ids = batch['participant_id']
            
            # 前向传播
            outputs = model(audio)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 保存完整概率
            all_ids.extend(participant_ids)
    
    # 转换为numpy数组
    all_probs_np = np.array(all_probs)
    all_preds_np = np.array(all_preds)
    
    # 如果有标签，显示调试信息并使用阈值调整
    if len(all_labels) > 0:
        all_labels_np = np.array(all_labels)
        
        # 调试信息：检查预测分布
        logger.info(f"\n🔍 测试集调试信息:")
        logger.info(f"  - 真实标签分布: 0={np.sum(all_labels_np==0)}, 1={np.sum(all_labels_np==1)}")
        logger.info(f"  - 预测标签分布(阈值0.5): 0={np.sum(all_preds_np==0)}, 1={np.sum(all_preds_np==1)}")
        logger.info(f"  - 类别1平均概率: {np.mean(all_probs_np[:, 1]):.4f}")
        logger.info(f"  - 类别1最大概率: {np.max(all_probs_np[:, 1]):.4f}")
        
        # V9.2: 先计算F1@0.5（固定阈值参考）
        f1_at_05 = compute_metrics(all_labels_np, all_preds_np)['f1']
        logger.info(f"  - F1@0.5: {f1_at_05:.4f}")
        
        # 使用调整阈值（与训练一致）
        all_preds_adjusted = (all_probs_np[:, 1] > threshold).astype(int)
        
        logger.info(f"\n🔧 使用调整阈值({threshold:.3f}):")
        logger.info(f"  - 预测标签分布: 0={np.sum(all_preds_adjusted==0)}, 1={np.sum(all_preds_adjusted==1)}")
        
        # 使用调整后的预测计算指标（传入概率以计算AUC）
        metrics = compute_metrics(all_labels_np, all_preds_adjusted, y_prob=all_probs_np[:, 1])
        metrics['f1_at_05'] = f1_at_05
        
        results = {
            'participant_ids': all_ids,
            'predictions': all_preds_adjusted.tolist(),
            'predictions_raw': all_preds_np.tolist(),
            'probabilities': all_probs_np[:, 1].tolist(),
            'labels': all_labels_np.tolist(),
            'metrics': metrics,
            'threshold': threshold
        }
    else:
        # 无标签，使用阈值调整预测
        all_preds_adjusted = (all_probs_np[:, 1] > threshold).astype(int)
        
        results = {
            'participant_ids': all_ids,
            'predictions': all_preds_adjusted.tolist(),
            'predictions_raw': all_preds_np.tolist(),
            'probabilities': all_probs_np[:, 1].tolist(),
            'labels': None,
            'metrics': None,
            'threshold': threshold
        }
    
    return results


def save_predictions(results, output_path):
    """保存预测结果"""
    df = pd.DataFrame({
        'Participant_ID': results['participant_ids'],
        'Prediction': results['predictions'],
        'Probability': results['probabilities']
    })
    
    if results['labels'] is not None:
        df['True_Label'] = results['labels']
    
    df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存: {output_path}")


def parse_args():
    """命令行参数: 指定要测试的 checkpoint 文件"""
    parser = argparse.ArgumentParser(description='V10.12 测试脚本')
    parser.add_argument(
        '--checkpoint', type=str, default='best_model.pth',
        help='要测试的 checkpoint 文件名 (位于 CHECKPOINT_DIR) 或绝对路径; '
             '默认 best_model.pth; 不同微调策略可用 best_model_ft0.pth / best_model_ft1.pth 等',
    )
    parser.add_argument(
        '--log-suffix', type=str, default='',
        help='测试日志/结果文件名后缀 (默认空), 例如 _ft0 / _ft1 便于区分',
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_suffix = args.log_suffix or ''
    logger = setup_logger(
        'test',
        config.LOG_DIR / f'test_{timestamp}{log_suffix}.log'
    )
    
    logger.info("="*60)
    logger.info("Conformer抑郁检测模型测试")
    logger.info(f"Checkpoint: {args.checkpoint} | log_suffix: '{log_suffix}'")
    logger.info("="*60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 测试配置（自动适配平台）
    test_config = {
        'batch_size': 16 if IS_LINUX else 8,  # 推理无梯度，16足够且安全
        'num_workers': 4 if IS_LINUX else 0,
    }
    
    logger.info(f"\nPlatform: {'Linux' if IS_LINUX else 'Windows'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Batch Size: {test_config['batch_size']}")
    logger.info(f"num_workers: {test_config['num_workers']}")
    
    # 加载测试数据 - 使用新的分层划分数据集
    logger.info("\n加载测试数据...")
    stratified_dir = config.get_split_dir()
    logger.info(f"Using official/prepared split: {stratified_dir}")
    test_csv = stratified_dir / 'test' / 'augmented_test.csv'
    
    if not test_csv.exists():
        logger.error(f"测试数据不存在: {test_csv}")
        logger.info("请先运行: python scripts/prepare_official_split.py --daic-woz-root <DAIC-WOZ路径>")
        return
    
    test_df = pd.read_csv(test_csv)
    logger.info(f"✅ 测试集: {len(test_df)} 样本")
    
    # 加载模型checkpoint（提前加载，以便读取训练配置）
    logger.info("\n加载模型...")
    # --checkpoint 支持文件名 (相对 CHECKPOINT_DIR) 或绝对路径
    ckpt_arg = Path(args.checkpoint)
    if ckpt_arg.is_absolute() or ckpt_arg.exists():
        model_path = ckpt_arg
    else:
        model_path = config.CHECKPOINT_DIR / args.checkpoint
    
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}")
        logger.info("请先训练模型: python scripts/02_train_model.py")
        return
    logger.info(f"\u2713 使用 checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # V9.2: 从checkpoint读取音频时长，保持与训练一致
    audio_duration = checkpoint.get('config', {}).get('audio_duration', config.AUDIO_DURATION)
    target_length = int(audio_duration * config.SAMPLE_RATE)
    logger.info(f"🎵 音频配置: {audio_duration}秒, {target_length}采样点")
    
    # 创建数据集
    test_dataset = DepressionDataset(
        test_df,
        sample_rate=config.SAMPLE_RATE,
        target_length=target_length,
        mode='pad'
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config['batch_size'],
        shuffle=False,
        num_workers=test_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=IS_LINUX,
        persistent_workers=True if test_config['num_workers'] > 0 else False,
        prefetch_factor=2 if test_config['num_workers'] > 0 else None,
    )
    
    # 从 checkpoint 读取模型架构配置
    # 优先使用 model_arch_config（新版checkpoint），否则使用默认值
    arch = checkpoint.get('model_arch_config', {})
    
    pretrain_model_type = arch.get('pretrain_model_type', config.PRETRAIN_MODEL_TYPE)
    # 始终使用当前平台的路径（checkpoint可能保存了另一个平台的路径）
    pretrain_model_name = config.PRETRAIN_MODEL_NAME
    pretrain_cache_dir = config.PRETRAIN_CACHE_DIR
    d_model = arch.get('d_model', 256)  # V10.3 default
    num_conformer_layers = arch.get('num_conformer_layers', 6)
    num_attention_heads = arch.get('num_attention_heads', 8)
    conv_kernel_size = arch.get('conv_kernel_size', 31)
    feed_forward_dim = arch.get('feed_forward_dim', 1024)
    num_classes = arch.get('num_classes', 2)
    use_gating = arch.get('use_gating', True)
    use_depression_extractor = arch.get('use_depression_extractor', True)
    dropout = checkpoint.get('config', {}).get('dropout', 0.35)  # V10.5 default
    drop_path_rate = arch.get('drop_path_rate', 0.0)  # V10.12: DropPath (disabled at eval regardless)
    
    logger.info(f"\n📊 模型配置:")
    logger.info(f"  - 预训练模型: {pretrain_model_type} ({pretrain_model_name})")
    logger.info(f"  - d_model: {d_model}")
    logger.info(f"  - num_attention_heads: {num_attention_heads}")
    logger.info(f"  - feed_forward_dim: {feed_forward_dim}")
    logger.info(f"  - dropout: {dropout}")
    
    model = DepressionDetectionModel(
        pretrain_model_type=pretrain_model_type,
        pretrain_model_name=pretrain_model_name,
        pretrain_cache_dir=pretrain_cache_dir,
        pretrain_freeze=True,
        pretrain_layers_to_finetune=0,
        use_depression_extractor=use_depression_extractor,
        d_model=d_model,
        num_conformer_layers=num_conformer_layers,
        num_attention_heads=num_attention_heads,
        conv_kernel_size=conv_kernel_size,
        feed_forward_dim=feed_forward_dim,
        dropout=dropout,
        num_classes=num_classes,
        use_gating=use_gating,
        drop_path_rate=drop_path_rate,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✅ 模型加载成功 (Epoch {checkpoint['epoch']})")
    logger.info(f"✅ 预训练模型: {pretrain_model_type.upper()}")
    
    # V10.5: 主要使用固定阈值0.5（学术标准，更泛化）
    # val校准阈值仅作参考
    threshold = 0.5
    if 'metrics' in checkpoint:
        logger.info(f"验证集最佳F1: {checkpoint['metrics']['f1']:.4f}")
        if 'threshold' in checkpoint['metrics']:
            val_threshold = checkpoint['metrics']['threshold']
            logger.info(f"验证集Youden阈值(仅供参考): {val_threshold:.3f}")
    
    logger.info(f"\n使用决策阈值: {threshold:.3f} (固定0.5，不依赖val校准)")
    results = test_model(model, test_loader, device, logger, threshold=threshold)
    
    # 打印结果
    if results['metrics'] is not None:
        logger.info("\n测试集结果:")
        print_metrics(results['metrics'], "Test Set Metrics")
        
        # 生成测试集混淆矩阵
        logger.info("\n生成测试集混淆矩阵...")
        cm = confusion_matrix(results['labels'], results['predictions'])
        cm_path = config.PLOT_DIR / f'confusion_matrix_test_{timestamp}{log_suffix}.png'
        plot_confusion_matrix(cm, save_path=cm_path)
        logger.info(f"✅ 混淆矩阵已保存: {cm_path}")
        
        # 生成ROC曲线（如果有概率）
        if len(results['probabilities']) > 0:
            logger.info("\n生成ROC曲线...")
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            roc_path = config.PLOT_DIR / f'roc_curve_test_{timestamp}{log_suffix}.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✅ ROC曲线已保存: {roc_path}")
        
        # 生成预测分布图
        logger.info("\n生成预测分布图...")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 按真实标签分组的概率分布
        normal_probs = [results['probabilities'][i] for i in range(len(results['labels'])) 
                       if results['labels'][i] == 0]
        depressed_probs = [results['probabilities'][i] for i in range(len(results['labels'])) 
                          if results['labels'][i] == 1]
        
        axes[0].hist(normal_probs, bins=30, alpha=0.7, label='Normal', color='green', edgecolor='black')
        axes[0].hist(depressed_probs, bins=30, alpha=0.7, label='Depressed', color='red', edgecolor='black')
        axes[0].set_xlabel('Depression Probability')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Prediction Probability Distribution by True Label')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 预测类别分布
        pred_counts = [sum(1 for p in results['predictions'] if p == 0),
                      sum(1 for p in results['predictions'] if p == 1)]
        true_counts = [sum(1 for l in results['labels'] if l == 0),
                      sum(1 for l in results['labels'] if l == 1)]
        
        x = np.arange(2)
        width = 0.35
        axes[1].bar(x - width/2, true_counts, width, label='True', color='skyblue', edgecolor='black')
        axes[1].bar(x + width/2, pred_counts, width, label='Predicted', color='orange', edgecolor='black')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Class Distribution: True vs Predicted')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Normal', 'Depressed'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        dist_path = config.PLOT_DIR / f'prediction_distribution_{timestamp}{log_suffix}.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ 预测分布图已保存: {dist_path}")
        
    else:
        logger.info("\n测试集无标签，仅生成预测结果")
        logger.info(f"预测为抑郁: {sum(results['predictions'])} / {len(results['predictions'])}")
        logger.info(f"平均抑郁概率: {np.mean(results['probabilities']):.4f}")
    
    # 保存预测结果
    output_path = config.LOG_DIR / f'test_predictions_{timestamp}{log_suffix}.csv'
    save_predictions(results, output_path)
    
    # 保存完整结果
    if results['metrics'] is not None:
        results_dict = {
            'timestamp': timestamp,
            'num_samples': len(results['predictions']),
            'metrics': results['metrics'],
            'predictions_file': str(output_path)
        }
        
        results_path = config.LOG_DIR / f'test_results_{timestamp}{log_suffix}.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"测试结果已保存: {results_path}")
    
    logger.info("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
