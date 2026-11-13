"""训练脚本 V10.12 - Conformer抑郁检测模型
V10.12优化 (RTX 4090 24GB):
  - 解冻HuBERT最后5层 + 分层学习率
  - HuBERT加权层聚合 (13层可学习权重)
  - 统计注意力池化 (均值+标准差)
  - EMA权重平均 (decay=0.999)
  - Mixup + SpecAugment (R-Drop已禁用)
  - 阈值固定0.5 (val/test统一)
  - 训练延长至80 epochs
  - 模型选择/早停: F1@0.5
  - Label smoothing: 0.1
  - V10.11: 固定随机种子42 (可复现性), LR恢复2e-4, patience 15→10
  - V10.12: DropPath(0~0.1线性, GatedConformer), CrossScaleAttn 1头→4头, DepExtractor输入LayerNorm
  - 自动适配 Windows / Linux
"""

import sys
import platform
import random
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
# V10.3: SWA removed, replaced by EMA
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import torch.nn.functional as F
import warnings

# 过滤警告信息
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*pytree.*')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import config
from data.dataset import DepressionDataset, collate_fn
from models.depression_model import DepressionDetectionModel
from utils.metrics import compute_metrics, EarlyStopping
from utils.logger import setup_logger
from utils.focal_loss import FocalLoss
from utils.visualization import plot_training_history, plot_confusion_matrix

IS_LINUX = platform.system() == 'Linux'


class ModelEMA:
    """V10.3: Exponential Moving Average of model weights
    
    EMA maintains a shadow copy of model parameters, updated each step:
        shadow = decay * shadow + (1 - decay) * param
    
    More stable than SWA, no need for special LR schedule or BN update.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update shadow weights after each optimizer step"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    
    def apply_shadow(self, model):
        """Apply EMA weights to model (for evaluation)"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self, model):
        """Restore original weights (after evaluation)"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


class Trainer:
    """训练器 V10"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 scheduler,
                 device,
                 config_dict,
                 logger,
                 criterion=None):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            config_dict: 配置字典
            logger: 日志记录器
            criterion: 损失函数（可选，默认为CrossEntropyLoss）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config_dict
        self.logger = logger
        self.timestamp = None  # 将在main中设置
        self.class_weight = None  # 将在main中设置
        self.model_arch_config = None  # 将在main中设置
        self.checkpoint_suffix = ''  # 将在main中设置，用于区分不同微调策略的checkpoint
        
        # 混合精度训练
        self.use_amp = config_dict.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            self.logger.info("AMP enabled")
        else:
            self.scaler = None
        
        # 损失函数
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # R-Drop正则化 (V10新增)
        self.use_rdrop = config_dict.get('use_rdrop', False)
        self.rdrop_alpha = config_dict.get('rdrop_alpha', 1.0)
        
        # Early Stopping - 基于F1@0.5监控
        self.early_stopping = EarlyStopping(
            patience=config_dict['early_stopping_patience'],
            verbose=True,
            delta=0.001,
            mode='max'  # 监控F1@0.5（越大越好）
        )
        
        # V10.3: EMA (替代SWA，更稳定)
        self.ema = None
        self.ema_decay = config_dict.get('ema_decay', 0.999)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_recall': [],
            'val_auc': [],
            'learning_rate': []
        }
    
    def _apply_spec_augment(self, audio):
        """SpecAugment风格的时间遮蔽（在原始波形上操作）"""
        if not self.config.get('use_spec_augment', False) or not self.model.training:
            return audio
        
        batch_size, length = audio.shape
        num_masks = self.config.get('time_mask_num', 2)
        mask_width = self.config.get('time_mask_width', 20)
        # 在波形层面，mask_width对应帧数 * hop_length（约320采样点/帧）
        mask_samples = mask_width * 320
        
        for _ in range(num_masks):
            starts = torch.randint(0, max(1, length - mask_samples), (batch_size,))
            for b in range(batch_size):
                audio[b, starts[b]:starts[b] + mask_samples] = 0.0
        return audio
    
    def _mixup_data(self, audio, labels):
        """Mixup数据增强：在输入空间混合样本"""
        alpha = self.config.get('mixup_alpha', 0.3)
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        
        batch_size = audio.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_audio = lam * audio + (1 - lam) * audio[index]
        labels_a, labels_b = labels, labels[index]
        return mixed_audio, labels_a, labels_b, lam
    
    def _mixup_criterion(self, outputs, labels_a, labels_b, lam):
        """Mixup的混合损失"""
        return lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
    
    def _compute_rdrop_loss(self, logits1, logits2):
        """R-Drop: KL散度一致性正则化"""
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)
        p_dist = F.softmax(logits1, dim=-1)
        q_dist = F.softmax(logits2, dim=-1)
        kl_pq = F.kl_div(p, q_dist, reduction='batchmean')
        kl_qp = F.kl_div(q, p_dist, reduction='batchmean')
        return (kl_pq + kl_qp) / 2
    
    def train_epoch(self, epoch):
        """训练一个epoch（支持梯度累积 + Mixup + SpecAugment + R-Drop）"""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0
        use_mixup = self.config.get('use_mixup', False)
        
        # 梯度累积步数
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for i, batch in enumerate(pbar):
            # 获取数据
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # SpecAugment时间遮蔽
            audio = self._apply_spec_augment(audio)
            
            # Mixup数据增强
            if use_mixup:
                audio, labels_a, labels_b, lam = self._mixup_data(audio, labels)
            
            # 前向传播 - 混合精度训练
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(audio)
                    if use_mixup:
                        loss = self._mixup_criterion(outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                    # V10.2: R-Drop正则化 - 与Mixup兼容
                    # 两次前向传播使用相同输入(含Mixup)，dropout产生不同激活
                    if self.use_rdrop:
                        outputs2 = self.model(audio)
                        if use_mixup:
                            loss2 = self._mixup_criterion(outputs2, labels_a, labels_b, lam)
                        else:
                            loss2 = self.criterion(outputs2, labels)
                        rdrop_loss = self._compute_rdrop_loss(outputs, outputs2)
                        loss = (loss + loss2) / 2 + self.rdrop_alpha * rdrop_loss
            else:
                outputs = self.model(audio)
                if use_mixup:
                    loss = self._mixup_criterion(outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                # V10.2: R-Drop与Mixup兼容
                if self.use_rdrop:
                    outputs2 = self.model(audio)
                    if use_mixup:
                        loss2 = self._mixup_criterion(outputs2, labels_a, labels_b, lam)
                    else:
                        loss2 = self.criterion(outputs2, labels)
                    rdrop_loss = self._compute_rdrop_loss(outputs, outputs2)
                    loss = (loss + loss2) / 2 + self.rdrop_alpha * rdrop_loss
            
            # 梯度累积：损失除以累积步数
            loss = loss / accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (i + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    self.optimizer.step()
                self.optimizer.zero_grad()
                # V10.3: EMA update after each optimizer step
                if self.ema is not None:
                    self.ema.update(self.model)
            
            # 统计
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            if use_mixup:
                correct += (lam * predicted.eq(labels_a).sum().item() 
                          + (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []  # 保存概率用于调试
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(audio)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 调试信息：检查预测分布
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        
        self.logger.info(f"\n🔍 验证集调试信息:")
        self.logger.info(f"  - 真实标签分布: 0={np.sum(all_labels_np==0)}, 1={np.sum(all_labels_np==1)}")
        self.logger.info(f"  - 预测标签分布(阈值0.5): 0={np.sum(all_preds_np==0)}, 1={np.sum(all_preds_np==1)}")
        self.logger.info(f"  - 类别1平均概率: {np.mean(all_probs_np[:, 1]):.4f}")
        self.logger.info(f"  - 类别1最大概率: {np.max(all_probs_np[:, 1]):.4f}")
        
        # V10.6: 统一使用固定阈值0.5（与测试完全一致，all_preds_np已经是@0.5）
        metrics = compute_metrics(all_labels_np, all_preds_np, y_prob=all_probs_np[:, 1])
        metrics['threshold'] = 0.5
        metrics['f1_at_05'] = metrics['f1']  # 与主指标相同
        
        self.logger.info(f"  - F1@0.5: {metrics['f1']:.4f}")
        
        return avg_loss, metrics
    
    def setup_ema(self):
        """V10.3: 初始化EMA（在main中调用）"""
        self.ema = ModelEMA(self.model, decay=self.ema_decay)
        self.logger.info(f"EMA initialized, decay={self.ema_decay}")
    
    def train(self):
        """完整训练流程 (V10.3: EMA替代SWA)"""
        self.logger.info("="*60)
        self.logger.info("开始训练 V10.10")
        self.logger.info("="*60)
        self.logger.info(f"训练样本: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证样本: {len(self.val_loader.dataset)}")
        self.logger.info(f"Batch Size: {self.config['batch_size']}")
        self.logger.info(f"学习率: {self.config['learning_rate']}")
        self.logger.info(f"Epochs: {self.config['num_epochs']}")
        self.logger.info("="*60)
        
        best_f1 = 0.0  # V10.9: 回F1@0.5选模（AUC选模会丢弃F1最优epoch，V10.8分析证明）
        best_auc_record = 0.0
        
        for epoch in range(self.config['num_epochs']):
            # 训练 (EMA在train_epoch中每步更新)
            train_loss, train_acc = self.train_epoch(epoch)
            
            # V10.3: 使用EMA权重进行验证
            if self.ema is not None:
                self.ema.apply_shadow(self.model)
            
            val_loss, val_metrics = self.validate()
            
            if self.ema is not None:
                self.ema.restore(self.model)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_auc'].append(val_metrics.get('auc', 0.0))
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 打印结果
            current_auc = val_metrics.get('auc', 0.0)
            current_f1_at_05 = val_metrics.get('f1_at_05', 0.0)
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['num_epochs']} [EMA]")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            self.logger.info(f"Val F1@0.5: {current_f1_at_05:.4f}, Val Recall: {val_metrics['recall']:.4f}, AUC: {current_auc:.4f}")
            self.logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # V10.9: F1选模 (保存时用EMA权重)
            if current_f1_at_05 > best_f1:
                best_f1 = current_f1_at_05
                best_auc_record = current_auc
                # V10.3: 保存EMA权重作为best model
                if self.ema is not None:
                    self.ema.apply_shadow(self.model)
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                if self.ema is not None:
                    self.ema.restore(self.model)
                self.logger.info(f"Save best EMA model (F1@0.5: {best_f1:.4f}, AUC: {current_auc:.4f})")
            
            # V10.9: Early Stopping基于F1@0.5
            self.early_stopping(current_f1_at_05)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered! (F1@0.5 no improve for {self.early_stopping.patience} epochs)")
                break
        
        self.logger.info("="*60)
        self.logger.info(f"Training done! Best F1@0.5: {best_f1:.4f}, AUC: {best_auc_record:.4f}")
        self.logger.info("="*60)
        
        return self.history
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'model_arch_config': self.model_arch_config,  # 保存模型架构参数
            'history': self.history
        }
        
        suffix = self.checkpoint_suffix or ''

        # 保存最新模型
        checkpoint_path = config.CHECKPOINT_DIR / f'latest_model{suffix}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = config.CHECKPOINT_DIR / f'best_model{suffix}.pth'
            torch.save(checkpoint, best_path)
    
    def save_final_backup(self):
        """训练结束后保存最终的带配置信息的副本"""
        suffix = self.checkpoint_suffix or ''
        best_path = config.CHECKPOINT_DIR / f'best_model{suffix}.pth'
        if best_path.exists() and self.timestamp and self.class_weight is not None:
            checkpoint = torch.load(best_path)
            metrics = checkpoint['metrics']
            
            # 构建文件名: best_model{suffix}_时间戳_权重_F1.pth
            backup_name = f'best_model{suffix}_{self.timestamp}_w{self.class_weight:.2f}_f1at05{metrics.get("f1_at_05", metrics["f1"]):.4f}.pth'
            backup_path = config.CHECKPOINT_DIR / backup_name
            torch.save(checkpoint, backup_path)
            self.logger.info(f"\n📦 训练完成，保存最终副本: {backup_name}")


def parse_args():
    """命令行参数: 控制 HuBERT 微调层数 + checkpoint 文件名后缀"""
    parser = argparse.ArgumentParser(description='V10.12 训练脚本')
    parser.add_argument(
        '--finetune-layers', type=int, default=0,
        help='HuBERT 末端解冻层数 (0=全冻结, 默认; 1=只微调最后1层; 5=旧主模型配置)',
    )
    parser.add_argument(
        '--checkpoint-suffix', type=str, default=None,
        help='checkpoint 文件名后缀 (默认 _ft{N}, N=0 时为空串 → 覆盖 best_model.pth)',
    )
    parser.add_argument(
        '--pretrain-lr-divisor', type=float, default=None,
        help='HuBERT 解冻层的分层学习率除数 (默认: N≥4→15, N≤2→30; N=0 时无意义)',
    )
    return parser.parse_args()


def main():
    """主函数 - V10.12 (RTX 4090 24GB 优化)"""
    args = parse_args()

    # 解析微调层数 → freeze 策略
    n_ft = max(0, int(args.finetune_layers))
    pretrain_freeze = (n_ft == 0)

    # 默认 suffix: 若用户未显式指定，按 _ft{N} 区分
    # 新主模型 = N=0（HuBERT 全冻结，F1=0.9169, AUC=0.9874, 18.44M trainable），所以 N==0 时 suffix='' 覆盖 best_model.pth
    # 其它 N 用 _ft{N} 作为对照实验，不覆盖主 ckpt
    if args.checkpoint_suffix is None:
        checkpoint_suffix = '' if n_ft == 0 else f'_ft{n_ft}'
    else:
        checkpoint_suffix = args.checkpoint_suffix

    # 默认 pretrain_lr 除数：N>=4 用 15（大微调更激进），N<=2 用 30（少微调更稳）
    if args.pretrain_lr_divisor is not None:
        pretrain_lr_divisor = float(args.pretrain_lr_divisor)
    else:
        pretrain_lr_divisor = 15.0 if n_ft >= 4 else 30.0

    # 设置随机种子 (V10.11: 完整固定, 确保可复现性)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # deterministic=True时需关闭benchmark
    
    # 创建目录
    config.create_dirs()
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        'train',
        config.LOG_DIR / f'train_{timestamp}{checkpoint_suffix}.log'
    )
    
    ft_tag = 'Freeze' if pretrain_freeze else f'Finetune({n_ft}L)'
    logger.info("="*60)
    logger.info(f"V10.12 Training - HuBERT {ft_tag} + EMA(0.999) + Mixup + WeightedLayerAgg + StatPooling + F1-Selection + Seed42 + LR2e-4 + DropPath(0.1) + CrossScaleAttn4H + InputLN")
    logger.info(f"Platform: {'Linux' if IS_LINUX else 'Windows'}")
    logger.info(f"Finetune layers: {n_ft} | checkpoint suffix: '{checkpoint_suffix}' | pretrain_lr_divisor: {pretrain_lr_divisor}")
    logger.info("="*60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ==================== V10训练配置 ====================
    # V10.4变化: unfreeze 3→5层, warmup 5→8ep, dropout 0.45→0.35, weight 2.0→1.7, ema 0.999→0.9995
    num_workers = 4 if IS_LINUX else 0
    pin_memory = IS_LINUX
    
    train_config = {
        # 基础训练参数
        'batch_size': 8,  # 实际8, 累积4步=有效32
        'audio_duration': 30,
        'learning_rate': 2e-4,   # V10.11: 恢复V10.4的2e-4 (V10.10降LR导致val-test gap扩大: val0.9327→test0.9112)
        'num_epochs': 80,  # V10.5: 60→80 (V10.4未收敛, EarlyStopping=14/15)
        'warmup_epochs': 8,   # V10.4: 5→8, V10.11保持8ep (恢复V10.4配置)
        'early_stopping_patience': 10,  # V10.11: 15→10 (V10.10最优在ep33, 节省训练时间)
        'dropout': 0.35,   # V10.4: 0.45→0.35 (泛化稳定, 适当减少过正则)
        'weight_decay': 1e-3,  # V10.1: 5e-4→1e-3 (更强L2正则)
        'label_smoothing': 0.1,   # V10.8: 回V10.4/V10.6值 (0.05过尖锐, 没有帮助泛化)
        'gradient_accumulation_steps': 4,
        'max_grad_norm': 1.0,
        'use_amp': True,
        # Mixup
        'use_mixup': True,
        'mixup_alpha': 0.2,   # V10.4: 0.3→0.2 (降低mixup强度, 配合更长warmup)
        # SpecAugment
        'use_spec_augment': True,
        'time_mask_num': 4,  # V10.1: 3→4
        'time_mask_width': 35,  # V10.1: 27→35 (更大遮蔽)
        # 解冻HuBERT层数 (V10.13 默认全冻结 N=0；通过 --finetune-layers 控制，例如 5 = 旧主模型配置)
        'pretrain_freeze': pretrain_freeze,
        'pretrain_layers_to_finetune': n_ft,
        # R-Drop正则化
        'use_rdrop': False,  # V10.3: 禁用R-Drop (V10.2证明与Mixup不兼容)
        'rdrop_alpha': 1.0,
        # V10.3: EMA替代SWA
        'ema_decay': 0.999,   # V10.8: 回V10.4最佳值 (0.9999导致EMA滞后过多导致训练浮动)
        # 损失函数
        'use_focal_loss': True,
        'focal_gamma': 2.0,  # V10.1: 1.5→2.0 (更关注困难样本)
        # 预训练模型配置
        'pretrain_model_type': config.PRETRAIN_MODEL_TYPE,
        'pretrain_model_name': config.PRETRAIN_MODEL_NAME,
        'pretrain_cache_dir': config.PRETRAIN_CACHE_DIR,
        # V10.12: Stochastic Depth (DropPath) - GatedConformer每层线性增长, 第0层=0, 末层=drop_path_rate
        'drop_path_rate': 0.1,
    }
    
    logger.info(f"\nHardware config:")
    logger.info(f"  Batch Size: {train_config['batch_size']}")
    logger.info(f"  num_workers: {num_workers}")
    logger.info(f"  pin_memory: {pin_memory}")
    
    # 加载数据
    logger.info("\nLoading data...")
    stratified_dir = config.get_split_dir()
    train_df = pd.read_csv(stratified_dir / 'train' / 'augmented_train.csv')
    val_df = pd.read_csv(stratified_dir / 'val' / 'augmented_val.csv')
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    
    # 类别权重
    class_counts = train_df['PHQ8_Binary'].value_counts().sort_index()
    total = len(train_df)
    raw_weights = total / (len(class_counts) * class_counts.values)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights[0]
    
    original_weight = class_weights[1].item()
    depressed_weight = 1.70  # V10.4: 2.0→1.7 (FP=47>FN=22, 进一步提升Precision)
    class_weights[1] = depressed_weight
    
    logger.info(f"\nClass distribution:")
    logger.info(f"  Normal: {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
    logger.info(f"  Depressed: {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
    logger.info(f"  Weight: [1.00, {depressed_weight:.2f}] (raw: {original_weight:.2f})")
    
    # 创建数据集
    audio_duration = train_config.get('audio_duration', config.AUDIO_DURATION)
    target_length = int(audio_duration * config.SAMPLE_RATE)
    logger.info(f"\nAudio: {audio_duration}s, {target_length} samples")
    
    train_dataset = DepressionDataset(
        train_df,
        sample_rate=config.SAMPLE_RATE,
        target_length=target_length,
        mode='random_crop'
    )
    
    val_dataset = DepressionDataset(
        val_df,
        sample_rate=config.SAMPLE_RATE,
        target_length=target_length,
        mode='pad'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # ==================== V10 模型架构 ====================
    logger.info("\nCreating model (V10.12)...")
    logger.info(f"Pretrained: {train_config['pretrain_model_type']}")
    
    model_arch_config = {
        'pretrain_model_type': train_config['pretrain_model_type'],
        'pretrain_model_name': train_config['pretrain_model_name'],
        'pretrain_cache_dir': train_config['pretrain_cache_dir'],
        'd_model': 256,              # V10: 160→256 (+60%)
        'num_conformer_layers': 6,   # V10: 4→6
        'num_attention_heads': 8,    # V10: 5→8
        'conv_kernel_size': 31,      # V10: 15→31
        'feed_forward_dim': 1024,    # V10: 640→1024
        'num_classes': 2,
        'use_gating': True,
        'use_depression_extractor': True,
        'drop_path_rate': train_config.get('drop_path_rate', 0.0),  # V10.12: Stochastic Depth
    }
    
    model = DepressionDetectionModel(
        pretrain_model_type=model_arch_config['pretrain_model_type'],
        pretrain_model_name=model_arch_config['pretrain_model_name'],
        pretrain_cache_dir=model_arch_config['pretrain_cache_dir'],
        pretrain_freeze=train_config.get('pretrain_freeze', True),
        pretrain_layers_to_finetune=train_config.get('pretrain_layers_to_finetune', 0),
        use_depression_extractor=model_arch_config['use_depression_extractor'],
        d_model=model_arch_config['d_model'],
        num_conformer_layers=model_arch_config['num_conformer_layers'],
        num_attention_heads=model_arch_config['num_attention_heads'],
        conv_kernel_size=model_arch_config['conv_kernel_size'],
        feed_forward_dim=model_arch_config['feed_forward_dim'],
        dropout=train_config['dropout'],
        num_classes=model_arch_config['num_classes'],
        use_gating=model_arch_config['use_gating'],
        drop_path_rate=model_arch_config['drop_path_rate'],
    ).to(device)
    
    logger.info("\nV10.12 Config Summary:")
    arch_ft_desc = f"freeze-all" if pretrain_freeze else f"finetune last {n_ft}"
    logger.info(f"  Architecture: HuBERT({arch_ft_desc}, weighted 13-layer) + DepExtractor(InputLN+4H-CrossScale) + GatedConformer(6L+DropPath) + StatPooling")
    logger.info(f"  d_model={model_arch_config['d_model']}, heads={model_arch_config['num_attention_heads']}, ffn={model_arch_config['feed_forward_dim']}")
    logger.info(f"  Audio: {train_config['audio_duration']}s, Batch: {train_config['batch_size']}")
    logger.info(f"  Augment: Mixup(a={train_config['mixup_alpha']}), SpecAug({train_config['time_mask_num']}x{train_config['time_mask_width']})")
    logger.info(f"  Regularize: Dropout={train_config['dropout']}, LS={train_config['label_smoothing']}, Weight={depressed_weight} (R-Drop disabled)")
    logger.info(f"  EMA: decay={train_config['ema_decay']}")
    logger.info(f"  Loss: FocalLoss(gamma={train_config.get('focal_gamma', 2.0)})")
    logger.info(f"  Selection: F1@0.5-based | LR={train_config['learning_rate']} (V10.12: DropPath={train_config['drop_path_rate']}, Seed42)")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nParams: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
    
    # ==================== 优化器 (分层学习率) ====================
    if train_config.get('pretrain_layers_to_finetune', 0) > 0:
        pretrain_params = []
        new_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'feature_encoder' in name:
                pretrain_params.append(param)
            else:
                new_params.append(param)
        pretrain_lr = train_config['learning_rate'] / pretrain_lr_divisor  # V10.4: 默认/15; N<=2时/30, 由 --pretrain-lr-divisor 控制
        optimizer = AdamW([
            {'params': pretrain_params, 'lr': pretrain_lr, 'weight_decay': 1e-4},
            {'params': new_params, 'lr': train_config['learning_rate'], 'weight_decay': train_config['weight_decay']},
        ])
        logger.info(f"\nLayered LR: HuBERT={pretrain_lr:.1e} ({len(pretrain_params)} tensors), New={train_config['learning_rate']:.1e} ({len(new_params)} tensors)")
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        logger.info(f"\nOptimizer: AdamW, lr={train_config['learning_rate']}, {len(trainable)} tensors")
    
    # ==================== 学习率调度 (V10.3: 无SWA阶段, cosine到底) ====================
    warmup_epochs = train_config.get('warmup_epochs', 5)
    cosine_end = train_config['num_epochs']
    
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs)
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_end - warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    logger.info(f"\nLR Schedule: Warmup({warmup_epochs}ep) + Cosine(→epoch {cosine_end}) + EMA(decay={train_config['ema_decay']})")
    
    # ==================== 损失函数 ====================
    if train_config.get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=class_weights.tolist(),
            gamma=train_config.get('focal_gamma', 2.0),
            label_smoothing=train_config['label_smoothing']
        )
        logger.info(f"\nLoss: FocalLoss(gamma={train_config.get('focal_gamma', 2.0)}, alpha={class_weights.tolist()}, ls={train_config['label_smoothing']})")
    else:
        class_weights_adjusted = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_adjusted,
            label_smoothing=train_config['label_smoothing']
        )
        logger.info(f"\nLoss: WeightedCE + LS={train_config['label_smoothing']}")
    
    # ==================== 创建训练器 ====================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config_dict=train_config,
        logger=logger
    )
    
    trainer.timestamp = timestamp
    trainer.class_weight = depressed_weight
    trainer.model_arch_config = model_arch_config
    trainer.checkpoint_suffix = checkpoint_suffix  # 透传给 save_checkpoint/save_final_backup
    
    # V10.3: 初始化EMA
    trainer.setup_ema()
    
    # ==================== 训练 ====================
    history = trainer.train()
    
    # 保存最终副本
    trainer.save_final_backup()
    
    # 保存训练历史
    history_path = config.LOG_DIR / f'history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nHistory saved: {history_path}")
    
    # 训练曲线图
    plot_path = config.PLOT_DIR / f'training_curves_{timestamp}.png'
    plot_training_history(history, save_path=plot_path)
    logger.info(f"Curves saved: {plot_path}")
    
    # 混淆矩阵
    logger.info("\nGenerating validation confusion matrix...")
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            outputs = model(audio)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    best_path = config.CHECKPOINT_DIR / f'best_model{checkpoint_suffix}.pth'
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        best_threshold = best_ckpt.get('metrics', {}).get('threshold', 0.5)
    else:
        best_threshold = 0.5
    
    all_probs_np = np.array(all_probs)
    all_preds = (all_probs_np > best_threshold).astype(int)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = config.PLOT_DIR / f'confusion_matrix_val_{timestamp}.png'
    plot_confusion_matrix(cm, save_path=cm_path)
    logger.info(f"Confusion matrix saved: {cm_path} (threshold={best_threshold:.3f})")
    
    logger.info("\nTraining complete!")
    logger.info("Next: python scripts/03_test_model.py")


if __name__ == "__main__":
    main()
