"""
Focal Loss实现
用于解决类别不平衡问题

论文: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 类别权重，list或tensor，长度为类别数
        gamma: 聚焦参数，默认2.0。gamma越大，对难分样本的关注越多
        label_smoothing: 标签平滑参数，默认0.0
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
            
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) 模型输出logits
            targets: (batch_size,) 真实标签
        """
        # 计算交叉熵（不做reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        p = torch.exp(-ce_loss)  # p_t
        
        # Focal Loss核心：(1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # 应用focal weight
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # 获取每个样本的alpha
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # 应用标签平滑
        if self.label_smoothing > 0:
            # 计算平滑后的loss
            num_classes = inputs.size(1)
            
            # 创建平滑标签
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # 计算KL散度
            log_probs = F.log_softmax(inputs, dim=1)
            smooth_loss = -(smooth_targets * log_probs).sum(dim=1)
            
            # 混合focal loss和smooth loss
            focal_loss = 0.5 * focal_loss + 0.5 * smooth_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    带动态权重的Focal Loss
    根据类别频率自动调整权重
    """
    
    def __init__(self, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes)
            targets: (batch_size,)
        """
        # 动态计算类别权重
        num_classes = inputs.size(1)
        class_counts = torch.bincount(targets, minlength=num_classes).float()
        
        # 避免除零
        class_counts = torch.clamp(class_counts, min=1.0)
        
        # 计算权重（逆频率）
        total = class_counts.sum()
        alpha = total / (num_classes * class_counts)
        
        # 归一化
        alpha = alpha / alpha.sum() * num_classes
        
        # 使用FocalLoss
        focal_loss = FocalLoss(
            alpha=alpha,
            gamma=self.gamma,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )
        
        return focal_loss(inputs, targets)


if __name__ == '__main__':
    # 测试Focal Loss
    print("测试Focal Loss...")
    
    # 创建测试数据
    batch_size = 8
    num_classes = 2
    
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    
    # 测试不同配置
    print("\n1. 标准Focal Loss (gamma=2.0)")
    criterion1 = FocalLoss(gamma=2.0)
    loss1 = criterion1(inputs, targets)
    print(f"   Loss: {loss1.item():.4f}")
    
    print("\n2. Focal Loss + 类别权重 [1.0, 1.5]")
    criterion2 = FocalLoss(alpha=[1.0, 1.5], gamma=2.0)
    loss2 = criterion2(inputs, targets)
    print(f"   Loss: {loss2.item():.4f}")
    
    print("\n3. Focal Loss + 类别权重 + 标签平滑 (0.1)")
    criterion3 = FocalLoss(alpha=[1.0, 1.5], gamma=2.0, label_smoothing=0.1)
    loss3 = criterion3(inputs, targets)
    print(f"   Loss: {loss3.item():.4f}")
    
    print("\n4. 动态加权Focal Loss")
    criterion4 = WeightedFocalLoss(gamma=2.0, label_smoothing=0.1)
    loss4 = criterion4(inputs, targets)
    print(f"   Loss: {loss4.item():.4f}")
    
    print("\n5. 对比标准CrossEntropyLoss")
    criterion5 = nn.CrossEntropyLoss()
    loss5 = criterion5(inputs, targets)
    print(f"   Loss: {loss5.item():.4f}")
    
    print("\n✅ Focal Loss测试完成！")
