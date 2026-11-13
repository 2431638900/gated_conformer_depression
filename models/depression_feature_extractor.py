"""
抑郁特征提取器 V10.12
从 HuBERT/Wav2Vec2的通用特征中提取抑郁特定特征

V10优化 (RTX 4090 24GB):
  - 默认output_dim 160→256，匹配V10 d_model
  - 新增SE通道注意力：自适应加权不同时间尺度的重要性
  - FFN扩展比 2x→4x，更大容量
  - 降低内部dropout（外部已有Mixup+R-Drop正则化）
  - 多尺度卷积核扩展: [5,11,21,41] → [3,7,15,31,63]（5尺度）

V10.12新增:
  - CrossScaleAttention: 1头 → 4头，建模更丰富的尺度间交互模式
  - 输入LayerNorm: HuBERT特征进入多尺度卷积前首先归一化，稳定梯度流
"""

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    SE通道注意力模块
    自适应学习不同卷积尺度的重要性权重
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),        # 全局平均池化
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            x * weights: (batch, channels, seq_len)
        """
        weights = self.se(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * weights


class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力模块 (V10.1核心创新)
    
    与SE通道注意力的关键区别:
    - SE: 独立地对每个通道计算权重，无法建模尺度间关系
    - 跨尺度注意力: 显式建模5个时间尺度之间的依赖关系
    
    临床动机: 抑郁症在语音中的表现跨越多个时间尺度，
    且不同尺度的特征是相互关联的:
    - 长时语调单调(k=63) + 短时停顿增多(k=7) → 强抑郁信号
    - 微观语音抖动(k=3) + 中时能量下降(k=15) → 情绪低落指标
    
    实现:
    1. 每个尺度通过全局池化+投影得到"尺度表示"
    2. 5个尺度表示通过自注意力进行信息交互
    3. 交互后的表示生成门控信号，反向调制原始特征
    """
    def __init__(self, scale_channels, attn_dim=64):
        """
        Args:
            scale_channels: 各尺度的通道数列表, e.g., [51, 51, 51, 51, 52]
            attn_dim: 跨尺度注意力的隐藏维度
        """
        super().__init__()
        self.num_scales = len(scale_channels)
        self.attn_dim = attn_dim
        
        # 每个尺度投影到统一的注意力空间
        self.scale_projs = nn.ModuleList([
            nn.Linear(ch, attn_dim) for ch in scale_channels
        ])
        
        # 跨尺度自注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(attn_dim)
        
        # 注意力输出 → 每个尺度的门控权重
        self.gate_projs = nn.ModuleList([
            nn.Linear(attn_dim, ch) for ch in scale_channels
        ])
    
    def forward(self, scale_features):
        """
        Args:
            scale_features: list of (batch, ch_i, seq_len), 各尺度的卷积特征
        Returns:
            gated_features: list of (batch, ch_i, seq_len), 跨尺度增强后的特征
        """
        # 1. 每个尺度全局池化 + 投影到注意力空间
        scale_reps = []
        for i, feat in enumerate(scale_features):
            pooled = feat.mean(dim=-1)  # (batch, ch_i)
            projected = self.scale_projs[i](pooled)  # (batch, attn_dim)
            scale_reps.append(projected)
        
        # (batch, num_scales, attn_dim)
        scale_seq = torch.stack(scale_reps, dim=1)
        
        # 2. 跨尺度自注意力交互
        attn_out, _ = self.cross_attn(scale_seq, scale_seq, scale_seq)
        scale_enhanced = self.norm(scale_seq + attn_out)
        
        # 3. 生成每个尺度的门控信号并调制原始特征
        gated_features = []
        for i, feat in enumerate(scale_features):
            gate = self.gate_projs[i](scale_enhanced[:, i, :])  # (batch, ch_i)
            gate = torch.sigmoid(gate).unsqueeze(-1)  # (batch, ch_i, 1)
            gated_features.append(feat * gate)
        
        return gated_features


class DepressionFeatureExtractor(nn.Module):
    """
    抑郁特征提取器 V10.3
    
    设计思路:
    1. 多尺度卷积: 5个尺度捕捉不同时间范围的抑郁特征
    2. SE通道注意力: 自适应加权不同尺度的重要性
    3. 时序注意力: 关注抑郁相关的时间段
    4. 特征增强: 4倍FFN学习抑郁特定的特征组合
    
    参数量:
    - 输入: 768维 (HuBERT输出)
    - 输出: 256维 (给Conformer)
    - 总参数: ~1.2M
    """
    
    def __init__(self, 
                 input_dim: int = 768,
                 output_dim: int = 256,  # V10: 160→256
                 dropout: float = 0.1,
                 dropout_conv: float = 0.05,   # V10: 0.12→0.05（外部有Mixup+R-Drop）
                 dropout_attn: float = 0.1,    # V10: 0.22→0.1
                 dropout_ffn: float = 0.15):   # V10: 0.32→0.15
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # V10.12: 输入LayerNorm - 对HuBERT输出归一化，稳定多尺度卷积的梯度流
        self.input_ln = nn.LayerNorm(input_dim)
        
        # ============================================================
        # 1. 多尺度卷积层 (5个尺度捕捉不同时间范围的抑郁特征)
        # ============================================================
        # V10: 5个尺度，每个 output_dim//5 通道（向上取整处理余数）
        ch = output_dim // 5
        ch_last = output_dim - ch * 4  # 最后一个分支吸收余数
        
        # 微观特征 (kernel=3): 瞬时变化（语音抖动、微停顿）
        self.conv_micro = nn.Conv1d(
            input_dim, ch, kernel_size=3, padding=1, bias=False
        )
        
        # 短时特征 (kernel=7): 快速变化（停顿、语速突变）
        self.conv_short = nn.Conv1d(
            input_dim, ch, kernel_size=7, padding=3, bias=False
        )
        
        # 中时特征 (kernel=15): 中等变化（音调、能量包络）
        self.conv_medium = nn.Conv1d(
            input_dim, ch, kernel_size=15, padding=7, bias=False
        )
        
        # 长时特征 (kernel=31): 慢速变化（情绪、语调单调性）
        self.conv_long = nn.Conv1d(
            input_dim, ch, kernel_size=31, padding=15, bias=False
        )
        
        # 超长时特征 (kernel=63): 长期模式（整体情绪状态）V10: 41→63
        self.conv_extra_long = nn.Conv1d(
            input_dim, ch_last, kernel_size=63, padding=31, bias=False
        )
        
        # 批归一化 + 激活
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.GELU()
        self.dropout_conv_layer = nn.Dropout(dropout_conv)
        
        # V10.1: 跨尺度注意力（替代SE，显式建模尺度间依赖关系）
        scale_channels = [ch, ch, ch, ch, ch_last]
        self.cross_scale_attention = CrossScaleAttention(scale_channels, attn_dim=64)
        
        # V10.1: 输入残差投影（保留原始HuBERT信息）
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # ============================================================
        # 2. 时序注意力层 (关注抑郁相关的时间段)
        # ============================================================
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout_attn,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(output_dim)
        
        # ============================================================
        # 3. 特征增强层 (V10: 4倍FFN扩展)
        # ============================================================
        self.feature_enhance = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),  # V10: 2x→4x
            nn.GELU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(output_dim * 4, output_dim)
        )
        self.ln2 = nn.LayerNorm(output_dim)
        
        print("抑郁特征提取器初始化完成（V10优化）")
        print(f"   输入维度: {input_dim}")
        print(f"   输出维度: {output_dim}")
        print(f"   卷积kernel: [3, 7, 15, 31, 63] (5尺度)")
        print(f"   跨尺度注意力: attn_dim=64, {len(scale_channels)}尺度交互 (4头)")
        print(f"   输入残差: {input_dim}→{output_dim}")
        print(f"   FFN扩展: 4x ({output_dim}→{output_dim*4}→{output_dim})")
        print(f"   Dropout: conv={dropout_conv}, attn={dropout_attn}, ffn={dropout_ffn}")
        self._print_params()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, seq_len, input_dim) - HuBERT/Wav2Vec2输出
        
        Returns:
            out: (batch, seq_len, output_dim) - 抑郁特定特征
        """
        batch_size, seq_len, _ = x.shape
        
        # V10.12: 输入归一化
        x = self.input_ln(x)
        
        # ============================================================
        # 1. 多尺度卷积特征提取 + SE注意力
        # ============================================================
        x_conv = x.transpose(1, 2)  # (batch, dim, seq_len)
        
        # 5尺度卷积
        feat_micro = self.conv_micro(x_conv)
        feat_short = self.conv_short(x_conv)
        feat_medium = self.conv_medium(x_conv)
        feat_long = self.conv_long(x_conv)
        feat_extra_long = self.conv_extra_long(x_conv)
        
        # V10.1: 跨尺度注意力（在拼接前让各尺度相互交互）
        scale_features = [feat_micro, feat_short, feat_medium, feat_long, feat_extra_long]
        gated_features = self.cross_scale_attention(scale_features)
        
        # 拼接跨尺度增强后的特征
        multi_scale_feat = torch.cat(gated_features, dim=1)  # (batch, output_dim, seq_len)
        
        # 批归一化 + 激活 + Dropout
        multi_scale_feat = self.bn(multi_scale_feat)
        multi_scale_feat = self.activation(multi_scale_feat)
        multi_scale_feat = self.dropout_conv_layer(multi_scale_feat)
        
        # 转回: (batch, output_dim, seq_len) -> (batch, seq_len, output_dim)
        multi_scale_feat = multi_scale_feat.transpose(1, 2)
        
        # V10.1: 输入残差连接（保留原始HuBERT信息）
        residual = self.input_proj(x)  # (batch, seq_len, output_dim)
        multi_scale_feat = multi_scale_feat + residual
        
        # ============================================================
        # 2. 时序注意力 (关注抑郁相关片段)
        # ============================================================
        attn_out, _ = self.attention(
            multi_scale_feat, multi_scale_feat, multi_scale_feat
        )
        x_attn = self.ln1(multi_scale_feat + attn_out)
        
        # ============================================================
        # 3. 特征增强 (学习抑郁特定组合)
        # ============================================================
        enhanced_feat = self.feature_enhance(x_attn)
        out = self.ln2(x_attn + enhanced_feat)
        
        return out
    
    def _print_params(self):
        """打印参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   总参数量: {total_params/1e6:.2f}M")


if __name__ == "__main__":
    print("测试抑郁特征提取器\n")
    
    # 创建模型
    extractor = DepressionFeatureExtractor(
        input_dim=768,
        output_dim=256,
        dropout=0.1
    )
    
    # 测试前向传播
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 768)
    
    print(f"\n输入形状: {x.shape}")
    out = extractor(x)
    print(f"输出形状: {out.shape}")
    
    print("\n测试通过！")
