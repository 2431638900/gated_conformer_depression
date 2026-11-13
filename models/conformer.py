"""
Conformer模型实现
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic Depth - 随机跳过残差路径以减少过拟合
    
    训练时以drop_prob概率随机丢弃整个残差分支输出。
    推理时 (training=False) 自动禁用，行为等价于正常残差连接。
    在DeiT/ConvNeXt/Swin等深层Transformer中广泛验证有效，
    专门解决深层模型val-test泛化差距问题。
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        output = torch.floor(random_tensor + keep_prob)
        return x * output / keep_prob


class ConformerBlock(nn.Module):
    """Conformer块"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 conv_kernel_size: int,
                 feed_forward_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Feed Forward 1
        self.ff1 = FeedForwardModule(d_model, feed_forward_dim, dropout)
        
        # Multi-Head Self-Attention
        self.mhsa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa_dropout = nn.Dropout(dropout)
        
        # Convolution
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Feed Forward 2
        self.ff2 = FeedForwardModule(d_model, feed_forward_dim, dropout)
        
        # Layer Norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Feed Forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Multi-Head Self-Attention
        residual = x
        x = self.mhsa_norm(x)
        x, _ = self.mhsa(x, x, x)
        x = self.mhsa_dropout(x)
        x = residual + x
        
        # Convolution
        x = x + self.conv(x)
        
        # Feed Forward 2
        x = x + 0.5 * self.ff2(x)
        
        # Layer Norm
        x = self.norm(x)
        
        return x


class RelativePositionBias(nn.Module):
    """
    相对位置偏置 - 为注意力机制提供相对位置信息
    相比绝对位置编码，相对位置在语音任务中更有效
    """
    
    def __init__(self, num_heads: int, max_distance: int = 128):
        """
        Args:
            num_heads: 注意力头数
            max_distance: 最大相对距离（超出部分会被截断）
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # 可学习的相对位置嵌入表
        # 范围: [-max_distance, max_distance]，共2*max_distance+1个位置
        self.relative_attention_bias = nn.Embedding(
            2 * max_distance + 1, 
            num_heads
        )
    
    def forward(self, seq_len: int, device: torch.device):
        """
        计算相对位置偏置矩阵
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            bias: (num_heads, seq_len, seq_len) 相对位置偏置
        """
        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device)
        
        # 计算相对位置矩阵
        # relative_positions[i, j] = i - j (查询位置i相对于键位置j的距离)
        relative_positions = positions[:, None] - positions[None, :]
        
        # 截断到[-max_distance, max_distance]范围
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_distance, 
            self.max_distance
        )
        
        # 转换为正索引 [0, 2*max_distance]
        relative_positions = relative_positions + self.max_distance
        
        # 获取相对位置嵌入 (seq_len, seq_len, num_heads)
        bias = self.relative_attention_bias(relative_positions)
        
        # 转换为注意力所需的形状 (num_heads, seq_len, seq_len)
        return bias.permute(2, 0, 1)


class GatedConformerBlock(nn.Module):
    """
    门控Conformer块 - 自适应学习每个模块的重要性
    通过门控机制动态调整FF、MHSA、Conv模块的贡献
    V13B优化: V11架构 + 多尺度卷积，捕获不同时间尺度的抑郁语音特征（短期韵律、中期模式、长期趋势）
    V10.12: 新增DropPath (Stochastic Depth) 随机深度正则化，减少val-test泛化差距
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 conv_kernel_size: int,
                 feed_forward_dim: int,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.0):
        super().__init__()
        
        self.drop_path = DropPath(drop_path_rate)
        
        # Feed Forward 1
        self.ff1 = FeedForwardModule(d_model, feed_forward_dim, dropout)
        
        # Multi-Head Self-Attention
        self.d_model = d_model
        self.num_heads = num_heads
        self.mhsa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa_dropout = nn.Dropout(dropout)
        
        # V13B: 多尺度卷积模块
        self.conv = MultiScaleConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Feed Forward 2
        self.ff2 = FeedForwardModule(d_model, feed_forward_dim, dropout)
        
        # V12A: 门控机制 - 学习每个模块的重要性权重
        # 使用全局平均池化 + 小型MLP生成门控值
        self.gate_ff1 = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # V11: 多头独立门控 - 每个头独立学习重要性
        self.gate_mhsa_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 8),
                nn.SiLU(),
                nn.Linear(d_model // 8, 1),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        self.gate_conv = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        self.gate_ff2 = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Layer Norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model)
        """
        # 计算全局特征用于门控（使用平均池化）
        global_feat = x.mean(dim=1)  # (batch, d_model)
        
        # Feed Forward 1 with gating
        ff1_out = self.ff1(x)
        gate1 = self.gate_ff1(global_feat).unsqueeze(1)  # (batch, 1, 1)
        x = x + self.drop_path(0.5 * gate1 * ff1_out)
        
        # V11: Multi-Head Self-Attention with 多头独立门控
        x_norm = self.mhsa_norm(x)
        mhsa_out, _ = self.mhsa(x_norm, x_norm, x_norm)
        mhsa_out = self.mhsa_dropout(mhsa_out)
        
        # V11: 多头独立门控 - 每个头独立学习重要性
        batch_size, seq_len, _ = mhsa_out.shape
        head_dim = self.d_model // self.num_heads
        
        # 重塑为 (batch, seq_len, num_heads, head_dim)
        mhsa_out_heads = mhsa_out.view(batch_size, seq_len, self.num_heads, head_dim)
        
        # 为每个头应用独立门控
        global_feat = x.mean(dim=1)  # (batch, d_model)
        gated_heads = []
        for i in range(self.num_heads):
            gate = self.gate_mhsa_heads[i](global_feat)  # (batch, 1)
            gate = gate.unsqueeze(1).unsqueeze(3)  # (batch, 1, 1, 1)
            gated_head = gate * mhsa_out_heads[:, :, i:i+1, :]  # (batch, seq_len, 1, head_dim)
            gated_heads.append(gated_head)
        
        # 合并所有头
        gated_mhsa_out = torch.cat(gated_heads, dim=2)  # (batch, seq_len, num_heads, head_dim)
        gated_mhsa_out = gated_mhsa_out.view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        x = x + self.drop_path(gated_mhsa_out)
        
        # Convolution with gating
        conv_out = self.conv(x)
        global_feat = x.mean(dim=1)
        gate3 = self.gate_conv(global_feat).unsqueeze(1)
        x = x + self.drop_path(gate3 * conv_out)
        
        # Feed Forward 2 with gating
        ff2_out = self.ff2(x)
        global_feat = x.mean(dim=1)
        gate4 = self.gate_ff2(global_feat).unsqueeze(1)
        x = x + self.drop_path(0.5 * gate4 * ff2_out)
        
        # Layer Norm
        x = self.norm(x)
        
        return x


class FeedForwardModule(nn.Module):
    """前馈模块"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class MultiScaleConvolutionModule(nn.Module):
    """
    V13B多尺度卷积模块 - 捕获不同时间尺度的抑郁语音特征
    
    三个并行卷积分支:
    - 短期分支 (kernel=7): 捕获快速韵律变化（如停顿、语速）
    - 中期分支 (kernel=15): 捕获中等时长模式（如句子级特征）
    - 长期分支 (kernel=31): 捕获长时趋势（如整体情绪基调）
    
    通过门控机制自适应融合三个尺度的特征
    """
    
    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        
        # 三个尺度的卷积分支（保持原始Conformer的GLU+Depthwise结构）
        # 短期分支 (kernel必须为奇数以保持序列长度不变)
        short_ks = max(3, kernel_size // 2 - 1)
        short_ks = short_ks if short_ks % 2 == 1 else short_ks + 1  # 确保奇数
        self.short_branch = self._make_conv_branch(d_model, short_ks)
        
        # 中期分支 (使用传入的kernel_size)
        self.medium_branch = self._make_conv_branch(d_model, kernel_size)
        
        # 长期分支 (kernel=kernel_size*2+1)
        self.long_branch = self._make_conv_branch(d_model, kernel_size * 2 + 1)
        
        # 门控融合网络 - 学习三个尺度的最优组合
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 3),
            nn.Softmax(dim=-1)  # 三个尺度的权重和为1
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _make_conv_branch(self, d_model: int, kernel_size: int):
        """创建单个卷积分支（遵循Conformer原始设计）"""
        return nn.ModuleDict({
            'pointwise_conv1': nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            'glu': nn.GLU(dim=1),
            'depthwise_conv': nn.Conv1d(
                d_model, d_model,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=d_model
            ),
            'batch_norm': nn.BatchNorm1d(d_model),
            'activation': nn.SiLU(),
            'pointwise_conv2': nn.Conv1d(d_model, d_model, kernel_size=1)
        })
    
    def _forward_branch(self, x, branch):
        """前向传播单个分支"""
        # x: (batch, d_model, seq_len)
        x = branch['pointwise_conv1'](x)
        x = branch['glu'](x)
        x = branch['depthwise_conv'](x)
        x = branch['batch_norm'](x)
        x = branch['activation'](x)
        x = branch['pointwise_conv2'](x)
        return x
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.norm(x)
        
        # Transpose for Conv1d
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # 三个尺度并行处理
        out_short = self._forward_branch(x_t, self.short_branch)
        out_medium = self._forward_branch(x_t, self.medium_branch)
        out_long = self._forward_branch(x_t, self.long_branch)
        
        # Transpose back
        out_short = out_short.transpose(1, 2)  # (batch, seq_len, d_model)
        out_medium = out_medium.transpose(1, 2)
        out_long = out_long.transpose(1, 2)
        
        # 门控融合 - 基于全局特征学习最优尺度组合
        global_feat = x.mean(dim=1)  # (batch, d_model)
        gate_weights = self.fusion_gate(global_feat)  # (batch, 3)
        
        # 加权融合三个尺度
        fused = (gate_weights[:, 0:1, None] * out_short +
                gate_weights[:, 1:2, None] * out_medium +
                gate_weights[:, 2:3, None] * out_long)
        
        return self.dropout(fused)


class ConvolutionModule(nn.Module):
    """卷积模块（保留用于非门控Conformer）"""
    
    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.norm(x)
        
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # Pointwise Conv 1 + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise Conv 2
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        
        return x


class Conformer(nn.Module):
    """Conformer编码器"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 conv_kernel_size: int,
                 feed_forward_dim: int,
                 dropout: float = 0.1,
                 use_gating: bool = False,
                 drop_path_rate: float = 0.0):
        super().__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Conformer块 (GatedConformer支持线性增长的DropPath率: 第0层=0, 末层=drop_path_rate)
        if use_gating:
            dp_rates = [drop_path_rate * i / max(num_layers - 1, 1)
                        for i in range(num_layers)]
            blocks = [
                GatedConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    conv_kernel_size=conv_kernel_size,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout,
                    drop_path_rate=dp_rates[i]
                )
                for i in range(num_layers)
            ]
        else:
            blocks = [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    conv_kernel_size=conv_kernel_size,
                    feed_forward_dim=feed_forward_dim,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        self.conformer_blocks = nn.ModuleList(blocks)
        
        self.d_model = d_model
        self.use_gating = use_gating
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # 输入投影
        x = self.input_projection(x)
        
        # Conformer块
        for block in self.conformer_blocks:
            x = block(x)
        
        return x


if __name__ == "__main__":
    # 测试
    print("Conformer模型模块")
    
    batch_size = 2
    seq_len = 100
    input_dim = 768
    d_model = 256
    
    model = Conformer(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=4,
        num_heads=4,
        conv_kernel_size=31,
        feed_forward_dim=1024,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ Conformer模块测试完成！")
