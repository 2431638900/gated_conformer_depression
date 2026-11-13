"""
语言自适应适配器 (Language Adapter)

在HuBERT特征提取之后、抑郁特征提取器之前插入，
将中文语音特征映射到类英语特征空间，弥补HuBERT英语预训练与中文语音的域差距。

设计原则：
1. 瓶颈结构 (768→bottleneck→768)：参数量极小 (~200K)
2. 残差连接 + 近零初始化：初始状态等价于恒等映射，不破坏原模型
3. 条件激活：仅在 use_adapter=True 时生效，英语输入完全不受影响

参考：
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019)
- Hou et al., "Exploiting Adapters for Cross-lingual Low-resource Speech Recognition" (2021)
"""

import torch.nn as nn


class LanguageAdapter(nn.Module):
    """
    语言自适应瓶颈适配器
    
    结构: LayerNorm → Linear(down) → GELU → Linear(up) → Dropout + Residual
    
    初始化策略：up projection权重初始化为近零，
    使得 adapter(x) ≈ x（恒等映射），训练开始时不破坏预训练特征。
    """
    
    def __init__(self, input_dim=768, bottleneck_dim=128, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度 (HuBERT输出维度，默认768)
            bottleneck_dim: 瓶颈维度 (控制参数量和容量)
            dropout: Dropout率
        """
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 近零初始化：使adapter初始输出≈0，residual连接保证初始行为=恒等映射
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
        self._print_info(input_dim, bottleneck_dim, dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - HuBERT加权聚合后的特征
        Returns:
            (batch, seq_len, input_dim) - 适配后的特征
        """
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return residual + x
    
    def _print_info(self, input_dim, bottleneck_dim, dropout):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"语言自适应适配器初始化完成")
        print(f"   结构: {input_dim} → {bottleneck_dim} → {input_dim}")
        print(f"   参数量: {total_params/1e3:.1f}K")
        print(f"   Dropout: {dropout}")
        print(f"   初始化: 近零 (恒等映射起始)")
