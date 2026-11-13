"""
完整的抑郁症检测模型
预训练模型(HuBERT/Wav2Vec2) + Conformer + 分类器
"""

import warnings
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers import HubertModel, HubertConfig

from .conformer import Conformer
from .depression_feature_extractor import DepressionFeatureExtractor
from .language_adapter import LanguageAdapter


class AttentionPooling(nn.Module):
    """
    注意力池化层 (V7新增)
    学习关注重要的时间段，保留关键的抑郁特征
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            pooled: (batch, d_model)
        """
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, d_model)
        return pooled


class StatisticsAttentionPooling(nn.Module):
    """
    统计注意力池化层 (V10.3新增)
    注意力加权均值 + 注意力加权标准差，信息量翻倍
    标准差捕捉语音特征的时序波动性（抑郁语音的关键特征之一）
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.output_dim = d_model * 2
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            pooled: (batch, 2 * d_model) - [mean; std]
        """
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 注意力加权均值
        mean = torch.sum(x * attn_weights, dim=1)  # (batch, d_model)
        
        # 注意力加权标准差
        diff = x - mean.unsqueeze(1)  # (batch, seq_len, d_model)
        var = torch.sum(attn_weights * diff ** 2, dim=1)  # (batch, d_model)
        std = torch.sqrt(var + 1e-8)  # (batch, d_model)
        
        return torch.cat([mean, std], dim=1)  # (batch, 2 * d_model)


class DepressionDetectionModel(nn.Module):
    """抑郁症检测模型"""
    
    def __init__(self,
                 pretrain_model_type: str = "hubert",
                 pretrain_model_name: str = "facebook/hubert-base-ls960",
                 pretrain_cache_dir: str = None,
                 pretrain_freeze: bool = True,
                 pretrain_layers_to_finetune: int = 0,
                 use_depression_extractor: bool = True,
                 d_model: int = 256,
                 num_conformer_layers: int = 4,
                 num_attention_heads: int = 4,
                 conv_kernel_size: int = 31,
                 feed_forward_dim: int = 1024,
                 dropout: float = 0.1,
                 num_classes: int = 2,
                 use_gating: bool = False,
                 drop_path_rate: float = 0.0,
                 adapter_config: dict = None):
        super().__init__()
        
        self.pretrain_model_type = pretrain_model_type
        
        # 加载预训练特征提取器（支持 Wav2Vec2 和 HuBERT）
        if pretrain_model_type == "hubert":
            print(f"加载HuBERT模型: {pretrain_model_name}")
            try:
                hubert_config = HubertConfig.from_pretrained(
                    pretrain_model_name,
                    cache_dir=pretrain_cache_dir,
                    local_files_only=True
                )
                # 抑制HuBERT权重参数化警告（新版PyTorch weight_norm API变更导致，不影响功能）
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*were not used.*")
                    warnings.filterwarnings("ignore", message=".*newly initialized.*")
                    warnings.filterwarnings("ignore", message=".*TRAIN this model.*")
                    self.feature_encoder = HubertModel.from_pretrained(
                        pretrain_model_name,
                        config=hubert_config,
                        cache_dir=pretrain_cache_dir,
                        local_files_only=True
                    )
                print("✅ HuBERT模型加载成功（本地）")
            except Exception as e:
                print(f"⚠️ 本地加载失败: {e}，尝试在线下载...")
                hubert_config = HubertConfig.from_pretrained(
                    "facebook/hubert-base-ls960",
                    cache_dir=pretrain_cache_dir
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*were not used.*")
                    warnings.filterwarnings("ignore", message=".*newly initialized.*")
                    warnings.filterwarnings("ignore", message=".*TRAIN this model.*")
                    self.feature_encoder = HubertModel.from_pretrained(
                        "facebook/hubert-base-ls960",
                        config=hubert_config,
                        cache_dir=pretrain_cache_dir
                    )
                print("✅ HuBERT模型加载成功（在线下载）")
        else:
            print(f"加载Wav2Vec 2.0模型: {pretrain_model_name}")
            wav2vec_config = Wav2Vec2Config.from_pretrained(
                pretrain_model_name,
                cache_dir=pretrain_cache_dir,
                local_files_only=True
            )
            self.feature_encoder = Wav2Vec2Model.from_pretrained(
                pretrain_model_name,
                config=wav2vec_config,
                cache_dir=pretrain_cache_dir,
                local_files_only=True
            )
            print("✅ Wav2Vec 2.0模型加载成功")
        
        # 冻结预训练模型参数（支持部分微调）
        if pretrain_freeze:
            for param in self.feature_encoder.parameters():
                param.requires_grad = False
            print(f"✅ {pretrain_model_type.upper()} 已冻结")
        elif pretrain_layers_to_finetune > 0:
            total_layers = len(self.feature_encoder.encoder.layers)
            for param in self.feature_encoder.parameters():
                param.requires_grad = False
            layers_to_unfreeze = min(pretrain_layers_to_finetune, total_layers)
            for i in range(total_layers - layers_to_unfreeze, total_layers):
                for param in self.feature_encoder.encoder.layers[i].parameters():
                    param.requires_grad = True
            print(f"✅ {pretrain_model_type.upper()} 部分微调: 解冻最后{layers_to_unfreeze}/{total_layers}层")
        else:
            print(f"✅ {pretrain_model_type.upper()} 全部解冻")
        
        wav2vec_output_dim = self.feature_encoder.config.hidden_size
        
        # 语言自适应适配器 (可选，用于跨语言迁移)
        self.language_adapter = None
        if adapter_config is not None:
            self.language_adapter = LanguageAdapter(
                input_dim=wav2vec_output_dim,
                bottleneck_dim=adapter_config.get('bottleneck_dim', 128),
                dropout=adapter_config.get('dropout', 0.1),
            )
        
        # V10.3: HuBERT加权层聚合 - 可学习权重聚合全12层+CNN输出
        num_hidden_layers = self.feature_encoder.config.num_hidden_layers  # 12 for base
        self.num_layer_weights = num_hidden_layers + 1  # +1 for CNN feature extractor output
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layer_weights))  # 初始均匀权重
        print(f"V10.3: HuBERT加权层聚合 - {self.num_layer_weights}层可学习权重")
        
        # 抑郁特征提取器 (可选)
        self.use_depression_extractor = use_depression_extractor
        if use_depression_extractor:
            self.depression_extractor = DepressionFeatureExtractor(
                input_dim=wav2vec_output_dim,
                output_dim=d_model,
                dropout=dropout
            )
            conformer_input_dim = d_model
        else:
            self.depression_extractor = None
            conformer_input_dim = wav2vec_output_dim
        
        # Conformer编码器
        self.conformer = Conformer(
            input_dim=conformer_input_dim,
            d_model=d_model,
            num_layers=num_conformer_layers,
            num_heads=num_attention_heads,
            conv_kernel_size=conv_kernel_size,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            use_gating=use_gating,
            drop_path_rate=drop_path_rate
        )
        
        self.use_gating = use_gating
        
        # V10.3: 统计注意力池化 (均值+标准差)
        self.pooling = StatisticsAttentionPooling(d_model)
        pool_output_dim = d_model * 2  # 均值+标准差
        
        # 分类器 - V10.3: 输入维度从 d_model 变为 2*d_model
        if num_classes == 4:
            # 四分类：4层渐进式降维
            self.classifier = nn.Sequential(
                nn.Linear(pool_output_dim, d_model),  # V10.3: 2*d_model → d_model
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.15),
                
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.20),
                
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.GELU(),
                nn.Dropout(0.15),
                
                nn.Linear(d_model // 4, num_classes)
            )
        else:
            # 二分类：V10.3 - 3层渐进式降维 + GELU
            self.classifier = nn.Sequential(
                nn.Linear(pool_output_dim, d_model),  # V10.3: 2*d_model → d_model
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.25),
                
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.30),
                
                nn.Linear(d_model // 2, num_classes)
            )
        
        # 缓存是否有可训练的预训练参数（避免forward中反复遍历）
        self._has_trainable_pretrain = any(
            p.requires_grad for p in self.feature_encoder.parameters()
        )
        
        print("✅ 抑郁症检测模型初始化完成\n")
        self._print_model_info()
    
    def forward(self, audio, use_adapter=False):
        """
        前向传播
        
        Args:
            audio: (batch_size, audio_length) - 原始音频输入
            use_adapter: 是否使用语言适配器（中文输入时设为True）
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # V10.3: 预训练模型特征提取 + 加权层聚合
        with torch.set_grad_enabled(self.training and self._has_trainable_pretrain):
            encoder_output = self.feature_encoder(audio, output_hidden_states=True)
            hidden_states = encoder_output.hidden_states  # tuple of (num_layers+1) tensors
        
        # V10.3: 可学习加权聚合所有层 (替代仅用last_hidden_state)
        stacked = torch.stack(hidden_states, dim=0)  # (num_layers+1, batch, seq_len, 768)
        weights = torch.softmax(self.layer_weights, dim=0)  # (num_layers+1,)
        features = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (batch, seq_len, 768)
        
        # 语言自适应适配器 (仅中文输入时激活)
        if use_adapter and self.language_adapter is not None:
            features = self.language_adapter(features)  # (batch, seq_len, 768)
        
        # 抑郁特征提取 (如果启用)
        if self.use_depression_extractor:
            features = self.depression_extractor(features)  # (batch, seq_len, d_model)
        
        # Conformer编码
        conformer_output = self.conformer(features)  # (batch, seq_len, d_model)
        
        # V10.3: 统计注意力池化 (均值+标准差)
        pooled = self.pooling(conformer_output)  # (batch, 2*d_model)
        
        # 分类
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print("="*60)
        print("模型信息")
        print("="*60)
        print(f"总参数量: {total_params/1e6:.2f}M")
        print(f"可训练参数: {trainable_params/1e6:.2f}M")
        print(f"冻结参数: {frozen_params/1e6:.2f}M")
        print("="*60 + "\n")


if __name__ == "__main__":
    print("抑郁症检测模型模块")
    
    # 测试
    model = DepressionDetectionModel(
        pretrain_model_type="hubert",
        pretrain_model_name="facebook/hubert-base-ls960",
        pretrain_cache_dir=None,
        d_model=256,
        num_conformer_layers=6,
        num_attention_heads=8,
        conv_kernel_size=31,
        feed_forward_dim=1024,
        dropout=0.35,
        num_classes=2
    )
    
    # 测试输入
    batch_size = 2
    audio_length = 16000 * 30  # 30秒
    audio = torch.randn(batch_size, audio_length)
    
    # 前向传播
    with torch.no_grad():
        logits = model(audio)
    
    print(f"输入形状: {audio.shape}")
    print(f"输出形状: {logits.shape}")
    print("\n✅ 模型测试完成！")
