"""
基线模型 - 用于对比实验
包括：LSTM、CNN、LSTM+Attention等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBaseline(nn.Module):
    """LSTM基线模型"""
    
    def __init__(self, 
                 input_dim=16000,
                 hidden_dim=256,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.3):
        """
        Args:
            input_dim: 输入维度（采样率）
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数
            dropout: Dropout率
        """
        super().__init__()
        
        # 特征提取（简单的卷积）
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, length)
        
        Returns:
            logits: (batch, num_classes)
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, length)
        
        # 特征提取
        features = self.feature_extractor(x)  # (batch, 128, seq_len)
        features = features.transpose(1, 2)  # (batch, seq_len, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden_dim*2)
        
        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # 分类
        logits = self.classifier(last_hidden)
        
        return logits


class CNNBaseline(nn.Module):
    """CNN基线模型"""
    
    def __init__(self,
                 input_dim=16000,
                 num_classes=2,
                 dropout=0.3):
        """
        Args:
            input_dim: 输入维度
            num_classes: 分类数
            dropout: Dropout率
        """
        super().__init__()
        
        # 深度卷积网络
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Block 4
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, length)
        
        Returns:
            logits: (batch, num_classes)
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, length)
        
        # 卷积特征提取
        features = self.conv_blocks(x)  # (batch, 512, 1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits


class LSTMAttentionBaseline(nn.Module):
    """LSTM + Attention基线模型"""
    
    def __init__(self,
                 input_dim=16000,
                 hidden_dim=256,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.3):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数
            dropout: Dropout率
        """
        super().__init__()
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, length)
        
        Returns:
            logits: (batch, num_classes)
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, length)
        
        # 特征提取
        features = self.feature_extractor(x)  # (batch, 128, seq_len)
        features = features.transpose(1, 2)  # (batch, seq_len, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden_dim*2)
        
        # Attention
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # 加权求和
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_dim*2)
        
        # 分类
        logits = self.classifier(context)
        
        return logits


class SimpleTransformerBaseline(nn.Module):
    """简单Transformer基线模型"""
    
    def __init__(self,
                 input_dim=16000,
                 d_model=256,
                 num_heads=8,
                 num_layers=4,
                 num_classes=2,
                 dropout=0.1):
        """
        Args:
            input_dim: 输入维度
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            num_classes: 分类数
            dropout: Dropout率
        """
        super().__init__()
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, length)
        
        Returns:
            logits: (batch, num_classes)
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, length)
        
        # 特征提取
        features = self.feature_extractor(x)  # (batch, d_model, seq_len)
        features = features.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Transformer
        transformer_out = self.transformer(features)  # (batch, seq_len, d_model)
        
        # 全局平均池化
        pooled = torch.mean(transformer_out, dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits


def get_baseline_model(model_name, **kwargs):
    """
    获取基线模型
    
    Args:
        model_name: 模型名称 ('lstm', 'cnn', 'lstm_attention', 'transformer')
        **kwargs: 模型参数
    
    Returns:
        模型实例
    """
    models = {
        'lstm': LSTMBaseline,
        'cnn': CNNBaseline,
        'lstm_attention': LSTMAttentionBaseline,
        'transformer': SimpleTransformerBaseline
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    # 测试
    batch_size = 4
    seq_len = 16000 * 30  # 30秒
    
    x = torch.randn(batch_size, seq_len)
    
    print("Testing baseline models...")
    
    for model_name in ['lstm', 'cnn', 'lstm_attention', 'transformer']:
        print(f"\n{model_name.upper()}:")
        model = get_baseline_model(model_name, input_dim=seq_len)
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
