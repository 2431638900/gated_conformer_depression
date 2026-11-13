"""
数据集加载器
"""

import time
import hashlib
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
from pathlib import Path


class DepressionDataset(Dataset):
    """抑郁症检测数据集"""
    
    def __init__(self,
                 data_df: pd.DataFrame,
                 sample_rate: int = 16000,
                 target_length: int = 480000,
                 mode: str = 'pad',
                 cache_audio: bool = False):
        """
        初始化数据集
        
        Args:
            data_df: 数据DataFrame
            sample_rate: 采样率
            target_length: 目标长度
            mode: 长度调整模式 ('pad', 'crop', 'random_crop')
            cache_audio: 是否启用.npy磁盘缓存（实验脚本推荐开启，避免重复librosa解码）
        """
        self.data_df = data_df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.mode = mode
        
        print(f"Dataset初始化:")
        print(f"  样本数: {len(self.data_df)}")
        print(f"  音频长度: {target_length/sample_rate:.0f}秒")
        print(f"  目标采样点数: {target_length}")
        
        # .npy磁盘缓存（不占用内存，加载速度比librosa快300倍）
        self._cache_paths = None
        if cache_audio:
            self._setup_disk_cache()
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Resolve audio path (supports leftover absolute Windows/Linux paths)
        from config.config import Config
        if 'Audio_Path' in row:
            audio_path = str(Config.resolve_audio_path(row['Audio_Path']))
        else:
            participant_id = int(row['Participant_ID'])
            audio_path = str(Config.resolve_audio_path(
                f"{int(row.get('Original_ID', participant_id))}_AUDIO.wav"
            ))
        
        # 加载音频（优先从.npy缓存读取）
        if self._cache_paths is not None:
            audio = np.load(str(self._cache_paths[idx]))
        else:
            audio = self._load_audio(audio_path)
        
        # 调整长度
        audio = self._adjust_length(audio)
        
        # 转换为tensor
        audio_tensor = torch.FloatTensor(audio)
        
        # 获取标签
        label = int(row['PHQ8_Binary'])
        label_tensor = torch.LongTensor([label])
        
        # 获取PHQ-8分数 (新增)
        phq8_score = float(row.get('PHQ8_Score', 0))
        phq8_score_tensor = torch.FloatTensor([phq8_score])
        
        # 转换为5类严重程度 (新增)
        severity = self._phq8_to_severity(phq8_score)
        severity_tensor = torch.LongTensor([severity])
        
        # 获取ID
        if 'Original_ID' in row:
            participant_id = row['Original_ID']
        else:
            participant_id = int(row['Participant_ID'])
        
        return {
            'audio': audio_tensor,
            'label': label_tensor,
            'phq8_score': phq8_score_tensor,
            'severity': severity_tensor,
            'participant_id': participant_id
        }
    
    @staticmethod
    def _phq8_to_severity(score: float) -> int:
        """将PHQ-8分数转换为5类严重程度"""
        if score <= 4:
            return 0  # 无抑郁
        elif score <= 9:
            return 1  # 轻度
        elif score <= 14:
            return 2  # 中度
        elif score <= 19:
            return 3  # 中重度
        else:
            return 4  # 重度
    
    def _get_audio_path(self, idx):
        """获取第idx个样本的音频文件路径"""
        row = self.data_df.iloc[idx]
        from config.config import Config
        if 'Audio_Path' in row:
            return str(Config.resolve_audio_path(row['Audio_Path']))
        participant_id = int(row['Participant_ID'])
        return str(Config.resolve_audio_path(
            f"{int(row.get('Original_ID', participant_id))}_AUDIO.wav"
        ))
    
    def _setup_disk_cache(self):
        """构建.npy磁盘缓存，首次将librosa解码结果保存为.npy，后续直接np.load()"""
        cache_dir = Path(__file__).resolve().parent / 'audio_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        n = len(self.data_df)
        self._cache_paths = []
        
        # 为每个音频文件计算缓存路径
        for idx in range(n):
            audio_path = self._get_audio_path(idx)
            key = hashlib.md5(audio_path.encode()).hexdigest()
            self._cache_paths.append(cache_dir / f"{key}.npy")
        
        # 检查哪些文件需要缓存
        uncached = [i for i, p in enumerate(self._cache_paths) if not p.exists()]
        
        if uncached:
            print(f"  构建.npy磁盘缓存 ({len(uncached)}/{n} 个新文件)...")
            start = time.time()
            for count, idx in enumerate(uncached):
                audio = self._load_audio(self._get_audio_path(idx))
                np.save(str(self._cache_paths[idx]), audio)
                
                if (count + 1) % 500 == 0 or (count + 1) == len(uncached):
                    elapsed = time.time() - start
                    speed = (count + 1) / elapsed
                    eta = (len(uncached) - count - 1) / speed if speed > 0 else 0
                    print(f"    [{count+1}/{len(uncached)}] {elapsed:.0f}s, ETA {eta:.0f}s")
            print(f"  ✅ 缓存构建完成: {time.time()-start:.1f}秒")
        else:
            print(f"  ✅ .npy缓存已就绪 ({n} 个文件)")
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """加载音频"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"加载音频失败 {audio_path}: {e}")
            # 返回静音
            return np.zeros(self.target_length, dtype=np.float32)
    
    def _adjust_length(self, audio: np.ndarray) -> np.ndarray:
        """调整音频长度"""
        current_length = len(audio)
        
        if current_length == self.target_length:
            return audio
        
        if self.mode == 'pad':
            # 填充或截断
            if current_length < self.target_length:
                padding = self.target_length - current_length
                audio = np.pad(audio, (0, padding), mode='constant')
            else:
                audio = audio[:self.target_length]
        
        elif self.mode == 'crop':
            # 从中间截取
            if current_length > self.target_length:
                start = (current_length - self.target_length) // 2
                audio = audio[start:start + self.target_length]
            else:
                # 填充
                padding = self.target_length - current_length
                pad_left = padding // 2
                pad_right = padding - pad_left
                audio = np.pad(audio, (pad_left, pad_right), mode='constant')
        
        elif self.mode == 'random_crop':
            # 随机截取
            if current_length > self.target_length:
                start = np.random.randint(0, current_length - self.target_length)
                audio = audio[start:start + self.target_length]
            else:
                # 填充
                padding = self.target_length - current_length
                audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio


def collate_fn(batch):
    """批次整理函数"""
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    participant_ids = [item['participant_id'] for item in batch]
    
    # 新增: PHQ8分数和严重程度
    phq8_scores = torch.cat([item['phq8_score'] for item in batch])
    severities = torch.cat([item['severity'] for item in batch])
    
    return {
        'audio': audios,
        'label': labels,
        'phq8_score': phq8_scores,
        'severity': severities,
        'participant_id': participant_ids
    }


def load_daic_woz_data(data_dir: Path, split: str = 'train') -> pd.DataFrame:
    """
    加载DAIC-WOZ数据
    
    Args:
        data_dir: 数据目录
        split: 数据划分 ('train', 'dev', 'test')
    
    Returns:
        DataFrame
    """
    print(f"加载DAIC-WOZ {split} 数据集...")
    
    # 读取CSV
    csv_path = data_dir / f'{split}_split_Depression_AVEC2017.csv'
    if not csv_path.exists():
        csv_path = data_dir / f'{split}_split.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到CSV文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 统一列名（测试集是participant_ID，训练集是Participant_ID）
    if 'participant_ID' in df.columns:
        df.rename(columns={'participant_ID': 'Participant_ID'}, inplace=True)
    
    # 统计信息
    print(f"✅ 成功加载 {len(df)} 个样本")
    if len(df) > 0 and 'PHQ8_Binary' in df.columns:
        print(f"   - 正常样本: {(df['PHQ8_Binary'] == 0).sum()}")
        print(f"   - 抑郁样本: {(df['PHQ8_Binary'] == 1).sum()}")
    elif len(df) > 0:
        print(f"   ⚠️  测试集没有标签信息")
    
    return df


if __name__ == "__main__":
    print("数据集模块")
    print("✅ 模块加载成功！")
