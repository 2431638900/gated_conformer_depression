"""
基于TRANSCRIPT的受试者语音提取
只提取Participant说话的部分，过滤掉Ellie的语音
"""

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple


class TranscriptBasedExtractor:
    """基于转录文件的受试者语音提取器"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        初始化
        
        Args:
            sample_rate: 采样率
        """
        self.sample_rate = sample_rate
    
    def load_transcript(self, transcript_path: str) -> pd.DataFrame:
        """
        加载转录文件
        
        Args:
            transcript_path: 转录文件路径
        
        Returns:
            DataFrame with columns: start_time, stop_time, speaker, value
        """
        df = pd.read_csv(transcript_path, sep='\t')
        return df
    
    def extract_participant_segments(self, 
                                    audio: np.ndarray,
                                    transcript_df: pd.DataFrame,
                                    padding: float = 0.1) -> List[np.ndarray]:
        """
        提取受试者说话的语音片段
        
        Args:
            audio: 音频数据
            transcript_df: 转录DataFrame
            padding: 每个片段前后的padding（秒）
        
        Returns:
            受试者语音片段列表
        """
        # 只保留Participant的行
        participant_df = transcript_df[transcript_df['speaker'] == 'Participant'].copy()
        
        if len(participant_df) == 0:
            print("  ⚠️  转录文件中没有Participant语音")
            return []
        
        segments = []
        
        for _, row in participant_df.iterrows():
            start_time = float(row['start_time'])
            stop_time = float(row['stop_time'])
            
            # 添加padding
            start_time = max(0, start_time - padding)
            stop_time = min(len(audio) / self.sample_rate, stop_time + padding)
            
            # 转换为样本索引
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(stop_time * self.sample_rate)
            
            # 提取片段
            if start_sample < len(audio) and end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
                if len(segment) > 0:
                    segments.append(segment)
        
        return segments
    
    def process_participant_audio(self,
                                 audio_path: str,
                                 transcript_path: str,
                                 padding: float = 0.1) -> Tuple[np.ndarray, List[np.ndarray], dict]:
        """
        处理受试者音频
        
        Args:
            audio_path: 音频文件路径
            transcript_path: 转录文件路径
            padding: 片段padding（秒）
        
        Returns:
            (完整音频, 受试者语音片段列表, 统计信息)
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 加载转录
        transcript_df = self.load_transcript(transcript_path)
        
        # 提取受试者片段
        participant_segments = self.extract_participant_segments(
            audio, transcript_df, padding
        )
        
        # 统计信息
        total_duration = len(audio) / self.sample_rate
        participant_duration = sum(len(seg) for seg in participant_segments) / self.sample_rate
        
        # 统计说话次数
        participant_count = len(transcript_df[transcript_df['speaker'] == 'Participant'])
        ellie_count = len(transcript_df[transcript_df['speaker'] == 'Ellie'])
        
        stats = {
            'total_duration': total_duration,
            'participant_duration': participant_duration,
            'participant_ratio': participant_duration / total_duration if total_duration > 0 else 0,
            'num_segments': len(participant_segments),
            'participant_turns': participant_count,
            'ellie_turns': ellie_count
        }
        
        return audio, participant_segments, stats


if __name__ == "__main__":
    print("基于转录的语音提取器模块")
    
    # 测试
    extractor = TranscriptBasedExtractor(sample_rate=16000)
    
    # 测试303号受试者
    # Example: set DAIC_WOZ_ROOT and point to one participant folder
    audio_path = "data/daic_woz/303_P/303_AUDIO.wav"
    transcript_path = "data/daic_woz/303_P/303_TRANSCRIPT.csv"
    
    if Path(audio_path).exists() and Path(transcript_path).exists():
        audio, segments, stats = extractor.process_participant_audio(
            audio_path, transcript_path
        )
        
        print(f"\n受试者303统计:")
        print(f"  总时长: {stats['total_duration']:.2f}秒")
        print(f"  受试者语音: {stats['participant_duration']:.2f}秒 ({stats['participant_ratio']*100:.1f}%)")
        print(f"  语音片段数: {stats['num_segments']}")
        print(f"  受试者发言次数: {stats['participant_turns']}")
        print(f"  Ellie发言次数: {stats['ellie_turns']}")
    
    print("\n✅ 模块测试完成！")
