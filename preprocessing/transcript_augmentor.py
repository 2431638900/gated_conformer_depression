"""
基于TRANSCRIPT的数据增强
只使用受试者(Participant)的语音，过滤掉询问者(Ellie)的语音
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import pandas as pd

from .transcript_processor import TranscriptBasedExtractor


class TranscriptBasedAugmentor:
    """基于TRANSCRIPT的数据增强器"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 sentences_per_sample: int = 5,
                 target_duration: float = 30.0,
                 padding: float = 0.1):
        """
        初始化
        
        Args:
            sample_rate: 采样率
            sentences_per_sample: 每个样本包含的句子数
            target_duration: 目标音频长度（秒）
            padding: 每个语音片段的padding（秒）
        """
        self.sample_rate = sample_rate
        self.sentences_per_sample = sentences_per_sample
        self.target_duration = target_duration
        self.target_length = int(sample_rate * target_duration)
        self.padding = padding
        
        # 初始化提取器
        self.extractor = TranscriptBasedExtractor(sample_rate=sample_rate)
    
    def process_single_participant(self, 
                                   audio_path: str,
                                   transcript_path: str,
                                   participant_id: int) -> Dict:
        """
        处理单个受试者的音频
        
        Args:
            audio_path: 音频文件路径
            transcript_path: 转录文件路径
            participant_id: 受试者ID
        
        Returns:
            处理结果字典
        """
        print(f"\n处理受试者 {participant_id}...")
        
        # 使用TRANSCRIPT提取受试者语音
        audio, participant_segments, stats = self.extractor.process_participant_audio(
            audio_path, transcript_path, padding=self.padding
        )
        
        print(f"  原始音频长度: {stats['total_duration']:.2f}秒")
        print(f"  受试者语音: {stats['participant_duration']:.2f}秒 (占{stats['participant_ratio']*100:.1f}%)")
        print(f"  受试者发言: {stats['participant_turns']}次, Ellie发言: {stats['ellie_turns']}次")
        
        # 关键改进：直接使用TRANSCRIPT片段作为句子，不再进行能量检测切分
        # 这样完全复现黄祥胜论文的方法
        sentences = participant_segments  # 每个TRANSCRIPT条目 = 1个句子
        
        print(f"  句子数（TRANSCRIPT条目数）: {len(sentences)}")
        
        if len(sentences) > 0:
            avg_duration = np.mean([len(s)/self.sample_rate for s in sentences])
            print(f"  平均句子长度: {avg_duration:.2f}秒")
        
        return {
            'participant_id': participant_id,
            'sentences': sentences,
            'num_sentences': len(sentences),
            'stats': stats
        }
    
    def generate_augmented_samples(self,
                                   sentences: List[np.ndarray],
                                   participant_id: int,
                                   label: int) -> List[Dict]:
        """
        从句子生成增强样本
        
        Args:
            sentences: 句子列表
            participant_id: 受试者ID
            label: 标签
        
        Returns:
            增强样本列表
        """
        num_sentences = len(sentences)
        
        # 如果句子数为0，跳过
        if num_sentences == 0:
            print("  ⚠️  没有句子，跳过增强")
            return []
        
        augmented_samples = []
        aug_idx = 0
        
        # 策略：每sentences_per_sample句生成一个样本
        start_idx = 0
        while start_idx < num_sentences:
            # 确定当前样本使用的句子范围
            end_idx = min(start_idx + self.sentences_per_sample, num_sentences)
            
            # 获取当前批次的句子索引
            selected_indices = list(range(start_idx, end_idx))
            
            # 拼接句子
            selected_sentences = [sentences[i] for i in selected_indices]
            combined_audio = np.concatenate(selected_sentences)
            
            # 调整到目标长度
            combined_audio = self._adjust_length(combined_audio)
            
            augmented_samples.append({
                'participant_id': participant_id,
                'augmentation_id': aug_idx,
                'audio': combined_audio,
                'label': label,
                'sentence_indices': selected_indices,
                'num_sentences': len(selected_sentences)
            })
            
            aug_idx += 1
            start_idx = end_idx
        
        print(f"  ✅ 生成 {len(augmented_samples)} 个增强样本 (共{num_sentences}句)")
        return augmented_samples
    
    def _adjust_length(self, audio: np.ndarray) -> np.ndarray:
        """
        调整音频长度到目标长度
        
        Args:
            audio: 音频数据
        
        Returns:
            调整后的音频
        """
        current_length = len(audio)
        
        if current_length > self.target_length:
            # 截断
            return audio[:self.target_length]
        elif current_length < self.target_length:
            # 填充
            padding = self.target_length - current_length
            return np.pad(audio, (0, padding), mode='constant')
        else:
            return audio
    
    def process_dataset(self,
                       data_df: pd.DataFrame,
                       data_dir: Path,
                       output_dir: Path) -> pd.DataFrame:
        """
        处理整个数据集
        
        Args:
            data_df: 数据DataFrame
            data_dir: 原始数据目录
            output_dir: 输出目录
        
        Returns:
            增强后的DataFrame
        """
        print(f"\n{'='*60}")
        print("开始数据增强处理（基于TRANSCRIPT）")
        print(f"{'='*60}")
        print(f"原始样本数: {len(data_df)}")
        print(f"每样本句子数: {self.sentences_per_sample}")
        print(f"{'='*60}\n")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        augmented_data = []
        stats = {
            'total_processed': 0,
            'total_augmented': 0,
            'skipped': 0,
            'normal_augmented': 0,
            'depressed_augmented': 0,
            'no_transcript': 0
        }
        
        # 处理每个受试者
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="处理受试者"):
            participant_id = int(row['Participant_ID'])
            # 测试集可能没有标签
            label = int(row['PHQ8_Binary']) if 'PHQ8_Binary' in row else -1
            
            # 构建路径
            audio_path = data_dir / f"{participant_id}_P" / f"{participant_id}_AUDIO.wav"
            transcript_path = data_dir / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"
            
            # 检查文件是否存在
            if not audio_path.exists():
                print(f"  ⚠️  音频文件不存在: {audio_path}")
                stats['skipped'] += 1
                continue
            
            if not transcript_path.exists():
                print(f"  ⚠️  转录文件不存在: {transcript_path}")
                stats['no_transcript'] += 1
                stats['skipped'] += 1
                continue
            
            try:
                # 处理音频
                result = self.process_single_participant(
                    str(audio_path), 
                    str(transcript_path),
                    participant_id
                )
                stats['total_processed'] += 1
                
                # 生成增强样本
                augmented_samples = self.generate_augmented_samples(
                    result['sentences'],
                    participant_id,
                    label
                )
                
                # 保存增强样本
                for sample in augmented_samples:
                    # 生成文件名
                    aug_id = sample['augmentation_id']
                    filename = f"{participant_id}_aug{aug_id}.wav"
                    output_path = output_dir / filename
                    
                    # 保存音频
                    sf.write(output_path, sample['audio'], self.sample_rate)
                    
                    # 记录信息
                    augmented_data.append({
                        'Participant_ID': f"{participant_id}_aug{aug_id}",
                        'Original_ID': participant_id,
                        'Augmentation_ID': aug_id,
                        'PHQ8_Binary': label,
                        'Audio_Path': str(output_path),
                        'Num_Sentences': sample['num_sentences'],
                        'Sentence_Indices': ','.join(map(str, sample['sentence_indices']))
                    })
                    
                    stats['total_augmented'] += 1
                    if label == 0:
                        stats['normal_augmented'] += 1
                    elif label == 1:
                        stats['depressed_augmented'] += 1
            
            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                stats['skipped'] += 1
                continue
        
        # 创建DataFrame
        augmented_df = pd.DataFrame(augmented_data)
        
        # 打印统计
        print(f"\n{'='*60}")
        print("数据增强完成！")
        print(f"{'='*60}")
        print(f"处理成功: {stats['total_processed']}")
        print(f"跳过: {stats['skipped']}")
        print(f"  - 缺少转录文件: {stats['no_transcript']}")
        print(f"生成增强样本: {stats['total_augmented']}")
        print(f"  - 正常样本: {stats['normal_augmented']}")
        print(f"  - 抑郁样本: {stats['depressed_augmented']}")
        print(f"{'='*60}\n")
        
        return augmented_df


if __name__ == "__main__":
    print("基于TRANSCRIPT的数据增强模块")
    print("✅ 模块加载成功！")
