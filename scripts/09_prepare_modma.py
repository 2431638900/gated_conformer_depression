# -*- coding: utf-8 -*-
"""
MODMA数据集预处理脚本
将MODMA数据集处理为与DAIC-WOZ一致的格式，用于跨数据集泛化验证。

MODMA数据集：
- 52名受试者（23 MDD, 29 HC）
- 每人29个音频任务（朗读、自由发言等）
- 原始采样率44100Hz，需重采样至16000Hz
- 标签来自subjects_information_audio_lanzhou_2015.xlsx

处理流程（与DAIC-WOZ训练数据处理对齐）：
1. 读取标签文件，获取受试者ID和MDD/HC标签
2. 对每个受试者，逐个加载音频任务并重采样至16kHz
3. 使用VAD检测语音活动区间，提取句子级片段（模拟Transcript切分）
   - 基于librosa能量检测找到语音区间
   - 将过短(<1s)的相邻片段合并，过长(>10s)的片段切分
   - 每个片段前后添加0.1s padding（与DAIC-WOZ一致）
4. 每5个句子合并为1个样本（与DAIC-WOZ的SENTENCES_PER_SAMPLE=5一致）
5. 对齐到30秒（不足补零，超长截断）
6. 输出CSV索引文件 + 处理后的音频WAV
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from config.config import Config


# ===================== 配置（与DAIC-WOZ处理参数对齐） =====================
# Override with --modma-root or the MODMA_ROOT environment variable.
MODMA_ROOT = Config.MODMA_ROOT
LABEL_FILE = MODMA_ROOT / "subjects_information_audio_lanzhou_2015.xlsx"
OUTPUT_DIR = Config.PROJECT_ROOT / "data" / "modma_processed"
TARGET_SR = Config.SAMPLE_RATE            # 16000 Hz
TARGET_LENGTH = Config.TARGET_LENGTH      # 480000 (30s)
SENTENCES_PER_SAMPLE = Config.SENTENCES_PER_SAMPLE  # 5 (每5句合并为1个样本)
PADDING_SEC = 0.1                         # 每个片段前后0.1s padding（与transcript_augmentor一致）
MIN_SENTENCE_SEC = 1.0                    # 最小句子长度（秒）
MAX_SENTENCE_SEC = 10.0                   # 最大句子长度（秒）
TOP_DB = 25                               # 能量检测阈值


def load_labels():
    """读取MODMA标签文件"""
    df = pd.read_excel(str(LABEL_FILE))
    # 列：subject id, type, age, gender, education, PHQ-9, ...
    df = df[['subject id', 'type', 'PHQ-9']].dropna(subset=['subject id', 'type'])
    df.columns = ['subject_id', 'type', 'phq9']
    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    df['label'] = (df['type'] == 'MDD').astype(int)  # MDD=1, HC=0
    print(f"标签文件加载完成: {len(df)} 名受试者 "
          f"(MDD={sum(df['label']==1)}, HC={sum(df['label']==0)})")
    return df


def extract_sentences_vad(audio, sr=16000, top_db=TOP_DB,
                          min_sec=MIN_SENTENCE_SEC, max_sec=MAX_SENTENCE_SEC,
                          padding_sec=PADDING_SEC):
    """
    基于VAD的句子级片段提取（模拟DAIC-WOZ的Transcript切分）

    流程：
    1. librosa.effects.split 检测语音区间
    2. 过短(<min_sec)的相邻片段合并（模拟连续说话）
    3. 过长(>max_sec)的片段切分
    4. 每个片段添加前后padding

    Returns:
        sentences: list of np.ndarray，每个是一个"句子"
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return []

    # 合并过短的相邻片段
    merged = []
    current_start, current_end = intervals[0]
    for i in range(1, len(intervals)):
        gap = (intervals[i][0] - current_end) / sr
        segment_dur = (current_end - current_start) / sr

        # 如果当前段过短 且 间隔<0.5s，合并
        if segment_dur < min_sec and gap < 0.5:
            current_end = intervals[i][1]
        else:
            merged.append((current_start, current_end))
            current_start, current_end = intervals[i]
    merged.append((current_start, current_end))

    # 提取句子，处理过长/过短
    sentences = []
    padding_samples = int(padding_sec * sr)

    for start, end in merged:
        duration = (end - start) / sr
        if duration < 0.3:
            continue  # 太短的噪声片段丢弃

        # 添加padding（与Transcript处理一致）
        padded_start = max(0, start - padding_samples)
        padded_end = min(len(audio), end + padding_samples)

        segment = audio[padded_start:padded_end]

        # 如果过长，按max_sec切分
        if duration > max_sec:
            max_samples = int(max_sec * sr)
            for sub_start in range(0, len(segment), max_samples):
                sub = segment[sub_start:sub_start + max_samples]
                if len(sub) / sr >= min_sec:
                    sentences.append(sub)
        else:
            sentences.append(segment)

    return sentences


def adjust_length(audio, target_length=TARGET_LENGTH):
    """调整音频长度到目标长度（与DepressionDataset._adjust_length一致）"""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), mode='constant')
    return audio


def process_subject(subject_id, subject_dir, label, phq9):
    """
    处理单个受试者（与DAIC-WOZ处理流程对齐）：
    1. 逐个加载音频任务
    2. 对每个音频提取句子级片段
    3. 汇总所有句子
    4. 每SENTENCES_PER_SAMPLE(5)个句子合并为1个样本
    5. 对齐到30秒
    """
    wav_files = sorted(subject_dir.glob("*.wav"))
    if len(wav_files) == 0:
        print(f"  {subject_id}: 无WAV文件，跳过")
        return []

    # 从所有音频任务中提取句子
    all_sentences = []
    total_dur = 0
    for wf in wav_files:
        try:
            y, _ = librosa.load(str(wf), sr=TARGET_SR)
            total_dur += len(y) / TARGET_SR
            sentences = extract_sentences_vad(y, sr=TARGET_SR)
            all_sentences.extend(sentences)
        except Exception as e:
            print(f"  {subject_id}: 加载 {wf.name} 失败: {e}")
            continue

    if len(all_sentences) == 0:
        print(f"  {subject_id}: 无有效句子，跳过")
        return []

    speech_dur = sum(len(s) / TARGET_SR for s in all_sentences)

    # 每SENTENCES_PER_SAMPLE句合并为1个样本（与DAIC-WOZ一致）
    samples = []
    sample_idx = 0
    start = 0
    while start < len(all_sentences):
        end_idx = min(start + SENTENCES_PER_SAMPLE, len(all_sentences))
        sentence_group = all_sentences[start:end_idx]

        # 拼接句子
        combined = np.concatenate(sentence_group)

        # 对齐到目标长度
        combined = adjust_length(combined, TARGET_LENGTH)

        # 保存
        out_name = f"{subject_id}_aug{sample_idx}.wav"
        out_path = OUTPUT_DIR / out_name
        sf.write(str(out_path), combined, TARGET_SR)

        samples.append({
            'Participant_ID': f"{subject_id}_aug{sample_idx}",
            'Original_ID': subject_id,
            'Augmentation_ID': sample_idx,
            'PHQ8_Binary': label,
            'PHQ8_Score': phq9 if not pd.isna(phq9) else 0,
            'Audio_Path': str(out_path),
            'Num_Sentences': len(sentence_group),
            'Sentence_Indices': ','.join(map(str, range(start, end_idx))),
        })
        sample_idx += 1
        start = end_idx

    print(f"  {subject_id}: 原始{total_dur:.1f}s → 语音{speech_dur:.1f}s → "
          f"{len(all_sentences)}句 → {sample_idx}样本 "
          f"({'MDD' if label == 1 else 'HC'}, PHQ-9={phq9})")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare MODMA segments")
    parser.add_argument("--modma-root", type=str, default=None,
                        help="Path to audio_lanzhou_2015 (or set MODMA_ROOT)")
    args = parser.parse_args()
    global MODMA_ROOT, LABEL_FILE, OUTPUT_DIR
    if args.modma_root:
        MODMA_ROOT = Path(args.modma_root).expanduser()
        LABEL_FILE = MODMA_ROOT / "subjects_information_audio_lanzhou_2015.xlsx"

    print("=" * 60)
    print("MODMA数据集预处理（与DAIC-WOZ处理流程对齐）")
    print("=" * 60)
    print(f"  采样率: {TARGET_SR} Hz")
    print(f"  目标长度: {TARGET_LENGTH} ({TARGET_LENGTH/TARGET_SR:.0f}s)")
    print(f"  每样本句子数: {SENTENCES_PER_SAMPLE}")
    print(f"  句子padding: {PADDING_SEC}s")
    print(f"  句子长度范围: [{MIN_SENTENCE_SEC}s, {MAX_SENTENCE_SEC}s]")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载标签
    labels_df = load_labels()

    # 2. 处理每个受试者
    all_samples = []
    missing_dirs = []
    empty_dirs = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="处理受试者"):
        sid = row['subject_id']
        # MODMA目录名是8位数字（如02010001），Excel中是7位（如2010001），需要补0
        sid_padded = f"0{sid}" if len(sid) == 7 else sid
        subject_dir = MODMA_ROOT / sid_padded
        
        if not subject_dir.exists():
            missing_dirs.append(sid_padded)
            continue
        
        # 检查目录是否为空
        wav_files = list(subject_dir.glob("*.wav"))
        if len(wav_files) == 0:
            empty_dirs.append(sid_padded)
            continue

        samples = process_subject(sid, subject_dir, row['label'], row['phq9'])
        all_samples.extend(samples)
    
    # 报告问题
    if missing_dirs:
        print(f"\n⚠️ 警告: {len(missing_dirs)} 个受试者目录不存在")
    if empty_dirs:
        print(f"\n⚠️ 警告: {len(empty_dirs)} 个受试者目录为空（无WAV文件）")
        print(f"   示例: {empty_dirs[:3]}")
        print(f"   请检查音频文件是否已解压到正确位置")

    # 3. 生成索引CSV（列名与DAIC-WOZ增强数据一致）
    if len(all_samples) == 0:
        print(f"\n{'=' * 60}")
        print("❌ 预处理失败: 未生成任何样本")
        print(f"{'=' * 60}")
        print("\n可能的原因:")
        print("1. 所有受试者目录都为空（无WAV文件）")
        print("2. 音频文件未解压到正确位置")
        print(f"\n请检查目录: {MODMA_ROOT}")
        print("每个受试者目录（如 02010001/）应包含29个WAV文件")
        print(f"{'=' * 60}")
        return
    
    df = pd.DataFrame(all_samples)
    csv_path = OUTPUT_DIR / "modma_samples.csv"
    df.to_csv(str(csv_path), index=False, encoding='utf-8')

    print(f"\n{'=' * 60}")
    print("✅ 预处理完成!")
    print(f"总样本数: {len(df)}")
    print(f"MDD样本: {sum(df['PHQ8_Binary'] == 1)}")
    print(f"HC样本: {sum(df['PHQ8_Binary'] == 0)}")
    print(f"受试者数: {df['Original_ID'].nunique()}")
    print(f"索引文件: {csv_path}")
    print(f"音频目录: {OUTPUT_DIR}")
    print(f"\nCSV列名与DAIC-WOZ增强数据一致:")
    print(f"  {list(df.columns)}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
