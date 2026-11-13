"""
WARNING: this script is kept only for historical reference.

It performs a random 70/15/15 participant-level split (132/28/29) and is
NOT the official AVEC 2017 split used in the paper (107/35/47,
4,589 / 983 / 984 segments). Running it will overwrite official-split
manifests and will not reproduce the reported results.

Use scripts/prepare_official_split.py instead.
"""
import sys as _sys
if "--i-understand-this-is-not-the-official-split" not in _sys.argv:
    raise SystemExit(
        "Refusing to run the legacy random resplit.\n"
        "This is not the official AVEC split used in the paper.\n"
        "Use: python scripts/prepare_official_split.py --daic-woz-root <path>\n"
        "To run this legacy script anyway, pass "
        "--i-understand-this-is-not-the-official-split"
    )
_sys.argv = [a for a in _sys.argv if a != "--i-understand-this-is-not-the-official-split"]

"""
重新划分数据集 - 按参与者ID分层抽样，防止数据泄漏
同一参与者的所有样本必须在同一个集合中，避免模型"记住"特定说话者的特征
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import json
import re

def extract_participant_id(value):
    """
    从 Participant_ID 列（如 '303_aug0'）或 Audio_Path 中提取原始参与者编号
    同一参与者的所有增强样本必须在同一个集合中
    """
    value_str = str(value)
    # 格式: 303_aug0 → 提取 303
    match = re.match(r'^(\d{3,4})_', value_str)
    if match:
        return int(match.group(1))
    # 如果是纯数字
    if value_str.isdigit():
        return int(value_str)
    # 从路径提取: .../<pid>_aug0.wav
    match = re.search(r'[/\\](\d{3,4})_', value_str)
    if match:
        return int(match.group(1))
    return None

def resplit_dataset():
    """重新划分数据集 - 按参与者ID划分，防止数据泄漏"""
    
    print("=" * 60)
    print("重新划分数据集 - 按参与者ID分层抽样（防止数据泄漏）")
    print("=" * 60)
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data' / 'augmented'
    
    print(f"\n📂 项目目录: {project_dir}")
    print(f"📂 数据目录: {data_dir}")
    
    # 1. 读取所有数据
    print("\n📂 读取所有数据...")
    train_df = pd.read_csv(data_dir / 'train' / 'augmented_train.csv')
    dev_df = pd.read_csv(data_dir / 'dev' / 'augmented_dev.csv')
    test_df = pd.read_csv(data_dir / 'test' / 'augmented_test.csv')
    
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(dev_df)} 样本")
    print(f"  测试集: {len(test_df)} 样本")
    print(f"  总计: {len(train_df) + len(dev_df) + len(test_df)} 样本")
    
    # 2. 合并所有数据
    print("\n🔄 合并所有数据...")
    all_data = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    # 提取原始参与者编号（用于按参与者分组）
    # CSV 中 Participant_ID 格式为 '303_aug0'，需要提取原始编号 303
    if 'Participant_ID' in all_data.columns:
        all_data['Original_PID'] = all_data['Participant_ID'].apply(extract_participant_id)
        print(f"  从 Participant_ID 列提取原始参与者编号")
    else:
        all_data['Original_PID'] = all_data['Audio_Path'].apply(extract_participant_id)
        print(f"  从 Audio_Path 提取原始参与者编号")
    
    # 检查无法提取ID的样本
    missing_ids = all_data['Original_PID'].isna().sum()
    if missing_ids > 0:
        print(f"  ⚠️  {missing_ids} 个样本无法提取参与者ID")
        all_data = all_data.dropna(subset=['Original_PID'])
        print(f"  已移除无法提取ID的样本，剩余: {len(all_data)}")
    
    all_data['Original_PID'] = all_data['Original_PID'].astype(int)
    
    # 统计参与者信息
    n_participants = all_data['Original_PID'].nunique()
    print(f"\n📊 参与者统计:")
    print(f"  原始参与者数（去增强后）: {n_participants}")
    print(f"  总样本数（含增强）: {len(all_data)}")
    print(f"  平均每参与者样本数: {len(all_data)/n_participants:.1f}")
    
    # 统计原始标签分布
    label_counts = all_data['PHQ8_Binary'].value_counts()
    print(f"\n📊 整体标签分布:")
    print(f"  正常 (0): {label_counts[0]} ({label_counts[0]/len(all_data)*100:.1f}%)")
    print(f"  抑郁 (1): {label_counts[1]} ({label_counts[1]/len(all_data)*100:.1f}%)")
    
    # 3. 按原始参与者ID划分（关键：防止数据泄漏）
    print("\n✂️  按参与者ID分层划分 (70% train, 15% val, 15% test)...")
    print("  ⚠️  确保同一参与者的所有样本（含增强）在同一个集合中")
    
    # 构建参与者级别的信息表（按原始参与者ID分组）
    participant_info = all_data.groupby('Original_PID').agg(
        label=('PHQ8_Binary', 'first'),  # 每个参与者的标签
        n_samples=('PHQ8_Binary', 'count')  # 每个参与者的样本数
    ).reset_index()
    
    participant_ids = participant_info['Original_PID'].values
    participant_labels = participant_info['label'].values
    
    print(f"  参与者数: {len(participant_ids)}")
    print(f"  正常参与者: {np.sum(participant_labels==0)}")
    print(f"  抑郁参与者: {np.sum(participant_labels==1)}")
    
    # 第一次划分: 70% train, 30% temp（按参与者ID划分）
    train_pids, temp_pids, train_plabels, temp_plabels = train_test_split(
        participant_ids, participant_labels,
        test_size=0.3,
        stratify=participant_labels,
        random_state=42
    )
    
    # 第二次划分: 将30%平分为15% val和15% test
    val_pids, test_pids, val_plabels, test_plabels = train_test_split(
        temp_pids, temp_plabels,
        test_size=0.5,
        stratify=temp_plabels,
        random_state=42
    )
    
    # 根据原始参与者ID筛选样本
    train_data = all_data[all_data['Original_PID'].isin(train_pids)].copy()
    val_data = all_data[all_data['Original_PID'].isin(val_pids)].copy()
    test_data = all_data[all_data['Original_PID'].isin(test_pids)].copy()
    
    # 验证无泄漏
    train_set = set(train_pids)
    val_set = set(val_pids)
    test_set = set(test_pids)
    assert len(train_set & val_set) == 0, "数据泄漏: 训练集和验证集有共同参与者!"
    assert len(train_set & test_set) == 0, "数据泄漏: 训练集和测试集有共同参与者!"
    assert len(val_set & test_set) == 0, "数据泄漏: 验证集和测试集有共同参与者!"
    print("\n✅ 数据泄漏检查通过: 三个集合无共同参与者")
    
    print(f"\n📋 划分结果 (按参与者ID):")
    print(f"  训练集: {len(train_pids)} 参与者, {len(train_data)} 样本 (抑郁: {np.sum(train_data['PHQ8_Binary']==1)}, {np.sum(train_data['PHQ8_Binary']==1)/len(train_data)*100:.1f}%)")
    print(f"  验证集: {len(val_pids)} 参与者, {len(val_data)} 样本 (抑郁: {np.sum(val_data['PHQ8_Binary']==1)}, {np.sum(val_data['PHQ8_Binary']==1)/len(val_data)*100:.1f}%)")
    print(f"  测试集: {len(test_pids)} 参与者, {len(test_data)} 样本 (抑郁: {np.sum(test_data['PHQ8_Binary']==1)}, {np.sum(test_data['PHQ8_Binary']==1)/len(test_data)*100:.1f}%)")
    
    # 6. 创建新的目录结构
    print("\n📁 创建新的目录结构...")
    new_data_dir = project_dir / 'data' / 'augmented_stratified'
    new_data_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        split_dir = new_data_dir / split
        split_dir.mkdir(exist_ok=True)
    
    # 7. 移动音频文件并更新CSV
    print("\n🔄 移动音频文件...")
    
    def move_files_and_update_csv(df, split_name):
        """移动音频文件并更新CSV中的路径"""
        new_rows = []
        target_dir = new_data_dir / split_name
        
        for idx, row in df.iterrows():
            old_path = Path(row['Audio_Path'])
            if old_path.exists():
                # 新文件路径
                new_path = target_dir / old_path.name
                
                # 复制文件
                shutil.copy2(old_path, new_path)
                
                # 更新行数据
                new_row = row.copy()
                new_row['Audio_Path'] = str(new_path)
                new_rows.append(new_row)
            else:
                print(f"  ⚠️  文件不存在: {old_path}")
        
        result_df = pd.DataFrame(new_rows)
        # 移除临时列
        if 'Original_PID' in result_df.columns:
            result_df = result_df.drop(columns=['Original_PID'])
        return result_df
    
    print("  处理训练集...")
    train_data_new = move_files_and_update_csv(train_data, 'train')
    print(f"    ✅ 已复制 {len(train_data_new)} 个文件")
    
    print("  处理验证集...")
    val_data_new = move_files_and_update_csv(val_data, 'val')
    print(f"    ✅ 已复制 {len(val_data_new)} 个文件")
    
    print("  处理测试集...")
    test_data_new = move_files_and_update_csv(test_data, 'test')
    print(f"    ✅ 已复制 {len(test_data_new)} 个文件")
    
    # 8. 保存新的CSV文件
    print("\n💾 保存新的CSV文件...")
    train_data_new.to_csv(new_data_dir / 'train' / 'augmented_train.csv', index=False)
    val_data_new.to_csv(new_data_dir / 'val' / 'augmented_val.csv', index=False)
    test_data_new.to_csv(new_data_dir / 'test' / 'augmented_test.csv', index=False)
    
    print("  ✅ CSV文件已保存")
    
    # 9. 保存划分信息
    print("\n📝 保存划分信息...")
    split_info = {
        'split_method': 'participant_level_stratified',
        'random_state': 42,
        'train_participants': len(train_pids),
        'val_participants': len(val_pids),
        'test_participants': len(test_pids),
        'train_participant_ids': [int(x) for x in train_pids],
        'val_participant_ids': [int(x) for x in val_pids],
        'test_participant_ids': [int(x) for x in test_pids],
        'train_samples': len(train_data_new),
        'val_samples': len(val_data_new),
        'test_samples': len(test_data_new),
        'train_depression_ratio': float(np.sum(train_data_new['PHQ8_Binary']==1) / len(train_data_new)),
        'val_depression_ratio': float(np.sum(val_data_new['PHQ8_Binary']==1) / len(val_data_new)),
        'test_depression_ratio': float(np.sum(test_data_new['PHQ8_Binary']==1) / len(test_data_new)),
        'train_depression_count': int(np.sum(train_data_new['PHQ8_Binary']==1)),
        'val_depression_count': int(np.sum(val_data_new['PHQ8_Binary']==1)),
        'test_depression_count': int(np.sum(test_data_new['PHQ8_Binary']==1))
    }
    
    with open(new_data_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print("  ✅ 划分信息已保存")
    
    # 10. 总结
    print("\n" + "=" * 60)
    print("✅ 数据集重新划分完成！")
    print("=" * 60)
    print(f"\n新数据集位置: {new_data_dir}")
    print(f"\n最终统计:")
    print(f"  训练集: {len(train_data_new)} 样本 ({split_info['train_depression_ratio']*100:.1f}% 抑郁)")
    print(f"  验证集: {len(val_data_new)} 样本 ({split_info['val_depression_ratio']*100:.1f}% 抑郁)")
    print(f"  测试集: {len(test_data_new)} 样本 ({split_info['test_depression_ratio']*100:.1f}% 抑郁)")
    print(f"\n✨ 三个集合的抑郁比例现在应该非常接近！")
    
    return new_data_dir

if __name__ == '__main__':
    resplit_dataset()
