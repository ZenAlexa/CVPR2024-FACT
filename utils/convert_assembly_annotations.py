import os
import json
import numpy as np
from pathlib import Path

def load_mapping(mapping_file):
    """加载标签映射"""
    label2idx = {}
    idx2label = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            # 跳过注释行
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                idx, label = parts
                idx = int(idx)
                label2idx[label] = idx
                idx2label[idx] = label
    
    return label2idx, idx2label

def convert_annotations(annotation_file, mapping_file, output_file, total_frames):
    """将标注转换为逐帧标签文件
    
    Args:
        annotation_file: 输入的JSON标注文件路径
        mapping_file: 标签映射文件路径
        output_file: 输出的逐帧标签文件路径
        total_frames: 视频总帧数
    """
    # 加载标注和映射
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    label2idx, idx2label = load_mapping(mapping_file)
    
    # 创建帧级标签，默认为背景类(SIL, idx=0)
    frame_labels = np.zeros(total_frames, dtype=int)
    
    # 将注释应用于每一帧
    for anno in annotations:
        action = anno["action"]
        start_frame = anno["start_frame"]
        end_frame = anno["end_frame"]
        
        if action in label2idx:
            frame_labels[start_frame:end_frame+1] = label2idx[action]
        else:
            print(f"警告: 未找到动作 '{action}' 的映射")
    
    # 写入标签文件，每行一个标签名称(不是索引)
    with open(output_file, 'w') as f:
        for idx in frame_labels:
            label = idx2label[idx]
            f.write(f"{label}\n")
    
    print(f"已生成标签文件: {output_file}")
    
    # 返回一些统计信息
    unique, counts = np.unique(frame_labels, return_counts=True)
    stats = {idx2label[u]: c for u, c in zip(unique, counts)}
    return stats

def process_assembly_videos(base_dir="data/assembly", videos=None):
    """处理所有装配视频的标注
    
    Args:
        base_dir: 装配数据集根目录
        videos: 要处理的视频列表，如果为None则处理全部
                例如: ["Pump_Down_1", "Pump_Front_1", "BearingHousing_Down_1"]
    """
    mapping_file = os.path.join(base_dir, "mapping.txt")
    
    # 获取视频列表
    if videos is None:
        annotations_dir = os.path.join(base_dir, "annotations")
        videos = [f[:-5] for f in os.listdir(annotations_dir) 
                 if f.endswith('.json')]
    
    for video in videos:
        annotation_file = os.path.join(base_dir, "annotations", f"{video}.json")
        output_file = os.path.join(base_dir, "groundTruth", f"{video}.txt")
        
        # 获取视频总帧数(从特征文件或配置)
        # 这里假设有一个frame_counts.json文件存储了每个视频的帧数
        # 实际应用中可以从特征文件大小或其他来源获取
        frame_counts_file = os.path.join(base_dir, "frame_counts.json")
        if os.path.exists(frame_counts_file):
            with open(frame_counts_file, 'r') as f:
                frame_counts = json.load(f)
            total_frames = frame_counts.get(video, 1000)  # 默认1000帧
        else:
            # 如果没有帧数信息，可以从特征文件推断
            feature_file = os.path.join(base_dir, "features", f"{video}.npy")
            if os.path.exists(feature_file):
                feature = np.load(feature_file)
                total_frames = feature.shape[0]
            else:
                print(f"警告: 无法获取 {video} 的帧数，默认使用1000")
                total_frames = 1000
        
        # 转换标注
        if os.path.exists(annotation_file):
            stats = convert_annotations(annotation_file, mapping_file, 
                                      output_file, total_frames)
            print(f"{video} 标签统计: {stats}")
        else:
            print(f"错误: 找不到标注文件 {annotation_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换装配标注为FACT兼容的逐帧标签")
    parser.add_argument("--base-dir", default="data/assembly", 
                        help="装配数据集根目录")
    parser.add_argument("--videos", nargs="+", 
                        help="要处理的视频列表，格式为'产品_视角_组号'，例如'Pump_Down_1'")
    
    args = parser.parse_args()
    
    # 使用示例：
    # python utils/convert_assembly_annotations.py --videos Pump_Down_1 Pump_Front_1 BearingHousing_Down_1
    process_assembly_videos(args.base_dir, args.videos) 