import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def extract_features(frames_dir, output_file, batch_size=32, model_name="resnext101"):
    """从视频帧中提取深度特征
    
    Args:
        frames_dir: 包含视频帧的目录
        output_file: 输出特征文件(.npy)
        batch_size: 批处理大小
        model_name: 使用的预训练模型名称
    
    Returns:
        特征数组的形状
    """
    print(f"正在提取特征: {frames_dir} -> {output_file}")
    print(f"使用模型: {model_name}, 批大小: {batch_size}")
    
    # 加载预训练模型
    if model_name == "resnext101":
        model = models.resnext101_32x8d(pretrained=True)
        feature_dim = 2048
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        feature_dim = 2048
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        feature_dim = 1280
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 移除最后的全连接层，只保留特征提取部分
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU (较慢)")
    
    model = model.to(device)
    
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # 读取所有帧
    frames_dir = Path(frames_dir)
    frame_files = sorted([f for f in frames_dir.glob("*.jpg") or 
                        frames_dir.glob("*.png")])
    
    total_frames = len(frame_files)
    if total_frames == 0:
        print(f"错误: 在 {frames_dir} 中未找到帧")
        return None
    
    print(f"开始处理 {total_frames} 帧...")
    
    # 分批处理
    features = []
    for i in tqdm(range(0, total_frames, batch_size)):
        batch_files = frame_files[i:i+batch_size]
        batch_tensor = []
        
        for frame_file in batch_files:
            try:
                image = Image.open(frame_file).convert('RGB')
                tensor = preprocess(image)
                batch_tensor.append(tensor)
            except Exception as e:
                print(f"处理 {frame_file} 出错: {e}")
                # 使用零张量替代
                batch_tensor.append(torch.zeros(3, 224, 224))
        
        # 转换为批处理张量
        batch_tensor = torch.stack(batch_tensor)
        
        # 提取特征
        with torch.no_grad():
            batch_tensor = batch_tensor.to(device)
            batch_features = model(batch_tensor)
            
            # 展平特征 [B, C, 1, 1] -> [B, C]
            batch_features = batch_features.reshape(batch_features.size(0), -1)
            
            # 转移到CPU并转换为NumPy数组
            batch_features = batch_features.cpu().numpy()
            features.append(batch_features)
    
    # 合并所有特征
    features = np.vstack(features)
    
    # 显示特征统计信息
    print(f"特征形状: {features.shape}")
    print(f"特征统计: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    
    # 保存特征
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, features.astype(np.float32))
    
    print(f"特征已保存到: {output_file}")
    return features.shape

def extract_and_fuse_features(view1_dir, view2_dir, output_dir, product_name, group_id, batch_size=32):
    """处理双视角视频并融合特征
    
    Args:
        view1_dir: 第一个视角的帧目录
        view2_dir: 第二个视角的帧目录
        output_dir: 输出目录
        product_name: 产品名称
        group_id: 组号
        batch_size: 批处理大小
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取每个视角的特征
    print(f"处理产品: {product_name}, 组号: {group_id}")
    
    view1_feat_file = output_dir / f"{product_name}_Down_{group_id}.npy"
    view2_feat_file = output_dir / f"{product_name}_Front_{group_id}.npy"
    
    down_shape = extract_features(view1_dir, view1_feat_file, batch_size)
    front_shape = extract_features(view2_dir, view2_feat_file, batch_size)
    
    # 创建融合特征 (均值融合)
    print("创建融合特征...")
    if down_shape and front_shape:
        try:
            down_feat = np.load(view1_feat_file)
            front_feat = np.load(view2_feat_file)
            
            # 处理可能的长度不一致
            min_len = min(down_feat.shape[0], front_feat.shape[0])
            down_feat = down_feat[:min_len]
            front_feat = front_feat[:min_len]
            
            # 均值融合
            mean_fused = (down_feat + front_feat) / 2
            mean_fused_file = output_dir / f"{product_name}_Fused_{group_id}.npy"
            np.save(mean_fused_file, mean_fused.astype(np.float32))
            
            print(f"均值融合特征已保存: {mean_fused_file}")
            print(f"融合特征形状: {mean_fused.shape}")
            
        except Exception as e:
            print(f"融合特征创建失败: {e}")
    else:
        print("无法创建融合特征，因为视角特征提取失败")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从视频帧中提取深度特征")
    parser.add_argument("--frames", help="帧目录路径")
    parser.add_argument("--output", help="输出特征文件路径(.npy)")
    parser.add_argument("--batch", type=int, default=32, help="批处理大小")
    parser.add_argument("--model", default="resnext101", 
                      choices=["resnext101", "resnet50", "efficientnet"],
                      help="预训练模型名称")
    
    # 双视角处理参数
    parser.add_argument("--view1", help="俯视角(Down)帧目录")
    parser.add_argument("--view2", help="前视角(Front)帧目录")
    parser.add_argument("--product", help="产品名称")
    parser.add_argument("--group", help="组号")
    
    args = parser.parse_args()
    
    if args.view1 and args.view2 and args.product and args.group:
        # 双视角处理模式
        extract_and_fuse_features(
            args.view1, args.view2, 
            args.output or "data/assembly/features",
            args.product, args.group,
            args.batch
        )
    elif args.frames and args.output:
        # 单一视角处理模式
        extract_features(args.frames, args.output, args.batch, args.model)
    else:
        parser.print_help() 