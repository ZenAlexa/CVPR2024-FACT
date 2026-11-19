import cv2
import os
from pathlib import Path
import argparse

def extract_frames(video_path, output_dir, fps=None, max_frames=None):
    """从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出帧的目录
        fps: 提取的帧率 (None表示使用原始帧率)
        max_frames: 最大提取帧数 (None表示提取全部)
    
    Returns:
        保存的帧数量
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return 0
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {video_fps} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f} 秒")
    
    # 计算提取间隔
    if fps is None:
        # 使用视频原始帧率
        interval = 1
    else:
        # 按指定帧率提取
        interval = max(1, round(video_fps / fps))
    
    print(f"提取设置:")
    print(f"  每 {interval} 帧提取一帧")
    if max_frames:
        print(f"  最多提取 {max_frames} 帧")
    
    # 提取帧
    count = 0
    saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔提取
        if count % interval == 0:
            # 保存帧
            frame_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved += 1
            
            # 显示进度
            if saved % 100 == 0:
                print(f"已提取 {saved} 帧...")
            
            # 检查是否达到最大帧数
            if max_frames and saved >= max_frames:
                break
        
        count += 1
    
    # 释放资源
    cap.release()
    
    print(f"完成! 已提取 {saved} 帧到 {output_dir}")
    return saved

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中提取帧")
    parser.add_argument("--video", required=True, help="输入视频文件")
    parser.add_argument("--output", required=True, help="输出帧目录")
    parser.add_argument("--fps", type=float, help="提取的帧率 (默认使用视频原始帧率)")
    parser.add_argument("--max", type=int, help="最大提取帧数")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps, args.max) 