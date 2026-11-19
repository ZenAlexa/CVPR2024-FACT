import cv2
import os
import numpy as np
from pathlib import Path

class VideoProcessor:
    def __init__(self, video_dir, output_dir):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames(self, video_path, frame_dir):
        """从视频中提取帧"""
        video = cv2.VideoCapture(str(video_path))
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # 保存帧
            frame_path = frame_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
            
        video.release()
        return frame_count
    
    def process_videos(self):
        """处理目录中的所有视频"""
        for video_file in self.video_dir.glob("*.mp4"):
            # 创建对应的输出目录
            video_name = video_file.stem
            frame_dir = self.output_dir / video_name / "frames"
            
            print(f"处理视频: {video_file}")
            frame_count = self.extract_frames(video_file, frame_dir)
            print(f"提取了 {frame_count} 帧")

def main():
    # 使用示例
    video_processor = VideoProcessor(
        video_dir="your_videos",  # 存放MP4视频的目录
        output_dir="processed_data"  # 输出目录
    )
    video_processor.process_videos()

if __name__ == "__main__":
    main() 