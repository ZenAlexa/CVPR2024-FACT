import cv2
import numpy as np
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ActionAnnotator:
    def __init__(self, frames_dir, output_dir):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_frame = 0
        self.frames = sorted(list(self.frames_dir.glob("*.jpg")))
        self.annotations = []
        self.current_action = ""
        self.start_frame = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("动作标注工具")
        
        # 图片显示区域
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        
        # 动作输入
        action_frame = ttk.Frame(self.root)
        action_frame.pack(pady=10)
        ttk.Label(action_frame, text="动作名称:").pack(side=tk.LEFT)
        self.action_entry = ttk.Entry(action_frame)
        self.action_entry.pack(side=tk.LEFT, padx=5)
        
        # 控制按钮
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="开始动作", command=self.start_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="结束动作", command=self.end_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="下一帧", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存标注", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        
        self.show_frame()
        
    def show_frame(self):
        if self.current_frame < len(self.frames):
            image = Image.open(self.frames[self.current_frame])
            image = image.resize((800, 600))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
    def start_action(self):
        self.current_action = self.action_entry.get()
        self.start_frame = self.current_frame
        
    def end_action(self):
        if self.current_action:
            self.annotations.append({
                "action": self.current_action,
                "start_frame": self.start_frame,
                "end_frame": self.current_frame
            })
            self.current_action = ""
            
    def next_frame(self):
        self.current_frame += 1
        if self.current_frame < len(self.frames):
            self.show_frame()
            
    def save_annotations(self):
        output_file = self.output_dir / "annotations.json"
        with open(output_file, "w") as f:
            json.dump(self.annotations, f, indent=4)
        print(f"标注已保存到: {output_file}")
        
    def run(self):
        self.root.mainloop()

def main():
    annotator = ActionAnnotator(
        frames_dir="processed_data/video_1/frames",  # 帧序列目录
        output_dir="processed_data/video_1"  # 标注输出目录
    )
    annotator.run()

if __name__ == "__main__":
    main() 