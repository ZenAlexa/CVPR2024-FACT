import cv2
import numpy as np
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class SimpleAnnotator:
    """简化版的装配动作标注工具"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("装配动作标注工具")
        self.root.geometry("1200x800")
        
        self.frames_dir = None
        self.output_dir = None
        self.mapping_file = None
        
        self.labels = []
        self.current_frame = 0
        self.total_frames = 0
        self.frames = []
        self.annotations = []
        self.current_action = ""
        self.start_frame = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        # 创建菜单
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开帧目录", command=self.open_frames_dir)
        file_menu.add_command(label="设置输出目录", command=self.set_output_dir)
        file_menu.add_command(label="加载映射文件", command=self.load_mapping_file)
        file_menu.add_separator()
        file_menu.add_command(label="保存标注", command=self.save_annotations)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)
        
        # 主界面分为左右两部分
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧 - 图像显示区
        left_frame = ttk.Frame(main_pane, width=800)
        main_pane.add(left_frame, weight=3)
        
        self.canvas = tk.Canvas(left_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 导航控制
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="◀◀", command=lambda: self.move_frames(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="◀", command=lambda: self.move_frames(-1)).pack(side=tk.LEFT, padx=2)
        self.frame_label = ttk.Label(nav_frame, text="0/0")
        self.frame_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="▶", command=lambda: self.move_frames(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="▶▶", command=lambda: self.move_frames(10)).pack(side=tk.LEFT, padx=2)
        
        self.frame_slider = ttk.Scale(nav_frame, orient=tk.HORIZONTAL, from_=0, to=100, 
                                     command=self.slider_moved)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # 右侧 - 标注控制
        right_frame = ttk.Frame(main_pane, width=300)
        main_pane.add(right_frame, weight=1)
        
        # 动作选择
        ttk.Label(right_frame, text="选择动作:").pack(anchor=tk.W, pady=5)
        self.action_listbox = tk.Listbox(right_frame, height=15)
        self.action_listbox.pack(fill=tk.X, pady=5)
        
        # 动作控制按钮
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="开始动作", command=self.start_action).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="结束动作", command=self.end_action).pack(fill=tk.X, pady=2)
        
        # 标注列表
        ttk.Label(right_frame, text="已标注动作:").pack(anchor=tk.W, pady=5)
        self.annotation_listbox = tk.Listbox(right_frame, height=15)
        self.annotation_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.annotation_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)
        
        # 删除按钮
        ttk.Button(right_frame, text="删除选中标注", command=self.delete_annotation).pack(fill=tk.X, pady=2)
        
        # 保存按钮
        ttk.Button(right_frame, text="保存标注", command=self.save_annotations).pack(fill=tk.X, pady=10)
        
    def open_frames_dir(self):
        """打开包含视频帧的目录"""
        dir_path = filedialog.askdirectory(title="选择帧目录")
        if not dir_path:
            return
            
        self.frames_dir = Path(dir_path)
        self.load_frames()
        
    def set_output_dir(self):
        """设置标注输出目录"""
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if not dir_path:
            return
            
        self.output_dir = Path(dir_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_mapping_file(self):
        """加载动作映射文件"""
        file_path = filedialog.askopenfilename(title="选择映射文件", 
                                            filetypes=[("文本文件", "*.txt")])
        if not file_path:
            return
            
        self.mapping_file = Path(file_path)
        self.load_labels()
        
    def load_labels(self):
        """从映射文件加载标签"""
        if not self.mapping_file or not self.mapping_file.exists():
            messagebox.showerror("错误", "映射文件不存在")
            return
            
        try:
            self.labels = []
            with open(self.mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        _, label = parts
                        self.labels.append(label)
            
            # 更新动作列表
            self.action_listbox.delete(0, tk.END)
            for label in self.labels:
                self.action_listbox.insert(tk.END, label)
                
            messagebox.showinfo("成功", f"已加载 {len(self.labels)} 个标签")
        except Exception as e:
            messagebox.showerror("错误", f"加载映射文件失败: {e}")
            
    def load_frames(self):
        """加载视频帧"""
        if not self.frames_dir or not self.frames_dir.exists():
            messagebox.showerror("错误", "帧目录不存在")
            return
            
        # 找到所有图像文件
        image_files = sorted([f for f in self.frames_dir.glob("*.jpg") or 
                             self.frames_dir.glob("*.png") or 
                             self.frames_dir.glob("*.jpeg")])
                             
        if not image_files:
            messagebox.showerror("错误", f"在 {self.frames_dir} 中未找到图像文件")
            return
            
        self.frames = image_files
        self.total_frames = len(self.frames)
        self.current_frame = 0
        
        # 更新滑块
        self.frame_slider.configure(from_=0, to=self.total_frames-1)
        
        # 尝试加载已有的标注
        self.load_existing_annotations()
        
        # 显示第一帧
        self.show_frame()
        
        messagebox.showinfo("成功", f"已加载 {self.total_frames} 帧")
        
    def load_existing_annotations(self):
        """尝试加载已有的标注"""
        if not self.output_dir:
            return
            
        # 基于帧目录名创建JSON文件名
        json_file = self.output_dir / f"{self.frames_dir.name}.json"
        
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    self.annotations = json.load(f)
                self.update_annotation_list()
                messagebox.showinfo("提示", f"已加载 {len(self.annotations)} 个已有标注")
            except Exception as e:
                messagebox.showerror("错误", f"加载已有标注失败: {e}")
                
    def show_frame(self):
        """显示当前帧"""
        if not self.frames or self.current_frame >= len(self.frames):
            return
            
        # 更新标签和滑块
        self.frame_label.configure(text=f"{self.current_frame+1}/{self.total_frames}")
        self.frame_slider.set(self.current_frame)
        
        # 加载图像
        try:
            image = Image.open(self.frames[self.current_frame])
            
            # 缩放图像适应画布
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 10 and canvas_height > 10:
                ratio = min(canvas_width/image.width, canvas_height/image.height)
                new_size = (int(image.width*ratio), int(image.height*ratio))
                image = image.resize(new_size, Image.LANCZOS)
                
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(
                canvas_width//2, canvas_height//2,
                image=photo
            )
            self.canvas.image = photo  # 防止被垃圾回收
            
            # 找到当前帧对应的标注
            current_action = "SIL"  # 默认背景动作
            for anno in self.annotations:
                if anno["start_frame"] <= self.current_frame <= anno["end_frame"]:
                    current_action = anno["action"]
                    break
                    
            # 显示当前动作
            if current_action:
                self.canvas.create_text(
                    10, 10, 
                    anchor=tk.NW,
                    text=f"当前动作: {current_action}",
                    fill="red",
                    font=("Arial", 16, "bold")
                )
                
        except Exception as e:
            messagebox.showerror("错误", f"显示帧失败: {e}")
            
    def move_frames(self, delta):
        """移动帧"""
        new_frame = self.current_frame + delta
        if 0 <= new_frame < self.total_frames:
            self.current_frame = new_frame
            self.show_frame()
            
    def slider_moved(self, value):
        """滑块移动回调"""
        frame = int(float(value))
        if 0 <= frame < self.total_frames:
            self.current_frame = frame
            self.show_frame()
            
    def start_action(self):
        """标记动作开始"""
        if not self.frames:
            messagebox.showinfo("提示", "请先加载帧序列")
            return
            
        selected = self.action_listbox.curselection()
        if not selected:
            messagebox.showinfo("提示", "请先选择一个动作")
            return
            
        self.current_action = self.action_listbox.get(selected[0])
        self.start_frame = self.current_frame
        
        messagebox.showinfo("提示", f"已标记 '{self.current_action}' 动作开始于帧 {self.start_frame}")
        
    def end_action(self):
        """标记动作结束"""
        if not self.current_action:
            messagebox.showinfo("提示", "请先使用'开始动作'标记起始帧")
            return
            
        end_frame = self.current_frame
        
        # 确保结束帧在开始帧之后
        if end_frame < self.start_frame:
            messagebox.showinfo("提示", "结束帧必须在开始帧之后")
            return
            
        # 添加标注
        annotation = {
            "action": self.current_action,
            "start_frame": self.start_frame,
            "end_frame": end_frame
        }
        
        # 检查与已有标注是否有重叠
        overlap = False
        for i, anno in enumerate(self.annotations):
            if (self.start_frame <= anno["end_frame"] and 
                end_frame >= anno["start_frame"]):
                overlap = True
                break
                
        if overlap:
            response = messagebox.askyesno(
                "提示", 
                "新标注与已有标注存在重叠，是否继续添加？"
            )
            if not response:
                return
        
        self.annotations.append(annotation)
        self.update_annotation_list()
        
        # 重置状态
        self.current_action = ""
        messagebox.showinfo("提示", f"已添加标注: {annotation}")
        
    def update_annotation_list(self):
        """更新标注列表"""
        self.annotation_listbox.delete(0, tk.END)
        
        # 按开始帧排序
        sorted_annotations = sorted(self.annotations, key=lambda x: x["start_frame"])
        
        for anno in sorted_annotations:
            text = f"{anno['action']}: {anno['start_frame']} → {anno['end_frame']}"
            self.annotation_listbox.insert(tk.END, text)
            
    def on_annotation_select(self, event):
        """当选择一个标注时"""
        selected = self.annotation_listbox.curselection()
        if not selected:
            return
            
        # 获取排序后的标注
        sorted_annotations = sorted(self.annotations, key=lambda x: x["start_frame"])
        index = selected[0]
        
        if 0 <= index < len(sorted_annotations):
            # 跳转到该标注的开始帧
            anno = sorted_annotations[index]
            self.current_frame = anno["start_frame"]
            self.show_frame()
            
    def delete_annotation(self):
        """删除选中的标注"""
        selected = self.annotation_listbox.curselection()
        if not selected:
            messagebox.showinfo("提示", "请先选择一个标注")
            return
            
        # 获取排序后的标注
        sorted_annotations = sorted(self.annotations, key=lambda x: x["start_frame"])
        index = selected[0]
        
        if 0 <= index < len(sorted_annotations):
            anno = sorted_annotations[index]
            self.annotations.remove(anno)
            self.update_annotation_list()
            messagebox.showinfo("提示", f"已删除标注: {anno}")
            
    def save_annotations(self):
        """保存标注到JSON文件"""
        if not self.annotations:
            messagebox.showinfo("提示", "没有标注可保存")
            return
            
        if not self.output_dir:
            self.set_output_dir()
            if not self.output_dir:
                return
                
        # 基于帧目录名创建JSON文件名
        json_file = self.output_dir / f"{self.frames_dir.name}.json"
        
        try:
            with open(json_file, 'w') as f:
                json.dump(self.annotations, f, indent=4)
            messagebox.showinfo("成功", f"已保存标注到 {json_file}")
        except Exception as e:
            messagebox.showerror("错误", f"保存标注失败: {e}")
    
    def run(self):
        """启动应用程序"""
        # 窗口大小改变时更新显示
        self.root.update()
        self.root.bind("<Configure>", lambda e: self.show_frame())
        self.root.mainloop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="装配动作标注工具")
    parser.add_argument("--frames", help="帧目录路径")
    parser.add_argument("--output", help="标注输出目录")
    parser.add_argument("--mapping", help="标签映射文件路径")
    
    args = parser.parse_args()
    
    app = SimpleAnnotator()
    
    # 如果提供了命令行参数，加载相应资源
    if args.frames:
        app.frames_dir = Path(args.frames)
        app.load_frames()
    
    if args.output:
        app.output_dir = Path(args.output)
        app.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mapping:
        app.mapping_file = Path(args.mapping)
        app.load_labels()
    
    app.run() 