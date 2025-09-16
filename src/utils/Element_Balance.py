from tkinter import Tk, filedialog, Button, Label, StringVar, messagebox, ttk, Entry, DoubleVar, OptionMenu, IntVar, Scale, Frame
import os
import shutil
import random
import math
import json
import sys
from PIL import Image
import numpy as np

class DataAugmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset_Balance_V1.0_zhjoy")
        
        # 初始化路径变量
        self.source_path = StringVar()
        self.output_path = StringVar()
        self.element_bad_path = StringVar()   # 坏果要素库
        self.element_health_path = StringVar()  # 未成熟果实要素库
        self.element_leaf_path = StringVar()   # 树叶要素库
        self.element_ripe_path = StringVar()   # 成熟果实要素库
        
        # 初始化系数变量
        self.k1 = DoubleVar(value=0.5)  # 果实数量系数
        self.k2 = DoubleVar(value=0.1)  # 果实大小系数（默认0.1）
        self.k3 = DoubleVar(value=0.5)  # 遮挡策略缩放系数
        
        # 策略选择
        self.strategy_var = StringVar(value="添加策略")
        self.mix_ratio = DoubleVar(value=0.5)  # 混合策略比例
        
        # 设置默认路径
        self.default_paths = {
            "source_path": "C:/Users/zhj/Desktop/balance/mix/starfruit",
            "output_path": "C:/Users/zhj/Desktop/balance/mix/Aug_balance",
            "element_bad_path": "C:/Users/zhj/Desktop/balance/mix/bad",
            "element_health_path": "C:/Users/zhj/Desktop/balance/mix/health",
            "element_leaf_path": "C:/Users/zhj/Desktop/balance/mix/leaf",
            "element_ripe_path": "C:/Users/zhj/Desktop/balance/mix/ripe"
        }
        
        # 要素库计数
        self.bad_count = 0
        self.health_count = 0
        self.leaf_count = 0
        self.ripe_count = 0
        
        # 标签统计信息
        self.class_counts = {0: 0, 1: 0, 2: 0}  # 0:坏果, 1:未成熟果实, 2:成熟果实
        
        # 进度变量
        self.progress_var = DoubleVar(value=0.0)
        self.status_var = StringVar(value="准备就绪")
        
        # 创建界面组件
        self.create_widgets()
        
        # 加载保存的配置
        #self.load_config()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)


    def create_widgets(self):
        """创建图形界面组件"""
        # 策略选择框架
        frame_strategy = ttk.LabelFrame(self.root, text="0. 选择增强策略", padding=10)
        frame_strategy.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        strategies = ["添加策略", "遮挡策略", "混合模式"]
        OptionMenu(frame_strategy, self.strategy_var, *strategies, command=self.update_strategy_ui).grid(row=0, column=0, padx=5)
        
        # 混合策略比例滑块（初始隐藏）
        self.mix_ratio_frame = ttk.Frame(frame_strategy)
        self.mix_ratio_frame.grid(row=0, column=1, padx=5, sticky="w")
        Label(self.mix_ratio_frame, text="添加比例:").grid(row=0, column=0)
        self.mix_ratio_slider = Scale(self.mix_ratio_frame, from_=0.0, to=1.0, resolution=0.1, 
                                     orient="horizontal", variable=self.mix_ratio, 
                                     showvalue=True, length=150)
        self.mix_ratio_slider.grid(row=0, column=1, padx=5)
        self.mix_ratio_frame.grid_remove()  # 初始隐藏
        
        # 遮挡缩放系数
        ttk.Label(frame_strategy, text="遮挡缩放系数 k3:").grid(row=0, column=2, padx=(20, 0))
        ttk.Entry(frame_strategy, textvariable=self.k3, width=8).grid(row=0, column=3, padx=5)
        
        # 原始数据选择框架
        frame_source = ttk.LabelFrame(self.root, text="1. 选择原始数据文件夹", padding=10)
        frame_source.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_source, text="路径:").grid(row=0, column=0)
        Entry(frame_source, textvariable=self.source_path, width=50).grid(row=0, column=1)
        Button(frame_source, text="浏览...", command=self.select_source).grid(row=0, column=2)
        Button(frame_source, text="统计标签", command=self.count_labels).grid(row=0, column=3, padx=5)

        # 标签统计框架
        frame_stats = ttk.LabelFrame(self.root, text="标签统计", padding=10)
        frame_stats.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        self.stats_label = Label(frame_stats, text="坏果: 0 | 未成熟果实: 0 | 成熟果实: 0")
        self.stats_label.grid(row=0, column=0, columnspan=4)
        
        # 进度条和状态
        frame_progress = ttk.Frame(self.root)
        frame_progress.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        self.progress_bar = ttk.Progressbar(frame_progress, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        self.status_label = Label(frame_progress, textvariable=self.status_var)
        self.status_label.grid(row=0, column=1)

        # 输出目录选择框架
        frame_output = ttk.LabelFrame(self.root, text="2. 选择输出文件夹", padding=10)
        frame_output.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_output, text="路径:").grid(row=0, column=0)
        Entry(frame_output, textvariable=self.output_path, width=50).grid(row=0, column=1)
        Button(frame_output, text="浏览...", command=self.select_output).grid(row=0, column=2)

        # 坏果要素库选择框架
        frame_bad = ttk.LabelFrame(self.root, text="3. 选择坏果要素库文件夹", padding=10)
        frame_bad.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_bad, text="路径:").grid(row=0, column=0)
        Entry(frame_bad, textvariable=self.element_bad_path, width=50).grid(row=0, column=1)
        Button(frame_bad, text="浏览...", command=lambda: self.select_element(self.element_bad_path, "bad")).grid(row=0, column=2)
        self.bad_count_label = Label(frame_bad, text="元素个数: 0")
        self.bad_count_label.grid(row=0, column=3, padx=5)

        # 未成熟果实要素库选择框架
        frame_health = ttk.LabelFrame(self.root, text="4. 选择未成熟果实要素库文件夹", padding=10)
        frame_health.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_health, text="路径:").grid(row=0, column=0)
        Entry(frame_health, textvariable=self.element_health_path, width=50).grid(row=0, column=1)
        Button(frame_health, text="浏览...", command=lambda: self.select_element(self.element_health_path, "health")).grid(row=0, column=2)
        self.health_count_label = Label(frame_health, text="元素个数: 0")
        self.health_count_label.grid(row=0, column=3, padx=5)
        
        # 成熟果实要素库选择框架
        frame_ripe = ttk.LabelFrame(self.root, text="5. 选择成熟果实要素库文件夹", padding=10)
        frame_ripe.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_ripe, text="路径:").grid(row=0, column=0)
        Entry(frame_ripe, textvariable=self.element_ripe_path, width=50).grid(row=0, column=1)
        Button(frame_ripe, text="浏览...", command=lambda: self.select_element(self.element_ripe_path, "ripe")).grid(row=0, column=2)
        self.ripe_count_label = Label(frame_ripe, text="元素个数: 0")
        self.ripe_count_label.grid(row=0, column=3, padx=5)
        
        # 系数设置框架
        frame_coeff = ttk.LabelFrame(self.root, text="6. 设置系数", padding=10)
        frame_coeff.grid(row=8, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_coeff, text="数量系数 k1:").grid(row=0, column=0)
        Entry(frame_coeff, textvariable=self.k1, width=8).grid(row=0, column=1, padx=5)
        
        Label(frame_coeff, text="大小系数 k2:").grid(row=0, column=2)
        Entry(frame_coeff, textvariable=self.k2, width=8).grid(row=0, column=3, padx=5)

        # 处理按钮
        process_btn = ttk.Button(self.root, text="开始处理", command=self.process_balance)
        process_btn.grid(row=9, column=0, pady=5)
        
        # 平衡按钮
        balance_btn = ttk.Button(self.root, text="开始平衡", command=self.process_balance)
        balance_btn.grid(row=10, column=0, pady=10)
    
    def update_strategy_ui(self, *args):
        """根据选择的策略更新UI"""
        if self.strategy_var.get() == "混合模式":
            self.mix_ratio_frame.grid()
        else:
            self.mix_ratio_frame.grid_remove()

    def update_status(self, message, progress=None):
        """更新状态信息"""
        self.status_var.set(message)
        if progress is not None:
            self.progress_var.set(progress)
        self.root.update_idletasks()

    def select_source(self):
        """选择原始数据文件夹"""
        path = filedialog.askdirectory()
        if path:
            self.source_path.set(path)

    def select_output(self):
        """选择输出文件夹"""
        path = filedialog.askdirectory()
        if path:
            self.output_path.set(path)

    def select_element(self, path_var, element_type):
        """选择要素库文件夹并更新计数"""
        path = filedialog.askdirectory()
        if path:
            path_var.set(path)
            self.update_element_count(path, element_type)

    def update_element_count(self, path, element_type):
        """更新要素库元素计数"""
        if not path or not os.path.exists(path):
            return
            
        elements = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        count = len(elements)
        
        if element_type == "bad":
            self.bad_count = count
            if hasattr(self, 'bad_count_label'):  # 确保标签已创建
                self.bad_count_label.config(text=f"元素个数: {count}")
        elif element_type == "health":
            self.health_count = count
            if hasattr(self, 'health_count_label'):  # 确保标签已创建
                self.health_count_label.config(text=f"元素个数: {count}")
        elif element_type == "leaf":
            self.leaf_count = count
            if hasattr(self, 'leaf_count_label'):  # 确保标签已创建
                self.leaf_count_label.config(text=f"元素个数: {count}")
        elif element_type == "ripe":
            self.ripe_count = count
            if hasattr(self, 'ripe_count_label'):  # 确保标签已创建
                self.ripe_count_label.config(text=f"元素个数: {count}")

    def count_labels(self):
        """统计原始数据集中的标签数量"""
        source = self.source_path.get()
        if not source:
            messagebox.showerror("错误", "请先选择原始数据文件夹")
            return
            
        labels_dir = os.path.join(source, "labels")
        
        # 检查标签文件夹是否存在
        if not os.path.exists(labels_dir):
            messagebox.showerror("错误", f"标签文件夹不存在: {labels_dir}")
            return
        
        # 重置统计
        self.class_counts = {0: 0, 1: 0, 2: 0}  # 0:坏果, 1:未成熟果实, 2:成熟果实
        
        # 获取标签文件列表
        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        
        if not label_files:
            messagebox.showerror("错误", f"标签文件夹中没有找到任何标签文件: {labels_dir}")
            return
        
        print(f"找到 {len(label_files)} 个标签文件")
        
        # 遍历所有标签文件
        valid_files = 0
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                    if not lines:
                        print(f"警告: 空标签文件 - {label_file}")
                        continue
                    
                    valid_files += 1
                    for line in lines:
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue
                            
                        parts = line.split()
                        if len(parts) < 5:
                            print(f"警告: 无效行格式 - {label_file}: {line}")
                            continue
                            
                        try:
                            # 尝试解析类别和坐标
                            cls = int(float(parts[0]))
                            # 只统计0,1,2类
                            if cls in [0, 1, 2]:
                                self.class_counts[cls] += 1
                            else:
                                print(f"警告: 无效类别 {cls} - {label_file}: {line}")
                        except (ValueError, IndexError) as e:
                            print(f"解析错误 - {label_file}: {line} - {str(e)}")
            except Exception as e:
                print(f"文件读取错误 - {label_file}: {str(e)}")
        
        # 如果没有找到有效的标签文件
        if valid_files == 0:
            messagebox.showerror("错误", "没有找到有效的标签文件")
            return
        
        # 更新显示
        stats_text = f"坏果: {self.class_counts[0]} | 未成熟果实: {self.class_counts[1]} | 成熟果实: {self.class_counts[2]}"
        self.stats_label.config(text=stats_text)
        
        # 显示详细结果
        messagebox.showinfo("标签统计", 
            f"标签统计完成:\n{stats_text}\n\n"
            f"扫描文件: {len(label_files)} 个\n"
            f"有效文件: {valid_files} 个\n"
            f"坏果标签: {self.class_counts[0]}\n"
            f"未成熟果实标签: {self.class_counts[1]}\n"
            f"成熟果实标签: {self.class_counts[2]}")

    def validate_source(self):
        """验证原始文件夹结构"""
        source = self.source_path.get()
        required_folders = ["images", "labels"]
        required_files = ["classes.txt"]
        
        # 检查必要文件夹
        for folder in required_folders:
            if not os.path.exists(os.path.join(source, folder)):
                return f"原始文件夹缺少必要子文件夹: {folder}"
                
        # 检查必要文件
        for file in required_files:
            if not os.path.exists(os.path.join(source, file)):
                return f"原始文件夹缺少必要文件: {file}"
                
        return None

    def create_output_structure(self):
        """创建输出文件夹结构"""
        output = self.output_path.get()
        os.makedirs(os.path.join(output, "images"), exist_ok=True)
        os.makedirs(os.path.join(output, "labels"), exist_ok=True)

    def load_classes(self, classes_path):
        """加载类别文件"""
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def parse_yolo_label(self, label_path, img_width, img_height):
        """解析YOLO格式标签"""
        boxes = []
        if not os.path.exists(label_path):
            return boxes
            
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls, xc, yc, w, h = map(float, parts[:5])
                    # 转换为像素坐标
                    xc *= img_width
                    yc *= img_height
                    w *= img_width
                    h *= img_height
                    boxes.append((int(cls), xc, yc, w, h))
                except ValueError:
                    continue
        return boxes

    def get_random_element(self, element_folder):
        """随机获取要素库图片"""
        if not element_folder or not os.path.exists(element_folder):
            return None
            
        elements = [f for f in os.listdir(element_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not elements:
            return None
            
        element_path = os.path.join(element_folder, random.choice(elements))
        try:
            return Image.open(element_path).convert("RGBA")
        except:
            return None

    def calculate_area_metrics(self, boxes, img_width, img_height):
        """计算面积指标"""
        img_area = img_width * img_height
        boxes_area = 0.0
        boxes_count = len(boxes)
        
        if boxes_count > 0:
            for _, _, _, w, h in boxes:
                boxes_area += w * h
            avg_area = boxes_area / boxes_count
        else:
            avg_area = (img_width * img_height) / 100  # 如果没有标注框，使用图片面积的1%作为平均面积
        
        return img_area, boxes_area, avg_area

    def resize_element(self, element, target_area, k):
        """按比例调整要素大小 - 基于目标面积"""
        # 获取要素原始面积
        original_width, original_height = element.size
        original_area = original_width * original_height
        
        # 计算目标面积范围
        min_target_area = k * target_area
        max_target_area = target_area
        
        # 随机选择目标面积
        target_area = random.uniform(min_target_area, max_target_area)
        
        # 计算缩放因子
        if original_area > 0:
            scale_factor = math.sqrt(target_area / original_area)
        else:
            scale_factor = 1.0  # 如果原始面积为0，不缩放
            
        # 应用缩放
        new_width = max(10, int(original_width * scale_factor))
        new_height = max(10, int(original_height * scale_factor))
        
        return element.resize((new_width, new_height), Image.LANCZOS)

    def place_element(self, img, element, existing_boxes, avoid_overlap=True):
        """在指定位置放置要素图片"""
        img_width, img_height = img.size
        element_width, element_height = element.size
        
        # 如果要素太大，无法放置
        if element_width > img_width or element_height > img_height:
            return img, None
            
        # 创建可绘制的图像副本
        processed_img = img.copy()
        
        # 随机放置要素图片
        max_attempts = 100
        for _ in range(max_attempts):
            # 随机生成位置
            x = random.randint(0, img_width - element_width)
            y = random.randint(0, img_height - element_height)
            
            # 检查是否与任何现有框重叠
            overlap = False
            if avoid_overlap:
                for box in existing_boxes:
                    cls, bx, by, bw, bh = box
                    # 计算框的边界
                    box_rect = (bx - bw/2, by - bh/2, bx + bw/2, by + bh/2)
                    element_rect = (x, y, x + element_width, y + element_height)
                    
                    # 计算交集
                    x_overlap = max(0, min(box_rect[2], element_rect[2]) - max(box_rect[0], element_rect[0]))
                    y_overlap = max(0, min(box_rect[3], element_rect[3]) - max(box_rect[1], element_rect[1]))
                    
                    if x_overlap > 0 and y_overlap > 0:
                        overlap = True
                        break
            
            if not overlap:
                # 粘贴要素图片
                processed_img.paste(element, (int(x), int(y)), element)
                return processed_img, (x, y, element_width, element_height)
                
        return processed_img, None  # 未找到合适位置

    def calculate_overlap(self, rect1, rect2):
        """计算两个矩形的重叠面积百分比"""
        # rect: (x1, y1, x2, y2)
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
        overlap_area = x_overlap * y_overlap
        
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        return overlap_area / area1 if area1 > 0 else 0

    def occlusion_strategy(self, img, boxes, element_folder, element_class, k3, update_label=True):
        """
        通用遮挡策略
        update_label: 是否更新标签文件
        """
        processed_img = img.copy()
        img_width, img_height = img.size
        new_boxes = [] if update_label else None
        existing_boxes = boxes.copy()
        
        # 如果没有标注框，直接返回
        if not boxes:
            return processed_img, new_boxes
            
        # 随机选择要遮挡的标注框数量 (1到总框数之间)
        k = random.randint(1, len(boxes))
        
        # 随机选择k个标注框
        selected_boxes_idx = random.sample(range(len(boxes)), k)
        
        for idx in selected_boxes_idx:
            cls, cx, cy, w, h = boxes[idx]
            
            # 1. 获取随机要素
            element = self.get_random_element(element_folder)
            if element is None:
                continue
                
            # 2. 缩放要素图片 - 使用当前标注框的面积作为基准
            box_area = w * h
            element = self.resize_element(element, box_area, k3)
            ew, eh = element.size
            
            # 3. 定义标注框边界
            box_left = cx - w/2
            box_top = cy - h/2
            box_right = cx + w/2
            box_bottom = cy + h/2
            
            # 4. 尝试放置要素（确保部分在标注框内）
            max_attempts = 100
            placed = False
            
            for _ in range(max_attempts):
                # 随机生成位置（确保部分在标注框内）
                x = random.randint(int(box_left - ew/2), int(box_right))
                y = random.randint(int(box_top - eh/2), int(box_bottom))
                
                # 确保在图片范围内
                x = max(0, min(img_width - ew, x))
                y = max(0, min(img_height - eh, y))
                
                # 要素矩形
                element_rect = (x, y, x + ew, y + eh)
                
                # 计算与当前标注框的重叠率
                box_rect = (box_left, box_top, box_right, box_bottom)
                overlap_ratio = self.calculate_overlap(element_rect, box_rect)
                
                # 检查重叠率是否在10%~50%之间
                if 0.1 <= overlap_ratio <= 0.5:
                    # 检查与其他框的重叠
                    overlap_with_others = False
                    for i, box in enumerate(existing_boxes):
                        if i == idx:
                            continue  # 跳过当前被遮挡的框
                        
                        _, bx, by, bw, bh = box
                        other_box_rect = (bx - bw/2, by - bh/2, bx + bw/2, by + bh/2)
                        if self.calculate_overlap(element_rect, other_box_rect) > 0:
                            overlap_with_others = True
                            break
                    
                    if not overlap_with_others:
                        # 放置要素
                        processed_img.paste(element, (int(x), int(y)), element)
                        
                        # 更新标签
                        if update_label:
                            # 转换为YOLO格式
                            x_center = (x + ew/2) / img_width
                            y_center = (y + eh/2) / img_height
                            norm_w = ew / img_width
                            norm_h = eh / img_height
                            
                            # 添加到新框列表
                            new_boxes.append((element_class, x_center, y_center, norm_w, norm_h))
                            
                            # 添加到现有框列表
                            existing_boxes.append((element_class, x + ew/2, y + eh/2, ew, eh))
                        placed = True
                        break
            
            if not placed:
                # 如果无法放置，尝试在标注框附近放置
                try:
                    # 在标注框附近随机放置
                    x = random.randint(int(box_left - ew), int(box_right))
                    y = random.randint(int(box_top - eh), int(box_bottom))
                    
                    # 确保在图片范围内
                    x = max(0, min(img_width - ew, x))
                    y = max(0, min(img_height - eh, y))
                    
                    # 放置要素
                    processed_img.paste(element, (int(x), int(y)), element)
                    
                    # 更新标签
                    if update_label:
                        # 转换为YOLO格式
                        x_center = (x + ew/2) / img_width
                        y_center = (y + eh/2) / img_height
                        norm_w = ew / img_width
                        norm_h = eh / img_height
                        
                        # 添加到新框列表
                        new_boxes.append((element_class, x_center, y_center, norm_w, norm_h))
                        
                        # 添加到现有框列表
                        existing_boxes.append((element_class, x + ew/2, y + eh/2, ew, eh))
                except:
                    # 如果仍然失败，跳过
                    continue
                
        return processed_img, new_boxes

    def process_image(self, img_path, label_path, bad_folder, health_folder, leaf_folder, ripe_folder, k2, k3, 
                     output_img_path, output_label_path, add_bad=0, add_health=0, add_leaf=0, add_ripe=0, strategy="添加策略", mix_ratio=0.5):
        """处理单张图片
        add_bad, add_health, add_leaf, add_ripe: 需要添加的各类别数量
        strategy: 使用的策略
        mix_ratio: 混合策略中"添加策略"的比例
        """
        # 加载原始图片
        try:
            img = Image.open(img_path).convert('RGBA')
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
            return False, 0, 0, 0, 0
            
        img_width, img_height = img.size
        
        # 解析标签
        boxes = self.parse_yolo_label(label_path, img_width, img_height)
        
        added_bad = 0
        added_health = 0
        added_leaf = 0
        added_ripe = 0
        
        # 处理每个要添加的果实
        processed_img = img.copy()
        new_boxes = []
        
        # 准备现有框列表（原始框 + 新添加的框）
        existing_boxes = boxes.copy()
        
        # 添加元素函数
        def add_element(element_class, element_folder, count):
            nonlocal processed_img, added_bad, added_health, added_leaf, added_ripe
            added = 0
            for _ in range(count):
                element = self.get_random_element(element_folder)
                if element is None:
                    continue
                    
                # 调整要素大小 - 基于平均标注框面积
                if boxes:
                    avg_area = sum(w * h for _, _, _, w, h in boxes) / len(boxes)
                else:
                    avg_area = (img_width * img_height) / 100
                element = self.resize_element(element, avg_area, k2)
                
                # 放置要素
                placed_img, new_box = self.place_element(processed_img, element, existing_boxes)
                if new_box is None:
                    continue  # 无法放置
                    
                x, y, ew, eh = new_box
                processed_img = placed_img
                
                # 转换为YOLO格式
                x_center = (x + ew/2) / img_width
                y_center = (y + eh/2) / img_height
                norm_w = ew / img_width
                norm_h = eh / img_height
                
                # 添加到新框列表
                new_boxes.append((element_class, x_center, y_center, norm_w, norm_h))
                
                # 更新计数器
                if element_class == 0:
                    added_bad += 1
                elif element_class == 1:
                    added_health += 1
                elif element_class == 2:
                    added_ripe += 1
                elif element_class == 3:  # 树叶类别
                    added_leaf += 1
                
                added += 1
                
                # 添加到现有框列表（用于后续碰撞检测）
                existing_boxes.append((element_class, x + ew/2, y + eh/2, ew, eh))
            return added
        
        # 添加坏果
        if add_bad > 0:
            # 根据策略决定处理方式
            if strategy == "添加策略" or (strategy == "混合模式" and random.random() < mix_ratio):
                added_bad += add_element(0, bad_folder, add_bad)
            else:
                # 遮挡策略
                _, new_boxes_occlusion = self.occlusion_strategy(
                    processed_img, boxes, leaf_folder, 3, k3, True
                )
                if new_boxes_occlusion:
                    new_boxes.extend(new_boxes_occlusion)
                    added_leaf += len(new_boxes_occlusion)
        
        # 添加未成熟果实
        if add_health > 0:
            if strategy == "添加策略" or (strategy == "混合模式" and random.random() < mix_ratio):
                added_health += add_element(1, health_folder, add_health)
            else:
                # 遮挡策略
                _, new_boxes_occlusion = self.occlusion_strategy(
                    processed_img, boxes, leaf_folder, 3, k3, True
                )
                if new_boxes_occlusion:
                    new_boxes.extend(new_boxes_occlusion)
                    added_leaf += len(new_boxes_occlusion)
        
        # 添加成熟果实
        if add_ripe > 0:
            if strategy == "添加策略" or (strategy == "混合模式" and random.random() < mix_ratio):
                added_ripe += add_element(2, ripe_folder, add_ripe)
            else:
                # 遮挡策略
                _, new_boxes_occlusion = self.occlusion_strategy(
                    processed_img, boxes, leaf_folder, 3, k3, True
                )
                if new_boxes_occlusion:
                    new_boxes.extend(new_boxes_occlusion)
                    added_leaf += len(new_boxes_occlusion)
        
        # 添加树叶
        if add_leaf > 0:
            added_leaf += add_element(3, leaf_folder, add_leaf)
        
        # 保存处理后的图像
        try:
            processed_img = processed_img.convert('RGB')
            processed_img.save(output_img_path)
        except Exception as e:
            print(f"无法保存图片 {output_img_path}: {e}")
            return False, 0, 0, 0, 0

        # 保存处理后的标签
        try:
            with open(output_label_path, 'w') as f:
                # 写入原始标签
                for box in boxes:
                    cls, xc, yc, w, h = box
                    f.write(f"{cls} {xc/img_width:.6f} {yc/img_height:.6f} {w/img_width:.6f} {h/img_height:.6f}\n")
                
                # 写入新添加的要素标签
                for box in new_boxes:
                    cls, xc, yc, w, h = box
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"无法保存标签 {output_label_path}: {e}")
            return False, 0, 0, 0, 0

        return True, added_bad, added_health, added_leaf, added_ripe

    def process_balance(self):
        """平衡处理主函数 - 确保每张图片都被添加新标签且三类标签数量一致"""
        # 验证输入路径
        if not all([self.source_path.get(), self.output_path.get(), 
                self.element_bad_path.get(), self.element_health_path.get(),
                self.element_ripe_path.get()]):
            messagebox.showerror("错误", "请完整选择所有文件夹路径")
            return

        # 验证原始文件夹结构
        validation_msg = self.validate_source()
        if validation_msg:
            messagebox.showerror("错误", validation_msg)
            return
            
        # 确保已经统计过标签
        if sum(self.class_counts.values()) == 0:
            self.count_labels()
            if sum(self.class_counts.values()) == 0:
                messagebox.showerror("错误", "请先统计标签数量")
                return

        # 获取各类别数量
        bad_count = self.class_counts[0]
        health_count = self.class_counts[1]
        ripe_count = self.class_counts[2]  # 成熟果实数量

        # 计算需要平衡的目标数量（三个类别中的最大值）
        target_count = max(bad_count, health_count, ripe_count)
        
        if target_count == 0:
            messagebox.showerror("错误", "未找到需要平衡的类别")
            return
            
        # 计算需要添加的数量
        need_bad = max(0, target_count - bad_count)
        need_health = max(0, target_count - health_count)
        need_ripe = max(0, target_count - ripe_count)
        
        if need_bad == 0 and need_health == 0 and need_ripe == 0:
            messagebox.showinfo("无需平衡", "所有类别数量已经平衡")
            return
            
        # 获取系数值和策略
        k2 = self.k2.get()
        k3 = self.k3.get()
        strategy = self.strategy_var.get()
        mix_ratio = self.mix_ratio.get()
        
        # 创建输出目录结构
        self.create_output_structure()

        try:
            # 准备原始图片
            source_images = os.path.join(self.source_path.get(), "images")
            source_labels = os.path.join(self.source_path.get(), "labels")
            
            img_files = [f for f in os.listdir(source_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_original = len(img_files)
            
            if total_original == 0:
                messagebox.showerror("错误", "原始图片文件夹中没有图片")
                return
                
            # 计算总需要添加的元素数量
            total_elements = need_bad + need_health + need_ripe
            
            # 计算每张图片需要添加的平均元素数量
            base_per_image = total_elements // total_original
            remainder = total_elements % total_original
            
            # 计算各类别比例
            if total_elements > 0:
                bad_ratio = need_bad / total_elements
                health_ratio = need_health / total_elements
                ripe_ratio = need_ripe / total_elements
            else:
                bad_ratio = health_ratio = ripe_ratio = 0
            
            # 初始化计数器
            added_bad_total = 0
            added_health_total = 0
            added_leaf_total = 0
            added_ripe_total = 0
            
            # 创建处理队列
            img_files_processed = []
            
            # 第一轮：均匀分配元素
            for idx, img_file in enumerate(img_files, 1):
                # 更新状态
                progress = idx / total_original * 100
                status = (f"处理中: {idx}/{total_original} | "
                         f"坏果: {added_bad_total}/{need_bad} | "
                         f"未成熟: {added_health_total}/{need_health} | "
                         f"成熟: {added_ripe_total}/{need_ripe}")
                self.update_status(status, progress)
                
                # 构建文件路径
                img_path = os.path.join(source_images, img_file)
                base_name, ext = os.path.splitext(img_file)
                label_file = base_name + ".txt"
                label_path = os.path.join(source_labels, label_file)
                output_img_path = os.path.join(self.output_path.get(), "images", img_file)
                output_label_path = os.path.join(self.output_path.get(), "labels", label_file)
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

                # 计算该图片需要添加的元素总数
                total_add = base_per_image
                if idx <= remainder:
                    total_add += 1
                
                # 根据比例分配各类别数量
                add_bad = 0
                add_health = 0
                add_ripe = 0
                add_leaf = 0  # 遮挡策略会产生树叶
                
                for i in range(total_add):
                    r = random.random()
                    if r < bad_ratio:
                        add_bad += 1
                    elif r < bad_ratio + health_ratio:
                        add_health += 1
                    else:
                        add_ripe += 1
                
                # 处理图片
                success, added_b, added_h, added_l, added_r = self.process_image(
                    img_path, 
                    label_path, 
                    self.element_bad_path.get(),
                    self.element_health_path.get(),
                    self.element_leaf_path.get(),
                    self.element_ripe_path.get(),
                    k2, k3,
                    output_img_path,
                    output_label_path,
                    add_bad, add_health, add_leaf, add_ripe,
                    strategy, mix_ratio
                )
                
                if success:
                    # 更新计数器
                    added_bad_total += added_b
                    added_health_total += added_h
                    added_leaf_total += added_l
                    added_ripe_total += added_r
                else:
                    # 如果处理失败，复制原始文件
                    shutil.copy(img_path, output_img_path)
                    if os.path.exists(label_path):
                        shutil.copy(label_path, output_label_path)
                    print(f"处理图片失败，已复制原始文件: {img_file}")
                    
                # 添加到已处理列表
                img_files_processed.append(img_file)
            
            # 第二轮：检查是否达到目标，如果未达到则继续处理
            if (added_bad_total < need_bad or 
                added_health_total < need_health or 
                added_ripe_total < need_ripe):
                
                # 随机打乱图片顺序
                random.shuffle(img_files)
                
                for img_file in img_files:
                    # 如果已经达到所有目标，提前结束
                    if (added_bad_total >= need_bad and 
                        added_health_total >= need_health and 
                        added_ripe_total >= need_ripe):
                        break
                        
                    # 跳过已处理过的图片
                    if img_file in img_files_processed:
                        continue
                        
                    # 更新状态
                    status = (f"补充处理 | "
                             f"坏果: {added_bad_total}/{need_bad} | "
                             f"未成熟: {added_health_total}/{need_health} | "
                             f"成熟: {added_ripe_total}/{need_ripe}")
                    self.update_status(status)
                    
                    # 构建文件路径
                    img_path = os.path.join(source_images, img_file)
                    base_name, ext = os.path.splitext(img_file)
                    label_file = base_name + ".txt"
                    label_path = os.path.join(source_labels, label_file)
                    output_img_path = os.path.join(self.output_path.get(), "images", img_file)
                    output_label_path = os.path.join(self.output_path.get(), "labels", label_file)
                    
                    # 计算该图片需要添加的各类别数量
                    add_bad = min(3, need_bad - added_bad_total)
                    add_health = min(3, need_health - added_health_total)
                    add_ripe = min(3, need_ripe - added_ripe_total)
                    add_leaf = 0
                    
                    # 处理图片
                    success, added_b, added_h, added_l, added_r = self.process_image(
                        img_path, 
                        label_path, 
                        self.element_bad_path.get(),
                        self.element_health_path.get(),
                        self.element_leaf_path.get(),
                        self.element_ripe_path.get(),
                        k2, k3,
                        output_img_path,
                        output_label_path,
                        add_bad, add_health, add_leaf, add_ripe,
                        strategy, mix_ratio
                    )
                    
                    if success:
                        # 更新计数器
                        added_bad_total += added_b
                        added_health_total += added_h
                        added_leaf_total += added_l
                        added_ripe_total += added_r
                        
                        # 添加到已处理列表
                        img_files_processed.append(img_file)

            # 显示结果
            final_bad = bad_count + added_bad_total
            final_health = health_count + added_health_total
            final_ripe = ripe_count + added_ripe_total
            
            # 更新标签统计显示
            self.stats_label.config(text=f"坏果: {final_bad} | 未成熟果实: {final_health} | 成熟果实: {final_ripe}")
            
            # 更新内存中的统计信息
            self.class_counts[0] = final_bad
            self.class_counts[1] = final_health
            self.class_counts[2] = final_ripe
            
            result_message = (
                f"平衡处理完成!\n\n"
                f"原始统计:\n"
                f"  坏果: {bad_count}\n"
                f"  未成熟果实: {health_count}\n"
                f"  成熟果实: {ripe_count}\n\n"
                f"添加统计:\n"
                f"  坏果: {added_bad_total}\n"
                f"  未成熟果实: {added_health_total}\n"
                f"  成熟果实: {added_ripe_total}\n"
                f"  树叶: {added_leaf_total}\n\n"
                f"最终统计:\n"
                f"  坏果: {final_bad}\n"
                f"  未成熟果实: {final_health}\n"
                f"  成熟果实: {final_ripe}\n\n"
                f"处理图片: {len(img_files_processed)} 张"
            )
            
            if (added_bad_total >= need_bad and 
                added_health_total >= need_health and 
                added_ripe_total >= need_ripe):
                messagebox.showinfo("平衡成功", result_message)
            else:
                messagebox.showwarning("平衡未完全完成", 
                    result_message + "\n\n" +
                    f"未达到目标数量:\n"
                    f"  坏果: {need_bad - added_bad_total} 个\n"
                    f"  未成熟果实: {need_health - added_health_total} 个\n"
                    f"  成熟果实: {need_ripe - added_ripe_total} 个\n\n"
                    f"可能需要调整系数或增加要素库元素")
            
            # 重置状态
            self.update_status("平衡完成", 100)
            self.root.title("Dataset_Balance_V1.0_zhjoy")
            
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            messagebox.showerror("错误", f"平衡处理过程中发生错误: {str(e)}")
            self.root.title("Dataset_Balance_V1.0_zhjoy")

    # 其他方法保持不变（load_config, save_config, on_close等）

if __name__ == "__main__":
    root = Tk()
    app = DataAugmentationApp(root)
    
    # 设置窗口大小
    root.geometry("850x750")
    
    # 运行主循环
    root.mainloop()