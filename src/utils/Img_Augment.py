import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'complex'):
    np.complex = complex

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import random
import numpy as np
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import shutil

class YOLOAugApp:
    """YOLO Dataset Augmentation Tool GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 数据集增广工具")  # YOLO Dataset Augmentation Tool
        self.root.geometry("800x680")  # Reduced height after removing fog limit controls

        # Subset selection variables (train, val, test)
        self.subset_vars = {
            "train": tk.BooleanVar(value=False),
            "val": tk.BooleanVar(value=False),
            "test": tk.BooleanVar(value=False)
        }
        self.subset_count_labels = {}

        # Augmentation parameters configuration
        # Fog augmentation has no parameters (simplified version)
        self.augmentation_params = {
            "MotionBlur": {"k": (5, 20), "angle": (-45, 45)},
            "GaussianBlur": {"sigma": (0, 5)},
            "AdditiveGaussianNoise": {"scale": (0, 20)},
            "Fog": {},  # No parameters for fog augmentation
            "Rain": {"drop_size": (0.1, 0.2), "speed": (0.1, 0.3)},
            "Affine": {"scale": (0.7, 1.3)},
            "Rotate": {"angle": (-45, 45)},
            "Rot90": {"k": (1, 3)},
            "TranslateX": {"percent": (-0.2, 0.2)},
            "TranslateY": {"percent": (-0.2, 0.2)},
            "GammaContrast": {"gamma": (0.5, 2.0)},
            "SigmoidContrast": {"gain": (3, 10), "cutoff": (0.4, 0.6)},
            "MultiplyBrightness": {"mul": (0.8, 1.2)}
        }

        # Parameter ranges for validation and UI hints
        self.param_ranges = {
            "k": (3, 100, "整数"),
            "angle": (-360, 360, "角度"),
            "sigma": (0, 50, "模糊度"),
            "scale": (0, 1, "比例"),
            "drop_size": (0.01, 0.5, "雨滴大小"),
            "speed": (0.01, 0.5, "雨速"),
            "percent": (-0.5, 0.5, "平移比例"),
            "gamma": (0.1, 3.0, "对比度"),
            "gain": (1, 20, "增益"),
            "cutoff": (0.1, 0.9, "阈值"),
            "mul": (0.1, 3.0, "亮度系数")
        }

        # Application state variables
        self.selected_folder = None
        self.aug_methods_vars = {}  # Augmentation method selection variables
        self.param_widgets = {}     # Parameter input widgets
        
        # Minimum bounding box area ratio threshold (relative to original area)
        self.min_area_ratio = 0.10  # Default: 10% of original area

        # Build the user interface
        self.build_ui()

    def build_ui(self):
        """Build the main user interface"""
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        # Dataset folder selection
        ttk.Label(frm, text="数据集文件夹:").grid(row=0, column=0, sticky="w")  # Dataset folder
        self.folder_entry = ttk.Entry(frm, width=50)
        self.folder_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(frm, text="浏览", command=self.select_folder).grid(row=0, column=2)  # Browse

        # Subset selection checkboxes with image count display
        row_idx = 1
        for subset in ["train", "val", "test"]:
            cb = ttk.Checkbutton(frm, text=subset, variable=self.subset_vars[subset], command=self.update_counts)
            cb.grid(row=row_idx, column=0, sticky="w")
            lbl = ttk.Label(frm, text="图片数量: 0")  # Image count: 0
            lbl.grid(row=row_idx, column=1, sticky="w")
            self.subset_count_labels[subset] = lbl
            row_idx += 1

        # Augmentation target count input
        ttk.Label(frm, text="增广目标数量:").grid(row=row_idx, column=0, sticky="w")  # Augmentation target count
        self.aug_target_entry = ttk.Entry(frm, width=20)
        self.aug_target_entry.grid(row=row_idx, column=1, sticky="w")

        # Multiple augmentation option
        row_idx += 1
        self.is_multiple_var = tk.BooleanVar()
        cb_multiple = ttk.Checkbutton(frm, text="倍数增广 (勾选则目标数量视为倍数)", variable=self.is_multiple_var)
        cb_multiple.grid(row=row_idx, column=0, sticky="w")  # Multiple augmentation (target as multiplier)
        
        # Minimum bounding box area ratio setting
        row_idx += 1
        ttk.Label(frm, text="边界框最小保留比例:").grid(row=row_idx, column=0, sticky="w")  # Minimum bbox retention ratio
        self.min_area_entry = ttk.Entry(frm, width=10)
        self.min_area_entry.insert(0, "0.10")  # Default: 10%
        self.min_area_entry.grid(row=row_idx, column=1, sticky="w")
        ttk.Label(frm, text="(0-1之间的小数，例如0.05表示保留至少5%面积的边界框)").grid(row=row_idx, column=2, sticky="w")
        # (Decimal between 0-1, e.g., 0.05 means keep bboxes with at least 5% area)

        # Augmentation methods selection
        row_idx += 1
        ttk.Label(frm, text="选择增广方法:").grid(row=row_idx, column=0, sticky="nw")  # Select augmentation methods

        aug_frame = ttk.Frame(frm)
        aug_frame.grid(row=row_idx, column=1, columnspan=2, sticky="w", pady=5)

        # Create checkboxes and parameter inputs for each augmentation method
        row_a = 0
        for method, params in self.augmentation_params.items():
            var = tk.BooleanVar()
            self.aug_methods_vars[method] = var
            cb = ttk.Checkbutton(aug_frame, text=method, variable=var, command=self.toggle_param_widgets)
            cb.grid(row=row_a, column=0, sticky="w")

            col_idx = 1
            self.param_widgets[method] = {}
            for pname, prange in params.items():
                # Parameter label
                ttk.Label(aug_frame, text=f"{pname}:").grid(row=row_a, column=col_idx, sticky="e")
                
                # Parameter input fields
                if isinstance(prange, tuple):
                    # Range input (min, max)
                    entry_min = ttk.Entry(aug_frame, width=5)
                    entry_max = ttk.Entry(aug_frame, width=5)
                    entry_min.insert(0, str(prange[0]))
                    entry_max.insert(0, str(prange[1]))
                    entry_min.grid(row=row_a, column=col_idx+1)
                    entry_max.grid(row=row_a, column=col_idx+2)
                    self.param_widgets[method][pname] = (entry_min, entry_max)
                    col_idx += 3
                else:
                    # Single value input
                    entry = ttk.Entry(aug_frame, width=8)
                    entry.insert(0, str(prange))
                    entry.grid(row=row_a, column=col_idx+1)
                    self.param_widgets[method][pname] = entry
                    col_idx += 2
                
                # Add parameter range hint labels
                if pname in self.param_ranges:
                    min_val, max_val, unit = self.param_ranges[pname]
                    range_label = ttk.Label(aug_frame, text=f"[{min_val}-{max_val}]", foreground="gray")
                    range_label.grid(row=row_a, column=col_idx, padx=(5, 0))
                    col_idx += 1
            row_a += 1

        # Initial state setup for parameter widgets
        self.toggle_param_widgets()

        # Start augmentation button
        row_idx += 1
        ttk.Button(frm, text="开始增广", command=self.start_augmentation).grid(row=row_idx, column=0, sticky="w")  # Start Augmentation

        # Log text area
        row_idx += 1
        self.log_text = tk.Text(frm, height=15)
        self.log_text.grid(row=row_idx, column=0, columnspan=3, sticky="nsew")

        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(row_idx, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def toggle_param_widgets(self):
        """Enable/disable parameter input widgets based on method selection"""
        for method, var in self.aug_methods_vars.items():
            enabled = var.get()
            for pname, widget in self.param_widgets.get(method, {}).items():
                if isinstance(widget, tuple):
                    # Range widgets (min, max)
                    for w in widget:
                        w.config(state="normal" if enabled else "disabled")
                else:
                    # Single value widget
                    widget.config(state="normal" if enabled else "disabled")

    def select_folder(self):
        """Open folder selection dialog and update UI"""
        folder = filedialog.askdirectory(title="选择YOLO格式数据集文件夹")  # Select YOLO format dataset folder
        if folder:
            self.selected_folder = Path(folder)
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)
            self.update_counts()

    def update_counts(self):
        """Update image count labels for each subset"""
        if not self.selected_folder:
            return
        for subset in ["train", "val", "test"]:
            lbl = self.subset_count_labels.get(subset)
            img_dir = self.selected_folder / "images" / subset
            if lbl and img_dir.exists():
                count = len(list(img_dir.glob("*.jpg")))
                lbl.config(text=f"图片数量: {count}")  # Image count
            elif lbl:
                lbl.config(text="图片数量: 0")  # Image count: 0

    def get_selected_augmenters(self):
        """Create and return combined augmenter based on selected methods and parameters"""
        augmenters = []
        for method, var in self.aug_methods_vars.items():
            if var.get():
                params = {}
                # Extract parameters from UI widgets
                for pname, widget in self.param_widgets.get(method, {}).items():
                    if isinstance(widget, tuple):
                        # Range parameter (min, max)
                        try:
                            vmin = float(widget[0].get())
                            vmax = float(widget[1].get())
                            if vmin == vmax:
                                params[pname] = vmin
                            else:
                                params[pname] = (vmin, vmax)
                        except Exception:
                            pass
                    else:
                        # Single value parameter
                        try:
                            params[pname] = float(widget.get())
                        except Exception:
                            pass
                
                # Create augmenter instance
                aug = self.create_augmenter(method, params)
                if aug is not None:
                    augmenters.append(aug)
        
        # Return combined augmenter using OneOf (randomly select one method per image)
        if augmenters:
            return iaa.OneOf(augmenters)
        return None

    def create_augmenter(self, method, params):
        """Create specific augmenter instance based on method name and parameters"""
        try:
            if method == "MotionBlur":
                k = self.get_param_range(params, "k", 5, 80, int)
                angle = self.get_param_range(params, "angle", -45, 45)
                return iaa.MotionBlur(k=k, angle=angle)
            elif method == "GaussianBlur":
                sigma = self.get_param_range(params, "sigma", 0, 20)
                return iaa.GaussianBlur(sigma=sigma)
            elif method == "AdditiveGaussianNoise":
                scale = self.get_param_range(params, "scale", 0, 0.3*255)
                return iaa.AdditiveGaussianNoise(scale=scale)
            elif method == "Fog":
                # Fog augmentation with no parameters
                return iaa.Fog()
            elif method == "Rain":
                drop_size = self.get_param_range(params, "drop_size", 0.1, 0.2)
                speed = self.get_param_range(params, "speed", 0.1, 0.3)
                return iaa.Rain(drop_size=drop_size, speed=speed)
            elif method == "Affine":
                scale = self.get_param_range(params, "scale", 0.5, 1.5)
                return iaa.Affine(scale=scale)
            elif method == "Rotate":
                angle = self.get_param_range(params, "angle", -45, 45)
                return iaa.Rotate(angle)
            elif method == "Rot90":
                k = self.get_param_range(params, "k", 1, 3, int)
                return iaa.Rot90(k=k, keep_size=False)
            elif method == "TranslateX":
                percent = self.get_param_range(params, "percent", -0.2, 0.2)
                return iaa.TranslateX(percent=percent)
            elif method == "TranslateY":
                percent = self.get_param_range(params, "percent", -0.2, 0.2)
                return iaa.TranslateY(percent=percent)
            elif method == "GammaContrast":
                gamma = self.get_param_range(params, "gamma", 0.5, 2.0)
                return iaa.GammaContrast(gamma=gamma)
            elif method == "SigmoidContrast":
                gain = self.get_param_range(params, "gain", 3, 10)
                cutoff = self.get_param_range(params, "cutoff", 0.4, 0.6)
                return iaa.SigmoidContrast(gain=gain, cutoff=cutoff)
            elif method == "MultiplyBrightness":
                mul = self.get_param_range(params, "mul", 0.5, 1.5)
                return iaa.MultiplyBrightness(mul=mul)
        except Exception as e:
            self.log_write(f"创建增强器 {method} 时出错: {e}")  # Error creating augmenter
        return None

    def get_param_range(self, params, pname, default_min, default_max, cast=float):
        """Extract parameter value from params dict with default fallback"""
        val = params.get(pname, None)
        if val is None:
            return (default_min, default_max)
        if isinstance(val, tuple):
            vmin, vmax = val
            return (cast(vmin), cast(vmax)) if vmin != vmax else cast(vmin)
        else:
            return cast(val)

    def start_augmentation(self):
        """Validate inputs and start augmentation process in separate thread"""
        # Validate dataset folder selection
        if not self.selected_folder:
            messagebox.showwarning("警告", "请先选择数据集文件夹")  # Please select dataset folder first
            return
        
        # Validate subset selection
        subsets = [s for s, var in self.subset_vars.items() if var.get()]
        if not subsets:
            messagebox.showwarning("警告", "请选择至少一个子集进行增广")  # Please select at least one subset
            return
        
        # Validate target count input
        try:
            target = int(self.aug_target_entry.get())
            if target <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("警告", "请输入有效的增广目标数量")  # Please enter valid target count
            return
            
        # Validate minimum area ratio input
        try:
            min_area_ratio = float(self.min_area_entry.get())
            if min_area_ratio < 0 or min_area_ratio > 1:
                raise ValueError
        except Exception:
            messagebox.showwarning("警告", "请输入有效的边界框最小保留比例 (0-1之间的小数)")  # Please enter valid bbox min ratio
            return
        self.min_area_ratio = min_area_ratio

        # Validate augmentation method selection
        self.augmenter = self.get_selected_augmenters()
        if self.augmenter is None:
            messagebox.showwarning("警告", "请至少选择一种增广方法")  # Please select at least one augmentation method
            return

        # Store configuration for background thread
        self.is_multiple = self.is_multiple_var.get()
        self.subsets = subsets
        self.target_aug = target

        # Initialize log and start processing
        self.log_text.delete(1.0, tk.END)
        self.log_write("开始增广...")  # Starting augmentation...
        self.log_write(f"边界框最小保留比例: {min_area_ratio*100:.1f}%")  # Bbox minimum retention ratio

        # Start augmentation in background thread
        threading.Thread(target=self.run_augmentation, daemon=True).start()

    def run_augmentation(self):
        """Main augmentation processing function (runs in background thread)"""
        base_dir = self.selected_folder
        out_dir = base_dir.parent / f"{base_dir.name}_AUG"
        
        # Create output directory structure
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        # Copy classes.txt file if exists
        classes_src = base_dir / "labels" / "classes.txt"
        classes_dst = out_dir / "labels" / "classes.txt"
        if classes_src.exists():
            classes_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(classes_src, classes_dst)

        # Create all necessary subdirectories
        for sub in ["train", "val", "test"]:
            for folder in [out_dir / "images" / sub, out_dir / "labels" / sub]:
                folder.mkdir(parents=True, exist_ok=True)

        # Copy data for non-selected subsets (preserve original data)
        for sub in ["train", "val", "test"]:
            if sub not in self.subsets:
                src_img_dir = base_dir / "images" / sub
                dst_img_dir = out_dir / "images" / sub
                src_lbl_dir = base_dir / "labels" / sub
                dst_lbl_dir = out_dir / "labels" / sub
                if src_img_dir.exists():
                    shutil.copytree(src_img_dir, dst_img_dir, dirs_exist_ok=True)
                if src_lbl_dir.exists():
                    shutil.copytree(src_lbl_dir, dst_lbl_dir, dirs_exist_ok=True)
        
        # Process each selected subset
        for subset in self.subsets:
            img_dir = base_dir / "images" / subset
            lbl_dir = base_dir / "labels" / subset
            out_img_dir = out_dir / "images" / subset
            out_lbl_dir = out_dir / "labels" / subset

            # Copy original images and labels for selected subsets
            if img_dir.exists() and lbl_dir.exists():
                shutil.copytree(img_dir, out_img_dir, dirs_exist_ok=True)
                shutil.copytree(lbl_dir, out_lbl_dir, dirs_exist_ok=True)

            # Get list of image files
            img_files = list(img_dir.glob("*.jpg"))
            if not img_files:
                self.log_write(f"{subset} 文件夹无图片，跳过")  # No images in folder, skipping
                continue

            # Initialize augmentation counters and parameters
            aug_counters = {}  # Track augmentation count per image
            idx = 0
            count = 0
            target_aug = self.target_aug
            
            # Calculate actual target based on multiplier setting
            if self.is_multiple:
                target_aug = int(len(img_files) * target_aug)

            # Main augmentation loop
            while count < target_aug:
                # Cycle through images
                img_path = img_files[idx % len(img_files)]
                base_name = img_path.stem

                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    self.log_write(f"{subset} - 读取图片失败: {img_path.name}")  # Failed to read image
                    idx += 1
                    continue

                # Load corresponding label file
                label_path = lbl_dir / f"{base_name}.txt"
                if not label_path.exists():
                    self.log_write(f"{subset} - 标签文件不存在: {base_name}.txt")  # Label file not found
                    idx += 1
                    continue

                # Read label content
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.read().strip().splitlines()

                # Parse YOLO format bounding boxes
                height, width = image.shape[:2]
                bbs = []
                original_areas = {}  # Store original bbox areas for comparison
                
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    # Parse YOLO format: class_id x_center y_center width height (normalized)
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    # Convert to absolute coordinates
                    x1 = x_center - w/2
                    y1 = y_center - h/2
                    x2 = x_center + w/2
                    y2 = y_center + h/2
                    
                    # Create bounding box object
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=cls_id)
                    bbs.append(bbox)
                    
                    # Store original area for later filtering
                    original_areas[i] = w * h

                bbs_on_image = BoundingBoxesOnImage(bbs, shape=image.shape)

                # Apply augmentation
                try:
                    aug_img, aug_bbs = self.augmenter(image=image, bounding_boxes=bbs_on_image)
                except Exception as e:
                    self.log_write(f"{subset} - 增强失败: {e}")  # Augmentation failed
                    idx += 1
                    continue

                # Clip bounding boxes to image boundaries
                aug_bbs = aug_bbs.clip_out_of_image()
                
                # Filter invalid bounding boxes (area > 0 and meets minimum area requirement)
                valid_bbs = []
                for i, bb in enumerate(aug_bbs.bounding_boxes):
                    if bb.width > 0 and bb.height > 0:
                        # Calculate current bbox area
                        current_area = bb.width * bb.height
                        
                        # Get original bbox area
                        original_area = original_areas.get(i, 0)
                        
                        # Check if current area meets minimum ratio requirement
                        if original_area > 0 and current_area / original_area >= self.min_area_ratio:
                            valid_bbs.append(bb)
                        else:
                            # Log filtered small bboxes
                            self.log_write(f"{subset} - 过滤小面积边界框: 原始面积={original_area:.1f}, 当前面积={current_area:.1f}, 比例={current_area/original_area:.2f}")
                            # Filtering small bbox: original area=X, current area=Y, ratio=Z
                
                aug_bbs = BoundingBoxesOnImage(valid_bbs, shape=image.shape)
                
                # Skip if no valid bboxes remain
                if not aug_bbs.bounding_boxes:
                    self.log_write(f"{subset} - 增广后无有效bbox，跳过")  # No valid bbox after augmentation, skipping
                    idx += 1
                    continue

                # Convert back to YOLO format
                out_lines = []
                for bb in aug_bbs.bounding_boxes:
                    cls_id = int(bb.label)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, bb.x1)
                    y1 = max(0, bb.y1)
                    x2 = min(image.shape[1], bb.x2)
                    y2 = min(image.shape[0], bb.y2)
                    
                    # Calculate bbox dimensions
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    # Convert to YOLO normalized format
                    x_center = (x1 + box_w / 2) / image.shape[1]
                    y_center = (y1 + box_h / 2) / image.shape[0]
                    w = box_w / image.shape[1]
                    h = box_h / image.shape[0]
                    
                    # Skip invalid bboxes
                    if w <= 0 or h <= 0:
                        continue
                    
                    out_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                # Skip if no valid output lines
                if not out_lines:
                    self.log_write(f"{subset} - 增广后无有效bbox，跳过")  # No valid bbox after augmentation, skipping
                    idx += 1
                    continue

                # Generate unique filename for augmented image
                aug_counters.setdefault(base_name, 0)
                aug_counters[base_name] += 1
                new_name = f"{base_name}A{aug_counters[base_name]}"

                # Save augmented image and label
                cv2.imwrite(str(out_img_dir / f"{new_name}.jpg"), aug_img)
                with open(out_lbl_dir / f"{new_name}.txt", "w", encoding="utf-8") as f:
                    f.writelines(line + "\n" for line in out_lines)

                # Update progress
                count += 1
                self.log_write(f"{subset}: 增广 {count}/{target_aug} - {new_name}")  # Augmentation progress
                idx += 1

        # Completion message
        self.log_write("增广完成！")  # Augmentation completed!
        messagebox.showinfo("完成", f"数据增广处理完成，保存至：{out_dir}")  # Processing completed, saved to: {out_dir}

    def log_write(self, msg):
        """Write message to log text area and update display"""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.update()

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAugApp(root)
    root.mainloop()