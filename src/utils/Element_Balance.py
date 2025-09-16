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
        
        # Initialize path variables
        self.source_path = StringVar()
        self.output_path = StringVar()
        self.element_bad_path = StringVar()   # Defective element library
        self.element_health_path = StringVar()  # Unripe fruit element library
        self.element_leaf_path = StringVar()   # Leaf element library
        self.element_ripe_path = StringVar()   # Ripe fruit element library
        
        # Initialize coefficient variables
        self.k1 = DoubleVar(value=0.5)  # Fruit quantity coefficient
        self.k2 = DoubleVar(value=0.1)  # Fruit size coefficient (default 0.1)
        self.k3 = DoubleVar(value=0.5)  # Occlusion strategy scaling coefficient
        
        # Strategy selection
        self.strategy_var = StringVar(value="Addition Strategy")
        self.mix_ratio = DoubleVar(value=0.5)  # Mixed strategy ratio
        
        # Set default paths
        self.default_paths = {
            "source_path": "C:/Users/zhj/Desktop/balance/mix/starfruit",
            "output_path": "C:/Users/zhj/Desktop/balance/mix/Aug_balance",
            "element_bad_path": "C:/Users/zhj/Desktop/balance/mix/bad",
            "element_health_path": "C:/Users/zhj/Desktop/balance/mix/health",
            "element_leaf_path": "C:/Users/zhj/Desktop/balance/mix/leaf",
            "element_ripe_path": "C:/Users/zhj/Desktop/balance/mix/ripe"
        }
        
        # Element library counts
        self.bad_count = 0
        self.health_count = 0
        self.leaf_count = 0
        self.ripe_count = 0
        
        # Label statistics
        self.class_counts = {0: 0, 1: 0, 2: 0}  # 0:Defective, 1:Unripe fruit, 2:Ripe fruit
        
        # Progress variables
        self.progress_var = DoubleVar(value=0.0)
        self.status_var = StringVar(value="Ready")
        
        # Create UI components
        self.create_widgets()
        
        # Load saved configuration
        #self.load_config()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)


    def create_widgets(self):
        """Create graphical user interface components"""
        # Strategy selection frame
        frame_strategy = ttk.LabelFrame(self.root, text="0. Select Augmentation Strategy", padding=10)
        frame_strategy.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        strategies = ["Addition Strategy", "Occlusion Strategy", "Mixed Mode"]
        OptionMenu(frame_strategy, self.strategy_var, *strategies, command=self.update_strategy_ui).grid(row=0, column=0, padx=5)
        
        # Mixed strategy ratio slider (initially hidden)
        self.mix_ratio_frame = ttk.Frame(frame_strategy)
        self.mix_ratio_frame.grid(row=0, column=1, padx=5, sticky="w")
        Label(self.mix_ratio_frame, text="Addition Ratio:").grid(row=0, column=0)
        self.mix_ratio_slider = Scale(self.mix_ratio_frame, from_=0.0, to=1.0, resolution=0.1, 
                                     orient="horizontal", variable=self.mix_ratio, 
                                     showvalue=True, length=150)
        self.mix_ratio_slider.grid(row=0, column=1, padx=5)
        self.mix_ratio_frame.grid_remove()  # Initially hidden
        
        # Occlusion scaling coefficient
        ttk.Label(frame_strategy, text="Occlusion Scale Factor k3:").grid(row=0, column=2, padx=(20, 0))
        ttk.Entry(frame_strategy, textvariable=self.k3, width=8).grid(row=0, column=3, padx=5)
        
        # Original data selection frame
        frame_source = ttk.LabelFrame(self.root, text="1. Select Source Data Folder", padding=10)
        frame_source.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_source, text="Path:").grid(row=0, column=0)
        Entry(frame_source, textvariable=self.source_path, width=50).grid(row=0, column=1)
        Button(frame_source, text="Browse...", command=self.select_source).grid(row=0, column=2)
        Button(frame_source, text="Count Labels", command=self.count_labels).grid(row=0, column=3, padx=5)

        # Label statistics frame
        frame_stats = ttk.LabelFrame(self.root, text="Label Statistics", padding=10)
        frame_stats.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        self.stats_label = Label(frame_stats, text="Defective: 0 | Unripe: 0 | Ripe: 0")
        self.stats_label.grid(row=0, column=0, columnspan=4)
        
        # Progress bar and status
        frame_progress = ttk.Frame(self.root)
        frame_progress.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        self.progress_bar = ttk.Progressbar(frame_progress, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        self.status_label = Label(frame_progress, textvariable=self.status_var)
        self.status_label.grid(row=0, column=1)

        # Output directory selection frame
        frame_output = ttk.LabelFrame(self.root, text="2. Select Output Folder", padding=10)
        frame_output.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_output, text="Path:").grid(row=0, column=0)
        Entry(frame_output, textvariable=self.output_path, width=50).grid(row=0, column=1)
        Button(frame_output, text="Browse...", command=self.select_output).grid(row=0, column=2)

        # Defective element library selection frame
        frame_bad = ttk.LabelFrame(self.root, text="3. Select Defective Element Library Folder", padding=10)
        frame_bad.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_bad, text="Path:").grid(row=0, column=0)
        Entry(frame_bad, textvariable=self.element_bad_path, width=50).grid(row=0, column=1)
        Button(frame_bad, text="Browse...", command=lambda: self.select_element(self.element_bad_path, "bad")).grid(row=0, column=2)
        self.bad_count_label = Label(frame_bad, text="Elements: 0")
        self.bad_count_label.grid(row=0, column=3, padx=5)

        # Unripe fruit element library selection frame
        frame_health = ttk.LabelFrame(self.root, text="4. Select Unripe Fruit Element Library Folder", padding=10)
        frame_health.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_health, text="Path:").grid(row=0, column=0)
        Entry(frame_health, textvariable=self.element_health_path, width=50).grid(row=0, column=1)
        Button(frame_health, text="Browse...", command=lambda: self.select_element(self.element_health_path, "health")).grid(row=0, column=2)
        self.health_count_label = Label(frame_health, text="Elements: 0")
        self.health_count_label.grid(row=0, column=3, padx=5)
        
        # Ripe fruit element library selection frame
        frame_ripe = ttk.LabelFrame(self.root, text="5. Select Ripe Fruit Element Library Folder", padding=10)
        frame_ripe.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_ripe, text="Path:").grid(row=0, column=0)
        Entry(frame_ripe, textvariable=self.element_ripe_path, width=50).grid(row=0, column=1)
        Button(frame_ripe, text="Browse...", command=lambda: self.select_element(self.element_ripe_path, "ripe")).grid(row=0, column=2)
        self.ripe_count_label = Label(frame_ripe, text="Elements: 0")
        self.ripe_count_label.grid(row=0, column=3, padx=5)
        
        # Coefficient settings frame
        frame_coeff = ttk.LabelFrame(self.root, text="6. Set Coefficients", padding=10)
        frame_coeff.grid(row=8, column=0, padx=10, pady=5, sticky="ew")
        
        Label(frame_coeff, text="Quantity Coefficient k1:").grid(row=0, column=0)
        Entry(frame_coeff, textvariable=self.k1, width=8).grid(row=0, column=1, padx=5)
        
        Label(frame_coeff, text="Size Coefficient k2:").grid(row=0, column=2)
        Entry(frame_coeff, textvariable=self.k2, width=8).grid(row=0, column=3, padx=5)

        # Process button
        process_btn = ttk.Button(self.root, text="Start Processing", command=self.process_balance)
        process_btn.grid(row=9, column=0, pady=5)
        
        # Balance button
        balance_btn = ttk.Button(self.root, text="Start Balancing", command=self.process_balance)
        balance_btn.grid(row=10, column=0, pady=10)
    
    def update_strategy_ui(self, *args):
        """Update UI based on selected strategy"""
        if self.strategy_var.get() == "Mixed Mode":
            self.mix_ratio_frame.grid()
        else:
            self.mix_ratio_frame.grid_remove()

    def update_status(self, message, progress=None):
        """Update status information"""
        self.status_var.set(message)
        if progress is not None:
            self.progress_var.set(progress)
        self.root.update_idletasks()

    def select_source(self):
        """Select source data folder"""
        path = filedialog.askdirectory()
        if path:
            self.source_path.set(path)

    def select_output(self):
        """Select output folder"""
        path = filedialog.askdirectory()
        if path:
            self.output_path.set(path)

    def select_element(self, path_var, element_type):
        """Select element library folder and update count"""
        path = filedialog.askdirectory()
        if path:
            path_var.set(path)
            self.update_element_count(path, element_type)

    def update_element_count(self, path, element_type):
        """Update element library count"""
        if not path or not os.path.exists(path):
            return
            
        elements = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        count = len(elements)
        
        if element_type == "bad":
            self.bad_count = count
            if hasattr(self, 'bad_count_label'):  # Ensure label is created
                self.bad_count_label.config(text=f"Elements: {count}")
        elif element_type == "health":
            self.health_count = count
            if hasattr(self, 'health_count_label'):  # Ensure label is created
                self.health_count_label.config(text=f"Elements: {count}")
        elif element_type == "leaf":
            self.leaf_count = count
            if hasattr(self, 'leaf_count_label'):  # Ensure label is created
                self.leaf_count_label.config(text=f"Elements: {count}")
        elif element_type == "ripe":
            self.ripe_count = count
            if hasattr(self, 'ripe_count_label'):  # Ensure label is created
                self.ripe_count_label.config(text=f"Elements: {count}")

    def count_labels(self):
        """Count labels in the source dataset"""
        source = self.source_path.get()
        if not source:
            messagebox.showerror("Error", "Please select source data folder first")
            return
            
        labels_dir = os.path.join(source, "labels")
        
        # Check if labels folder exists
        if not os.path.exists(labels_dir):
            messagebox.showerror("Error", f"Labels folder does not exist: {labels_dir}")
            return
        
        # Reset statistics
        self.class_counts = {0: 0, 1: 0, 2: 0}  # 0:Defective, 1:Unripe fruit, 2:Ripe fruit
        
        # Get list of label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        
        if not label_files:
            messagebox.showerror("Error", f"No label files found in labels folder: {labels_dir}")
            return
        
        print(f"Found {len(label_files)} label files")
        
        # Iterate through all label files
        valid_files = 0
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                    if not lines:
                        print(f"Warning: Empty label file - {label_file}")
                        continue
                    
                    valid_files += 1
                    for line in lines:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        parts = line.split()
                        if len(parts) < 5:
                            print(f"Warning: Invalid line format - {label_file}: {line}")
                            continue
                            
                        try:
                            # Try to parse class and coordinates
                            cls = int(float(parts[0]))
                            # Only count classes 0,1,2
                            if cls in [0, 1, 2]:
                                self.class_counts[cls] += 1
                            else:
                                print(f"Warning: Invalid class {cls} - {label_file}: {line}")
                        except (ValueError, IndexError) as e:
                            print(f"Parse error - {label_file}: {line} - {str(e)}")
            except Exception as e:
                print(f"File read error - {label_file}: {str(e)}")
        
        # If no valid label files found
        if valid_files == 0:
            messagebox.showerror("Error", "No valid label files found")
            return
        
        # Update display
        stats_text = f"Defective: {self.class_counts[0]} | Unripe: {self.class_counts[1]} | Ripe: {self.class_counts[2]}"
        self.stats_label.config(text=stats_text)
        
        # Show detailed results
        messagebox.showinfo("Label Statistics", 
            f"Label statistics completed:\n{stats_text}\n\n"
            f"Files scanned: {len(label_files)}\n"
            f"Valid files: {valid_files}\n"
            f"Defective labels: {self.class_counts[0]}\n"
            f"Unripe fruit labels: {self.class_counts[1]}\n"
            f"Ripe fruit labels: {self.class_counts[2]}")

    def validate_source(self):
        """Validate source folder structure"""
        source = self.source_path.get()
        required_folders = ["images", "labels"]
        required_files = ["classes.txt"]
        
        # Check required folders
        for folder in required_folders:
            if not os.path.exists(os.path.join(source, folder)):
                return f"Source folder missing required subfolder: {folder}"
                
        # Check required files
        for file in required_files:
            if not os.path.exists(os.path.join(source, file)):
                return f"Source folder missing required file: {file}"
                
        return None

    def create_output_structure(self):
        """Create output folder structure"""
        output = self.output_path.get()
        os.makedirs(os.path.join(output, "images"), exist_ok=True)
        os.makedirs(os.path.join(output, "labels"), exist_ok=True)

    def load_classes(self, classes_path):
        """Load classes file"""
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def parse_yolo_label(self, label_path, img_width, img_height):
        """Parse YOLO format label"""
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
                    # Convert to pixel coordinates
                    xc *= img_width
                    yc *= img_height
                    w *= img_width
                    h *= img_height
                    boxes.append((int(cls), xc, yc, w, h))
                except ValueError:
                    continue
        return boxes

    def get_random_element(self, element_folder):
        """Randomly get an element from library"""
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
        """Calculate area metrics"""
        img_area = img_width * img_height
        boxes_area = 0.0
        boxes_count = len(boxes)
        
        if boxes_count > 0:
            for _, _, _, w, h in boxes:
                boxes_area += w * h
            avg_area = boxes_area / boxes_count
        else:
            avg_area = (img_width * img_height) / 100  # If no boxes, use 1% of image area as average
        
        return img_area, boxes_area, avg_area

    def resize_element(self, element, target_area, k):
        """Resize element proportionally - based on target area"""
        # Get element original area
        original_width, original_height = element.size
        original_area = original_width * original_height
        
        # Calculate target area range
        min_target_area = k * target_area
        max_target_area = target_area
        
        # Randomly select target area
        target_area = random.uniform(min_target_area, max_target_area)
        
        # Calculate scale factor
        if original_area > 0:
            scale_factor = math.sqrt(target_area / original_area)
        else:
            scale_factor = 1.0  # If original area is 0, don't scale
            
        # Apply scaling
        new_width = max(10, int(original_width * scale_factor))
        new_height = max(10, int(original_height * scale_factor))
        
        return element.resize((new_width, new_height), Image.LANCZOS)

    def place_element(self, img, element, existing_boxes, avoid_overlap=True):
        """Place element image at specified position"""
        img_width, img_height = img.size
        element_width, element_height = element.size
        
        # If element is too big to place
        if element_width > img_width or element_height > img_height:
            return img, None
            
        # Create a copy of the image for drawing
        processed_img = img.copy()
        
        # Randomly place the element image
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random position
            x = random.randint(0, img_width - element_width)
            y = random.randint(0, img_height - element_height)
            
            # Check if overlaps with any existing boxes
            overlap = False
            if avoid_overlap:
                for box in existing_boxes:
                    cls, bx, by, bw, bh = box
                    # Calculate box boundaries
                    box_rect = (bx - bw/2, by - bh/2, bx + bw/2, by + bh/2)
                    element_rect = (x, y, x + element_width, y + element_height)
                    
                    # Calculate intersection
                    x_overlap = max(0, min(box_rect[2], element_rect[2]) - max(box_rect[0], element_rect[0]))
                    y_overlap = max(0, min(box_rect[3], element_rect[3]) - max(box_rect[1], element_rect[1]))
                    
                    if x_overlap > 0 and y_overlap > 0:
                        overlap = True
                        break
            
            if not overlap:
                # Paste the element image
                processed_img.paste(element, (int(x), int(y)), element)
                return processed_img, (x, y, element_width, element_height)
                
        return processed_img, None  # No suitable position found

    def calculate_overlap(self, rect1, rect2):
        """Calculate overlap percentage between two rectangles"""
        # rect: (x1, y1, x2, y2)
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
        overlap_area = x_overlap * y_overlap
        
        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        return overlap_area / area1 if area1 > 0 else 0

    def occlusion_strategy(self, img, boxes, element_folder, element_class, k3, update_label=True):
        """
        Generic occlusion strategy
        update_label: Whether to update label file
        """
        processed_img = img.copy()
        img_width, img_height = img.size
        new_boxes = [] if update_label else None
        existing_boxes = boxes.copy()
        
        # If no bounding boxes, return directly
        if not boxes:
            return processed_img, new_boxes
            
        # Randomly select number of boxes to occlude (between 1 and total boxes)
        k = random.randint(1, len(boxes))
        
        # Randomly select k bounding boxes
        selected_boxes_idx = random.sample(range(len(boxes)), k)
        
        for idx in selected_boxes_idx:
            cls, cx, cy, w, h = boxes[idx]
            
            # 1. Get random element
            element = self.get_random_element(element_folder)
            if element is None:
                continue
                
            # 2. Scale element image - use current box area as reference
            box_area = w * h
            element = self.resize_element(element, box_area, k3)
            ew, eh = element.size
            
            # 3. Define bounding box boundaries
            box_left = cx - w/2
            box_top = cy - h/2
            box_right = cx + w/2
            box_bottom = cy + h/2
            
            # 4. Attempt to place element (ensuring partial overlap with target box)
            max_attempts = 100
            placed = False
            
            for _ in range(max_attempts):
                # Generate random position (ensuring partial overlap with target box)
                x = random.randint(int(box_left - ew/2), int(box_right))
                y = random.randint(int(box_top - eh/2), int(box_bottom))
                
                # Ensure within image boundaries
                x = max(0, min(img_width - ew, x))
                y = max(0, min(img_height - eh, y))
                
                # Element rectangle
                element_rect = (x, y, x + ew, y + eh)
                
                # Calculate overlap ratio with current bounding box
                box_rect = (box_left, box_top, box_right, box_bottom)
                overlap_ratio = self.calculate_overlap(element_rect, box_rect)
                
                # Check if overlap ratio is between 10%~50%
                if 0.1 <= overlap_ratio <= 0.5:
                    # Check overlap with other boxes
                    overlap_with_others = False
                    for i, box in enumerate(existing_boxes):
                        if i == idx:
                            continue  # Skip current occluded box
                        
                        _, bx, by, bw, bh = box
                        other_box_rect = (bx - bw/2, by - bh/2, bx + bw/2, by + bh/2)
                        if self.calculate_overlap(element_rect, other_box_rect) > 0:
                            overlap_with_others = True
                            break
                    
                    if not overlap_with_others:
                        # Place element
                        processed_img.paste(element, (int(x), int(y)), element)
                        
                        # Update label
                        if update_label:
                            # Convert to YOLO format
                            x_center = (x + ew/2) / img_width
                            y_center = (y + eh/2) / img_height
                            norm_w = ew / img_width
                            norm_h = eh / img_height
                            
                            # Add to new boxes list
                            new_boxes.append((element_class, x_center, y_center, norm_w, norm_h))
                            
                            # Add to existing boxes list
                            existing_boxes.append((element_class, x + ew/2, y + eh/2, ew, eh))
                        placed = True
                        break
            
            if not placed:
                # If unable to place, try placing near the bounding box
                try:
                    # Random placement near bounding box
                    x = random.randint(int(box_left - ew), int(box_right))
                    y = random.randint(int(box_top - eh), int(box_bottom))
                    
                    # Ensure within image boundaries
                    x = max(0, min(img_width - ew, x))
                    y = max(0, min(img_height - eh, y))
                    
                    # Place element
                    processed_img.paste(element, (int(x), int(y)), element)
                    
                    # Update label
                    if update_label:
                        # Convert to YOLO format
                        x_center = (x + ew/2) / img_width
                        y_center = (y + eh/2) / img_height
                        norm_w = ew / img_width
                        norm_h = eh / img_height
                        
                        # Add to new boxes list
                        new_boxes.append((element_class, x_center, y_center, norm_w, norm_h))
                        
                        # Add to existing boxes list
                        existing_boxes.append((element_class, x + ew/2, y + eh/2, ew, eh))
                except:
                    # If still fails, skip
                    continue
                
        return processed_img, new_boxes

    def process_image(self, img_path, label_path, bad_folder, health_folder, leaf_folder, ripe_folder, k2, k3, 
                     output_img_path, output_label_path, add_bad=0, add_health=0, add_leaf=0, add_ripe=0, strategy="Addition Strategy", mix_ratio=0.5):
        """Process single image
        add_bad, add_health, add_leaf, add_ripe: Number of each category to add
        strategy: Strategy to use
        mix_ratio: Ratio of "Addition Strategy" in mixed mode
        """
        # Load original image
        try:
            img = Image.open(img_path).convert('RGBA')
        except Exception as e:
            print(f"Cannot load image {img_path}: {e}")
            return False, 0, 0, 0, 0
            
        img_width, img_height = img.size
        
        # Parse labels
        boxes = self.parse_yolo_label(label_path, img_width, img_height)
        
        added_bad = 0
        added_health = 0
        added_leaf = 0
        added_ripe = 0
        
        # Process each fruit to be added
        processed_img = img.copy()
        new_boxes = []
        
        # Prepare existing boxes list (original boxes + newly added boxes)
        existing_boxes = boxes.copy()
        
        # Add element function
        def add_element(element_class, element_folder, count):
            nonlocal processed_img, added_bad, added_health, added_leaf, added_ripe
            added = 0
            for _ in range(count):
                element = self.get_random_element(element_folder)
                if element is None:
                    continue
                    
                # Adjust element size - based on average bounding box area
                if boxes:
                    avg_area = sum(w * h for _, _, _, w, h in boxes) / len(boxes)
                else:
                    avg_area = (img_width * img_height) / 100
                element = self.resize_element(element, avg_area, k2)
                
                # Place element
                placed_img, new_box = self.place_element(processed_img, element, existing_boxes)
                if new_box is None:
                    continue  # Unable to place
                    
                x, y, ew, eh = new_box
                processed_img = placed_img
                
                # Convert to YOLO format
                x_center = (x + ew/2) / img_width
                y_center = (y + eh/2) / img_height
                norm_w = ew / img_width
                norm_h = eh / img_height
                
                # Add to new boxes list
                new_boxes.append((element_class, x_center, y_center, norm_w, norm_h))
                
                # Update counters
                if element_class == 0:
                    added_bad += 1
                elif element_class == 1:
                    added_health += 1
                elif element_class == 2:
                    added_ripe += 1
                elif element_class == 3:  # Leaf category
                    added_leaf += 1
                
                added += 1
                
                # Add to existing boxes list (for subsequent collision detection)
                existing_boxes.append((element_class, x + ew/2, y + eh/2, ew, eh))
            return added
        
        # Add defective fruits
        if add_bad > 0:
            # Determine processing method based on strategy
            if strategy == "Addition Strategy" or (strategy == "Mixed Mode" and random.random() < mix_ratio):
                added_bad += add_element(0, bad_folder, add_bad)
            else:
                # Occlusion strategy
                _, new_boxes_occlusion = self.occlusion_strategy(
                    processed_img, boxes, leaf_folder, 3, k3, True
                )
                if new_boxes_occlusion:
                    new_boxes.extend(new_boxes_occlusion)
                    added_leaf += len(new_boxes_occlusion)
        
        # Add unripe fruits
        if add_health > 0:
            if strategy == "Addition Strategy" or (strategy == "Mixed Mode" and random.random() < mix_ratio):
                added_health += add_element(1, health_folder, add_health)
            else:
                # Occlusion strategy
                _, new_boxes_occlusion = self.occlusion_strategy(
                    processed_img, boxes, leaf_folder, 3, k3, True
                )
                if new_boxes_occlusion:
                    new_boxes.extend(new_boxes_occlusion)
                    added_leaf += len(new_boxes_occlusion)
        
        # Add ripe fruits
        if add_ripe > 0:
            if strategy == "Addition Strategy" or (strategy == "Mixed Mode" and random.random() < mix_ratio):
                added_ripe += add_element(2, ripe_folder, add_ripe)
            else:
                # Occlusion strategy
                _, new_boxes_occlusion = self.occlusion_strategy(
                    processed_img, boxes, leaf_folder, 3, k3, True
                )
                if new_boxes_occlusion:
                    new_boxes.extend(new_boxes_occlusion)
                    added_leaf += len(new_boxes_occlusion)
        
        # Add leaves
        if add_leaf > 0:
            added_leaf += add_element(3, leaf_folder, add_leaf)
        
        # Save processed image
        try:
            processed_img = processed_img.convert('RGB')
            processed_img.save(output_img_path)
        except Exception as e:
            print(f"Cannot save image {output_img_path}: {e}")
            return False, 0, 0, 0, 0

        # Save processed labels
        try:
            with open(output_label_path, 'w') as f:
                # Write original labels
                for box in boxes:
                    cls, xc, yc, w, h = box
                    f.write(f"{cls} {xc/img_width:.6f} {yc/img_height:.6f} {w/img_width:.6f} {h/img_height:.6f}\n")
                
                # Write newly added element labels
                for box in new_boxes:
                    cls, xc, yc, w, h = box
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"Cannot save label {output_label_path}: {e}")
            return False, 0, 0, 0, 0

        return True, added_bad, added_health, added_leaf, added_ripe

    def process_balance(self):
        """Main balancing function - ensures each image gets new labels and all three categories have equal counts"""
        # Validate input paths
        if not all([self.source_path.get(), self.output_path.get(), 
                self.element_bad_path.get(), self.element_health_path.get(),
                self.element_ripe_path.get()]):
            messagebox.showerror("Error", "Please select all folder paths completely")
            return

        # Validate source folder structure
        validation_msg = self.validate_source()
        if validation_msg:
            messagebox.showerror("Error", validation_msg)
            return
            
        # Ensure labels have been counted
        if sum(self.class_counts.values()) == 0:
            self.count_labels()
            if sum(self.class_counts.values()) == 0:
                messagebox.showerror("Error", "Please count labels first")
                return

        # Get category counts
        bad_count = self.class_counts[0]
        health_count = self.class_counts[1]
        ripe_count = self.class_counts[2]  # Ripe fruit count

        # Calculate target balance count (maximum of the three categories)
        target_count = max(bad_count, health_count, ripe_count)
        
        if target_count == 0:
            messagebox.showerror("Error", "No categories found needing balancing")
            return
            
        # Calculate number to add
        need_bad = max(0, target_count - bad_count)
        need_health = max(0, target_count - health_count)
        need_ripe = max(0, target_count - ripe_count)
        
        if need_bad == 0 and need_health == 0 and need_ripe == 0:
            messagebox.showinfo("No Balancing Needed", "All categories are already balanced")
            return
            
        # Get coefficient values and strategy
        k2 = self.k2.get()
        k3 = self.k3.get()
        strategy = self.strategy_var.get()
        mix_ratio = self.mix_ratio.get()
        
        # Create output directory structure
        self.create_output_structure()

        try:
            # Prepare source images
            source_images = os.path.join(self.source_path.get(), "images")
            source_labels = os.path.join(self.source_path.get(), "labels")
            
            img_files = [f for f in os.listdir(source_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_original = len(img_files)
            
            if total_original == 0:
                messagebox.showerror("Error", "No images found in source images folder")
                return
                
            # Calculate total elements to add
            total_elements = need_bad + need_health + need_ripe
            
            # Calculate average elements to add per image
            base_per_image = total_elements // total_original
            remainder = total_elements % total_original
            
            # Calculate category ratios
            if total_elements > 0:
                bad_ratio = need_bad / total_elements
                health_ratio = need_health / total_elements
                ripe_ratio = need_ripe / total_elements
            else:
                bad_ratio = health_ratio = ripe_ratio = 0
            
            # Initialize counters
            added_bad_total = 0
            added_health_total = 0
            added_leaf_total = 0
            added_ripe_total = 0
            
            # Create processing queue
            img_files_processed = []
            
            # First round: Evenly distribute elements
            for idx, img_file in enumerate(img_files, 1):
                # Update status
                progress = idx / total_original * 100
                status = (f"Processing: {idx}/{total_original} | "
                         f"Defective: {added_bad_total}/{need_bad} | "
                         f"Unripe: {added_health_total}/{need_health} | "
                         f"Ripe: {added_ripe_total}/{need_ripe}")
                self.update_status(status, progress)
                
                # Build file paths
                img_path = os.path.join(source_images, img_file)
                base_name, ext = os.path.splitext(img_file)
                label_file = base_name + ".txt"
                label_path = os.path.join(source_labels, label_file)
                output_img_path = os.path.join(self.output_path.get(), "images", img_file)
                output_label_path = os.path.join(self.output_path.get(), "labels", label_file)
                
                # Ensure output directories exist
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

                # Calculate total elements to add to this image
                total_add = base_per_image
                if idx <= remainder:
                    total_add += 1
                
                # Distribute category counts according to ratio
                add_bad = 0
                add_health = 0
                add_ripe = 0
                add_leaf = 0  # Occlusion strategy will produce leaves
                
                for i in range(total_add):
                    r = random.random()
                    if r < bad_ratio:
                        add_bad += 1
                    elif r < bad_ratio + health_ratio:
                        add_health += 1
                    else:
                        add_ripe += 1
                
                # Process image
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
                    # Update counters
                    added_bad_total += added_b
                    added_health_total += added_h
                    added_leaf_total += added_l
                    added_ripe_total += added_r
                else:
                    # If processing fails, copy original files
                    shutil.copy(img_path, output_img_path)
                    if os.path.exists(label_path):
                        shutil.copy(label_path, output_label_path)
                    print(f"Image processing failed, copied original file: {img_file}")
                    
                # Add to processed list
                img_files_processed.append(img_file)
            
            # Second round: Check if targets are met, continue processing if not
            if (added_bad_total < need_bad or 
                added_health_total < need_health or 
                added_ripe_total < need_ripe):
                
                # Randomly shuffle images
                random.shuffle(img_files)
                
                for img_file in img_files:
                    # If all targets are met, end early
                    if (added_bad_total >= need_bad and 
                        added_health_total >= need_health and 
                        added_ripe_total >= need_ripe):
                        break
                        
                    # Skip already processed images
                    if img_file in img_files_processed:
                        continue
                        
                    # Update status
                    status = (f"Supplemental Processing | "
                             f"Defective: {added_bad_total}/{need_bad} | "
                             f"Unripe: {added_health_total}/{need_health} | "
                             f"Ripe: {added_ripe_total}/{need_ripe}")
                    self.update_status(status)
                    
                    # Build file paths
                    img_path = os.path.join(source_images, img_file)
                    base_name, ext = os.path.splitext(img_file)
                    label_file = base_name + ".txt"
                    label_path = os.path.join(source_labels, label_file)
                    output_img_path = os.path.join(self.output_path.get(), "images", img_file)
                    output_label_path = os.path.join(self.output_path.get(), "labels", label_file)
                    
                    # Calculate category counts to add to this image
                    add_bad = min(3, need_bad - added_bad_total)
                    add_health = min(3, need_health - added_health_total)
                    add_ripe = min(3, need_ripe - added_ripe_total)
                    add_leaf = 0
                    
                    # Process image
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
                        # Update counters
                        added_bad_total += added_b
                        added_health_total += added_h
                        added_leaf_total += added_l
                        added_ripe_total += added_r
                        
                        # Add to processed list
                        img_files_processed.append(img_file)

            # Show results
            final_bad = bad_count + added_bad_total
            final_health = health_count + added_health_total
            final_ripe = ripe_count + added_ripe_total
            
            # Update label statistics display
            self.stats_label.config(text=f"Defective: {final_bad} | Unripe: {final_health} | Ripe: {final_ripe}")
            
            # Update in-memory statistics
            self.class_counts[0] = final_bad
            self.class_counts[1] = final_health
            self.class_counts[2] = final_ripe
            
            result_message = (
                f"Balancing completed!\n\n"
                f"Original statistics:\n"
                f"  Defective: {bad_count}\n"
                f"  Unripe: {health_count}\n"
                f"  Ripe: {ripe_count}\n\n"
                f"Added statistics:\n"
                f"  Defective: {added_bad_total}\n"
                f"  Unripe: {added_health_total}\n"
                f"  Ripe: {added_ripe_total}\n"
                f"  Leaves: {added_leaf_total}\n\n"
                f"Final statistics:\n"
                f"  Defective: {final_bad}\n"
                f"  Unripe: {final_health}\n"
                f"  Ripe: {final_ripe}\n\n"
                f"Processed images: {len(img_files_processed)}"
            )
            
            if (added_bad_total >= need_bad and 
                added_health_total >= need_health and 
                added_ripe_total >= need_ripe):
                messagebox.showinfo("Balancing Successful", result_message)
            else:
                messagebox.showwarning("Balancing Incomplete", 
                    result_message + "\n\n" +
                    f"Targets not reached:\n"
                    f"  Defective: {need_bad - added_bad_total}\n"
                    f"  Unripe: {need_health - added_health_total}\n"
                    f"  Ripe: {need_ripe - added_ripe_total}\n\n"
                    f"May need to adjust coefficients or add more elements to libraries")
            
            # Reset status
            self.update_status("Balancing completed", 100)
            self.root.title("Dataset_Balance_V1.0_zhjoy")
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error during balancing process: {str(e)}")
            self.root.title("Dataset_Balance_V1.0_zhjoy")

    # Other methods remain unchanged (load_config, save_config, on_close, etc.)

if __name__ == "__main__":
    root = Tk()
    app = DataAugmentationApp(root)
    
    # Set window size
    root.geometry("850x750")
    
    # Run main loop
    root.mainloop()
