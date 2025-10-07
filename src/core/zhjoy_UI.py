import sys
import os
import cv2
import time
import torch
import warnings
from datetime import datetime

# ==============================================================================
# aviod FutureWarning 
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
# ==============================================================================

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QTextEdit,
    QFrame, QSizePolicy
)
from PyQt5.QtGui import QFont, QPixmap, QImage, QPalette, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

# Check if ultralytics is installed and import
try:
    from ultralytics import YOLO
    print("Ultralytics library loaded successfully.")
except ImportError:
    print("Error: The 'ultralytics' library is not installed.")
    print("Please install it with 'pip install ultralytics'")
    sys.exit(1)

# A separate thread for object detection to prevent UI freezing
class DetectionThread(QThread):
    # Signals for updating the UI
    frame_ready = pyqtSignal(QImage, float, int, dict)
    log_message = pyqtSignal(str)
    fps_stats_ready = pyqtSignal(float, float, float)
    
    def __init__(self, model_path, mode, file_path=None, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.mode = mode
        self.file_path = file_path
        self.running = True
        self.paused = False
        self.yolo_model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Define class names for stat counting
        self.class_names = {0: 'Bad', 1: 'Unripe', 2: 'Ripe'}
        
        self.fps_history = []
        
        # Store results for saving
        self.processed_image = None
        self.video_writer = None
        self.video_output_path = None

    def run(self):
        self.log_message.emit("Starting to load model...")
        try:
            self.yolo_model = YOLO(self.model_path, task='detect')
            if self.model_path.endswith('.pt'):
                self.yolo_model.to(self.device)
            self.log_message.emit(f"Model loaded successfully on {self.device}.")
        except Exception as e:
            self.log_message.emit(f"Failed to load model: {e}")
            return

        if self.mode == "Image Mode":
            self.process_image()
        elif self.mode == "Video Mode":
            self.process_video()
        elif self.mode == "Camera Mode":
            self.process_camera()
    
    def process_image(self):
        try:
            image = cv2.imread(self.file_path)
            if image is None:
                self.log_message.emit("Could not load image file.")
                return

            start_time = time.time()
            results = self.yolo_model(image, verbose=False, imgsz=640) 
            inference_time = time.time() - start_time
            
            class_counts = {0: 0, 1: 0, 2: 0} 
            total_detections = 0
            if results and results[0].boxes:
                total_detections = len(results[0].boxes)
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls in class_counts:
                        class_counts[cls] += 1
            
            annotated_image = results[0].plot(conf=True)
            self.processed_image = annotated_image  # Store for saving
            
            h, w, ch = annotated_image.shape
            bytes_per_line = ch * w
            q_image = QImage(annotated_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            self.frame_ready.emit(q_image, inference_time, total_detections, class_counts)
            self.log_message.emit(f"Image inference completed in {inference_time:.4f} seconds.")
            
        except Exception as e:
            self.log_message.emit(f"An error occurred while processing the image: {e}")
        finally:
            self.running = False
    
    def _process_stream(self, cap, is_video=False):
        if not cap.isOpened():
            self.log_message.emit("Could not open video stream or camera.")
            self.running = False
            return
            
        self.fps_history = []
        
        # Setup video writer for video mode
        if is_video and self.file_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Generate temporary output path
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            video_name = os.path.splitext(os.path.basename(self.file_path))[0]
            temp_dir = os.path.join(os.path.dirname(self.file_path), "temp_detection_output")
            os.makedirs(temp_dir, exist_ok=True)
            self.video_output_path = os.path.join(temp_dir, f"{model_name}_{video_name}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_output_path, fourcc, fps, (width, height))
            
        while self.running and cap.isOpened():
            if self.paused:
                time.sleep(0.1)
                continue
            
            ret, frame = cap.read()
            if not ret:
                self.log_message.emit("Video stream ended or failed to read frame.")
                break
            
            start_time = time.time()
            results = self.yolo_model(frame, verbose=False, imgsz=640) 
            inference_time = time.time() - start_time
            
            fps = (1.0 / inference_time) if inference_time > 0 else 0
            
            self.fps_history.append(fps) 
            
            class_counts = {0: 0, 1: 0, 2: 0}
            total_detections = 0
            log_lines = []

            if results and results[0].boxes:
                total_detections = len(results[0].boxes)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                    
                    class_name = self.class_names.get(cls_id, f'Unknown({cls_id})')
                    coords = box.xyxy[0].int().tolist()
                    confidence = float(box.conf[0])
                    log_lines.append(
                        f"  - Found: {class_name} | Confidence: {confidence:.2f} | BBox: {coords}"
                    )

            if log_lines:
                current_timestamp = time.strftime("%H:%M:%S", time.localtime())
                frame_log_message = f"[{current_timestamp}] FPS: {fps:.2f} | Detections:\n" + "\n".join(log_lines)
                self.log_message.emit(frame_log_message)

            annotated_frame = results[0].plot(conf=True)
            
            # Write to video file if in video mode
            if self.video_writer is not None:
                self.video_writer.write(annotated_frame)
            
            h, w, ch = annotated_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            self.frame_ready.emit(q_image, fps, total_detections, class_counts)

        cap.release()
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.log_message.emit(f"Video processing completed. Temporary file ready for saving.")
        
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            max_fps = max(self.fps_history)
            min_fps = min(self.fps_history)
            
            self.fps_stats_ready.emit(avg_fps, max_fps, min_fps)

        self.running = False
        self.log_message.emit("Detection ended.")

    def process_video(self):
        cap = cv2.VideoCapture(self.file_path)
        self._process_stream(cap, is_video=True)

    def process_camera(self):
        cap = cv2.VideoCapture(0) 
        self._process_stream(cap, is_video=False)

    def stop(self):
        self.running = False
        self.wait()

    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Starfruit Detection and Grading System")
        self.resize(1200, 800)
        self.detection_thread = None

        self.model_path = None
        self.source_file_path = None

        self.setup_ui()
        self.apply_stylesheet()
        
        self.background_path = r"C:\Users\zhj\Desktop\Deploy_File\BAK.jpg" 
        self.set_background(self.background_path)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Top control panel
        top_frame = QFrame()
        top_frame.setObjectName("topFrame")
        top_layout = QHBoxLayout(top_frame)
        top_layout.setSpacing(15)

        options_layout = QHBoxLayout()
        
        # Model selection
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Model File:", self))
        self.model_button = QPushButton("Select .pt / .engine File")
        self.model_button.setObjectName("fileButton")
        self.model_button.clicked.connect(self.select_model_file)
        model_layout.addWidget(self.model_button)
        options_layout.addLayout(model_layout)
        
        # Mode selection
        mode_layout = QVBoxLayout()
        mode_layout.addWidget(QLabel("Detection Mode:", self))
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("modeComboBox")
        self.mode_combo.addItems(["Image Mode", "Video Mode", "Camera Mode"])
        self.mode_combo.currentIndexChanged.connect(self.update_file_selection_logic)
        mode_layout.addWidget(self.mode_combo)
        options_layout.addLayout(mode_layout)

        # File selection
        file_layout = QVBoxLayout()
        file_layout.addWidget(QLabel("Input Source:", self))
        self.file_button = QPushButton("Select File")
        self.file_button.setObjectName("fileButton")
        self.file_button.clicked.connect(self.select_source_file)
        file_layout.addWidget(self.file_button)
        options_layout.addLayout(file_layout)

        top_layout.addLayout(options_layout)
        
        # Controls
        control_layout = QVBoxLayout()
        control_layout.addWidget(QLabel("Controls:", self))
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("controlButton")
        self.start_button.clicked.connect(self.start_detection)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setObjectName("controlButton")
        self.pause_button.clicked.connect(self.pause_detection)
        self.end_button = QPushButton("Stop")
        self.end_button.setObjectName("controlButton")
        self.end_button.clicked.connect(self.end_detection)
        
        # Save button (initially hidden)
        self.save_button = QPushButton("Save Result")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setVisible(False)
        
        self.pause_button.setEnabled(False)
        self.end_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.end_button)
        button_layout.addWidget(self.save_button)
        control_layout.addLayout(button_layout)
        top_layout.addLayout(control_layout)

        main_layout.addWidget(top_frame)
        
        bottom_frame = QFrame()
        bottom_frame.setObjectName("bottomFrame")
        bottom_layout = QHBoxLayout(bottom_frame)

        display_panel = QFrame()
        display_panel.setObjectName("displayPanel")
        display_layout = QVBoxLayout(display_panel)

        self.output_display = QLabel("Please select a model and start detection")
        self.output_display.setAlignment(Qt.AlignCenter)
        self.output_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        display_layout.addWidget(self.output_display)

        stats_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: N/A")
        self.fps_label.setObjectName("fpsLabel")
        self.avg_fps_stat_label = QLabel("Avg FPS: N/A")
        self.avg_fps_stat_label.setObjectName("avgStatFPSLabel")

        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.avg_fps_stat_label)
        stats_layout.addStretch()
        
        self.total_count_label = QLabel("Total: 0")
        self.total_count_label.setObjectName("totalCountLabel")
        self.bad_count_label = QLabel("Bad: 0")
        self.bad_count_label.setObjectName("badCountLabel")
        self.unripe_count_label = QLabel("Unripe: 0")
        self.unripe_count_label.setObjectName("unripeCountLabel")
        self.ripe_count_label = QLabel("Ripe: 0")
        self.ripe_count_label.setObjectName("ripeCountLabel")

        self.max_fps_label = QLabel("Max FPS: N/A")
        self.max_fps_label.setObjectName("maxStatFPSLabel")
        self.min_fps_label = QLabel("Min FPS: N/A")
        self.min_fps_label.setObjectName("minStatFPSLabel")
        
        stats_layout.addWidget(self.total_count_label)
        stats_layout.addWidget(self.max_fps_label)
        stats_layout.addStretch()
        
        stats_layout.addWidget(self.bad_count_label)
        stats_layout.addWidget(self.min_fps_label)
        stats_layout.addStretch()

        stats_layout.addWidget(self.unripe_count_label)
        stats_layout.addStretch()

        stats_layout.addWidget(self.ripe_count_label)
        stats_layout.addStretch()

        self.set_stats_visibility(True)
        
        display_layout.addLayout(stats_layout)
        bottom_layout.addWidget(display_panel, 3)

        # Log panel setup
        log_panel = QFrame()
        log_panel.setObjectName("logPanel")
        log_layout = QVBoxLayout(log_panel)
        
        # Log header with save button
        log_header_layout = QHBoxLayout()
        log_header_layout.addWidget(QLabel("Log Output:"))
        log_header_layout.addStretch()
        self.save_log_button = QPushButton("Save Log")
        self.save_log_button.setObjectName("saveLogButton")
        self.save_log_button.clicked.connect(self.save_log)
        log_header_layout.addWidget(self.save_log_button)
        log_layout.addLayout(log_header_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        bottom_layout.addWidget(log_panel, 1)
        
        main_layout.addWidget(bottom_frame)

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #ffffff;
                font-family: "Times New Roman";
                font-size: 11pt;
            }
            QLabel {
                font-size: 12pt;
                font-weight: bold;
                color: #e0e0e0;
            }
            /* FPS 实时标签样式 */
            #fpsLabel {
                font-size: 20pt;
                font-weight: bold;
                color: #ff4d4d;
            }
            
            /* FPS 统计标签样式 */
            #avgStatFPSLabel {
                font-size: 20pt;
                font-weight: bold;
                color: #FF0000;
            }
            #maxStatFPSLabel {
                font-size: 20pt;
                font-weight: bold;
                color: #00FF00;
            }
            #minStatFPSLabel {
                font-size: 20pt;
                font-weight: bold;
                color: #00BFFF;
            }

            /* 实时计数标签样式 */
            #totalCountLabel, #badCountLabel, #unripeCountLabel, #ripeCountLabel {
                font-size: 14pt;
                font-weight: bold;
            }
            #totalCountLabel  { color: #ffffff; }
            #ripeCountLabel   { color: #4CAF50; }
            #unripeCountLabel { color: #FFC107; }
            #badCountLabel    { color: #E91E63; }

            #topFrame, #displayPanel, #logPanel {
                background-color: rgba(30, 30, 30, 0.7);
                border-radius: 10px;
                padding: 15px;
            }
            QPushButton {
                background-color: #007bff;
                border: none;
                border-radius: 8px;
                color: #ffffff;
                padding: 10px 15px;
                font-size: 12pt;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:pressed { background-color: #003d80; }
            
            QPushButton#fileButton { background-color: #6c757d; }
            QPushButton#fileButton:hover { background-color: #5a6268; }
            QPushButton#fileButton:disabled { background-color: #495057; color: #a0a0a0; }

            QPushButton#controlButton { background-color: #28a745; }
            QPushButton#controlButton:hover { background-color: #218838; }
            QPushButton#controlButton:disabled { background-color: #495057; }
            
            QPushButton#saveButton { background-color: #ffc107; }
            QPushButton#saveButton:hover { background-color: #e0a800; }
            QPushButton#saveButton:pressed { background-color: #d39e00; }
            
            QPushButton#saveLogButton { 
                background-color: #17a2b8; 
                padding: 5px 10px;
                font-size: 10pt;
            }
            QPushButton#saveLogButton:hover { background-color: #138496; }
            QPushButton#saveLogButton:pressed { background-color: #0f6674; }
            
            QComboBox {
                background-color: #444444;
                border: 1px solid #666666;
                border-radius: 5px;
                padding: 5px;
                color: #ffffff;
                font-size: 12pt;
            }
            QComboBox::drop-down { border-left: 1px solid #666666; }
            QComboBox QAbstractItemView {
                background-color: #444444;
                color: #ffffff;
                selection-background-color: #007bff;
                margin-top: 3px;
            }
            QTextEdit {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                color: #e0e0e0;
                border-radius: 5px;
                padding: 5px;
                font-size: 10pt;
                font-family: "Courier New", monospace;
            }
        """)

    def resizeEvent(self, event):
        self.set_background(self.background_path)
        super().resizeEvent(event)

    def set_background(self, image_path):
        if not os.path.exists(image_path):
            if self.palette().color(QPalette.Background) == QBrush(Qt.transparent):
                 palette = QPalette()
                 palette.setBrush(QPalette.Background, QBrush(Qt.darkGray))
                 self.setPalette(palette)
            self.log_text.append(f"Warning: Background image file not found: {image_path}. Using dark gray background.")
            return
        
        pixmap = QPixmap(image_path)
        palette = QPalette()
        scaled_pixmap = pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette.setBrush(QPalette.Background, QBrush(scaled_pixmap))
        self.setPalette(palette)

    def set_stats_visibility(self, is_realtime):
        mode = self.mode_combo.currentText()
        if mode == "Image Mode":
            self.fps_label.setVisible(False)
            self.avg_fps_stat_label.setVisible(False)
            self.max_fps_label.setVisible(False)
            self.min_fps_label.setVisible(False)
        else:
            self.fps_label.setVisible(is_realtime)
            self.avg_fps_stat_label.setVisible(not is_realtime)
            self.max_fps_label.setVisible(not is_realtime)
            self.min_fps_label.setVisible(not is_realtime)

        self.total_count_label.setVisible(is_realtime)
        self.bad_count_label.setVisible(is_realtime)
        self.unripe_count_label.setVisible(is_realtime)
        self.ripe_count_label.setVisible(is_realtime)

    def update_file_selection_logic(self):
        mode = self.mode_combo.currentText()
        if mode == "Camera Mode":
            self.file_button.setText("No File Needed")
            self.source_file_path = None
            self.file_button.setEnabled(False)
            self.log_text.append("Switched to camera mode, no file needed.")
        else:
            self.file_button.setText("Select File")
            self.file_button.setEnabled(True)

    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "AI Models (*.pt *.engine)")
        if file_path:
            self.model_button.setText(os.path.basename(file_path))
            self.model_path = file_path
            self.log_text.append(f"Selected model file: {self.model_path}")

    def select_source_file(self):
        mode = self.mode_combo.currentText()
        file_path = None
        
        if mode == "Image Mode":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.jpg *.jpeg *.png)")
        elif mode == "Video Mode":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        else:
            self.log_text.append("Camera mode selected, no file needed.")
            return
            
        if file_path:
            self.file_button.setText(os.path.basename(file_path))
            self.source_file_path = file_path
            self.log_text.append(f"Selected source file: {self.source_file_path}")

    def start_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.log_text.append("Detection is already running.")
            return

        mode = self.mode_combo.currentText()
        if self.model_path is None:
            self.log_text.append("Error: Please select a model file first.")
            return
        if mode in ["Image Mode", "Video Mode"] and (self.source_file_path is None or not os.path.exists(self.source_file_path)):
            self.log_text.append("Error: Please select a valid source file first.")
            return
        
        self.log_text.clear()

        # Hide save button when starting new detection
        self.save_button.setVisible(False)

        # UI Setup for detection mode
        self.set_stats_visibility(True)

        if mode == "Image Mode":
            source_path = self.source_file_path
        else:
            self.fps_label.setText("FPS: N/A")
            source_path = self.source_file_path

        # Reset labels
        self.total_count_label.setText("Total: 0")
        self.bad_count_label.setText("Bad: 0")
        self.unripe_count_label.setText("Unripe: 0")
        self.ripe_count_label.setText("Ripe: 0")
        self.avg_fps_stat_label.setText("Avg FPS: N/A")
        self.max_fps_label.setText("Max FPS: N/A")
        self.min_fps_label.setText("Min FPS: N/A")

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True if mode != "Image Mode" else False) 
        self.end_button.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.model_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.output_display.setText("Loading...")

        self.detection_thread = DetectionThread(self.model_path, mode, source_path)
        self.detection_thread.frame_ready.connect(self.update_ui)
        self.detection_thread.log_message.connect(self.log_text.append)
        self.detection_thread.fps_stats_ready.connect(self.display_fps_stats)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.start()

    def pause_detection(self):
        if self.detection_thread and self.detection_thread.isRunning() and not self.detection_thread.paused:
            self.detection_thread.pause()
            self.log_text.append("Detection paused.")
            self.pause_button.setText("Resume")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.resume_detection)

    def resume_detection(self):
        if self.detection_thread and self.detection_thread.isRunning() and self.detection_thread.paused:
            self.detection_thread.resume()
            self.log_text.append("Detection resumed.")
            self.pause_button.setText("Pause")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_detection)

    def end_detection(self):
        if self.detection_thread:
            self.detection_thread.stop()
            self.log_text.append("Stopping detection...")

    def on_detection_finished(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.end_button.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.model_button.setEnabled(True)
        if self.mode_combo.currentText() != "Camera Mode":
            self.file_button.setEnabled(True)
        
        # 重置暂停按钮状态
        if self.pause_button.text() == "Resume":
            self.pause_button.setText("Pause")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_detection)
        
        # Show save button for Image and Video modes
        mode = self.mode_combo.currentText()
        if mode in ["Image Mode", "Video Mode"]:
            self.save_button.setVisible(True)
            self.log_text.append("Detection completed. Click 'Save Result' to save the output.")
        
        self.log_text.append("Detection has ended.")

    def save_result(self):
        """Save detection results to user-selected folder"""
        if self.detection_thread is None:
            self.log_text.append("Error: No detection results to save.")
            return
        
        mode = self.mode_combo.currentText()
        
        # Select save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            self.log_text.append("Save operation cancelled.")
            return
        
        try:
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            
            if mode == "Image Mode":
                if self.detection_thread.processed_image is None:
                    self.log_text.append("Error: No processed image available.")
                    return
                
                image_name = os.path.splitext(os.path.basename(self.source_file_path))[0]
                output_filename = f"{model_name}_{image_name}.jpg"
                output_path = os.path.join(save_dir, output_filename)
                cv2.imwrite(output_path, self.detection_thread.processed_image)
                self.log_text.append(f"Image saved successfully to: {output_path}")
                
            elif mode == "Video Mode":
                if self.detection_thread.video_output_path is None or not os.path.exists(self.detection_thread.video_output_path):
                    self.log_text.append("Error: No processed video available.")
                    return
                
                video_name = os.path.splitext(os.path.basename(self.source_file_path))[0]
                output_filename = f"{model_name}_{video_name}.mp4"
                output_path = os.path.join(save_dir, output_filename)
                
                # Move temporary file to selected directory
                import shutil
                shutil.move(self.detection_thread.video_output_path, output_path)
                self.log_text.append(f"Video saved successfully to: {output_path}")
                
                # Clean up temporary directory if empty
                temp_dir = os.path.dirname(self.detection_thread.video_output_path)
                try:
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except:
                    pass
            
            self.save_button.setVisible(False)
            
        except Exception as e:
            self.log_text.append(f"Error saving result: {e}")

    def save_log(self):
        """Save log output to a file with timestamp as filename"""
        log_content = self.log_text.toPlainText()
        
        if not log_content.strip():
            self.log_text.append("Warning: Log is empty, nothing to save.")
            return
        
        # Select save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Log")
        if not save_dir:
            self.log_text.append("Log save operation cancelled.")
            return
        
        try:
            # Generate filename with current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{timestamp}.log"
            log_path = os.path.join(save_dir, log_filename)
            
            # Write log content to file
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            
            self.log_text.append(f"Log saved successfully to: {log_path}")
            
        except Exception as e:
            self.log_text.append(f"Error saving log: {e}")

    def display_fps_stats(self, avg_fps, max_fps, min_fps):
        """Receive FPS statistics, update UI and write into Log"""
        mode = self.mode_combo.currentText()
        if mode in ["Video Mode", "Camera Mode"]:
            # Core logic: hide real-time counts and display FPS statistics
            self.set_stats_visibility(False)
            
            # Update FPS statistic labels
            self.avg_fps_stat_label.setText(f"Avg FPS: {avg_fps:.2f}")
            self.max_fps_label.setText(f"Max FPS: {max_fps:.2f}")
            self.min_fps_label.setText(f"Min FPS: {min_fps:.2f}")
            
            # Write to Log
            stats_message = (
                f"\n--- Performance Summary for {mode} ---\n"
                f" Average FPS: {avg_fps:.2f}\n"
                f" Maximum FPS: {max_fps:.2f}\n"
                f" Minimum FPS: {min_fps:.2f}\n"
                f"--------------------------------------\n"
            )
            self.log_text.append(stats_message)

    def update_ui(self, q_image, metric, total_count, class_counts):
        """Update frame and counting info in real time"""
        
        mode = self.mode_combo.currentText()
        
        # During real-time operation, ensure displaying real-time counts
        self.set_stats_visibility(True)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.output_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.output_display.setPixmap(scaled_pixmap)
        
        # Update real-time counts
        self.total_count_label.setText(f"Total: {total_count}")
        self.bad_count_label.setText(f"Bad: {class_counts.get(0, 0)}")
        self.unripe_count_label.setText(f"Unripe: {class_counts.get(1, 0)}")
        self.ripe_count_label.setText(f"Ripe: {class_counts.get(2, 0)}")

        # Update FPS label only in Video/Camera mode
        if mode in ["Video Mode", "Camera Mode"]:
            self.fps_label.setText(f"FPS: {metric:.2f}")



if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    window = MainApp()
    window.update_file_selection_logic() 
    window.show()
    sys.exit(app.exec_())

