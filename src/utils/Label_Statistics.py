"""
YOLO Label Statistics Tool

This script provides a graphical user interface (GUI) application using PyQt5
to analyze and visualize the class distribution within YOLO-formatted datasets.
It supports multiple datasets (train, val, test) and can generate a combined
report, helping users understand their data balance.
"""

import os
import sys
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
                             QHeaderView, QProgressBar, QMessageBox, QTabWidget, QGroupBox, 
                             QGridLayout, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

class YOLOStatsApp(QMainWindow):
    """
    Main application window for the YOLO dataset statistics tool.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Annotation Statistics Tool")
        self.setGeometry(100, 100, 1000, 700)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create setup tab
        self.setup_tab = QWidget()
        self.tab_widget.addTab(self.setup_tab, "Folder Setup")
        self.setup_tab_layout = QVBoxLayout(self.setup_tab)
        
        # Folder selection area
        self.folder_group = QGroupBox("Select Dataset Folders")
        self.folder_layout = QGridLayout(self.folder_group)
        
        # Train folder
        self.train_label = QLabel("Train Folder:")
        self.train_path_label = QLabel("Not selected")
        self.train_path_label.setStyleSheet("border: 1px solid gray; padding: 5px; background-color: #f0f0f0;")
        self.train_path_label.setMinimumHeight(30)
        self.train_button = QPushButton("Browse...")
        self.train_button.setFixedWidth(80)
        self.train_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.train_button.clicked.connect(lambda: self.browse_folder("train"))
        
        # Validation folder
        self.val_label = QLabel("Validation Folder:")
        self.val_path_label = QLabel("Not selected")
        self.val_path_label.setStyleSheet("border: 1px solid gray; padding: 5px; background-color: #f0f0f0;")
        self.val_path_label.setMinimumHeight(30)
        self.val_button = QPushButton("Browse...")
        self.val_button.setFixedWidth(80)
        self.val_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.val_button.clicked.connect(lambda: self.browse_folder("val"))
        
        # Test folder
        self.test_label = QLabel("Test Folder:")
        self.test_path_label = QLabel("Not selected")
        self.test_path_label.setStyleSheet("border: 1px solid gray; padding: 5px; background-color: #f0f0f0;")
        self.test_path_label.setMinimumHeight(30)
        self.test_button = QPushButton("Browse...")
        self.test_button.setFixedWidth(80)
        self.test_button.setStyleSheet("background-color: #FF9800; color: white;")
        self.test_button.clicked.connect(lambda: self.browse_folder("test"))
        
        # Add widgets to layout
        self.folder_layout.addWidget(self.train_label, 0, 0)
        self.folder_layout.addWidget(self.train_path_label, 0, 1)
        self.folder_layout.addWidget(self.train_button, 0, 2)
        
        self.folder_layout.addWidget(self.val_label, 1, 0)
        self.folder_layout.addWidget(self.val_path_label, 1, 1)
        self.folder_layout.addWidget(self.val_button, 1, 2)
        
        self.folder_layout.addWidget(self.test_label, 2, 0)
        self.folder_layout.addWidget(self.test_path_label, 2, 1)
        self.folder_layout.addWidget(self.test_button, 2, 2)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 10px;
            }
        """)
        
        # Stats button
        self.stats_button = QPushButton("Start Analysis")
        self.stats_button.setFixedHeight(40)
        self.stats_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; 
                color: white; 
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #E1BEE7;
            }
        """)
        self.stats_button.clicked.connect(self.calculate_stats)
        self.stats_button.setEnabled(False)
        
        # Add folder group and button to setup tab layout
        self.setup_tab_layout.addWidget(self.folder_group)
        self.setup_tab_layout.addWidget(self.progress_bar)
        self.setup_tab_layout.addWidget(self.stats_button)
        
        # Create result tab
        self.result_tab = QWidget()
        self.tab_widget.addTab(self.result_tab, "Statistics Results")
        self.result_layout = QVBoxLayout(self.result_tab)
        
        # Dataset selector
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItem("All Datasets")
        self.dataset_selector.addItem("Train Set")
        self.dataset_selector.addItem("Validation Set")
        self.dataset_selector.addItem("Test Set")
        self.dataset_selector.currentIndexChanged.connect(self.update_result_display)
        self.dataset_selector.setEnabled(False)
        self.dataset_selector.setStyleSheet("""
            QComboBox {
                padding: 5px;
                font-size: 12px;
            }
        """)
        
        # Result table
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["Class ID", "Class Name", "Count", "Percentage"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSortingEnabled(True)
        self.result_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                font-size: 11pt;
            }
            QHeaderView::section {
                background-color: #607D8B;
                color: white;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
        # Export button
        self.export_button = QPushButton("Export CSV")
        self.export_button.setFixedWidth(100)
        self.export_button.setStyleSheet("background-color: #F44336; color: white;")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        
        # Add widgets to result layout
        self.result_layout.addWidget(QLabel("Select Dataset:"))
        self.result_layout.addWidget(self.dataset_selector)
        self.result_layout.addWidget(self.result_table, 1)
        self.result_layout.addWidget(self.export_button, 0, Qt.AlignRight)
        
        # State variables to store folder paths and stats
        self.folders = {
            "train": {"path": "", "label": self.train_path_label, "stats": []},
            "val": {"path": "", "label": self.val_path_label, "stats": []},
            "test": {"path": "", "label": self.test_path_label, "stats": []}
        }
        self.class_names = {}
        self.all_stats = []
        self.total_objects = 0

    def browse_folder(self, dataset_type):
        """Opens a file dialog to select a dataset folder."""
        folder = QFileDialog.getExistingDirectory(self, f"Select {dataset_type.upper()} Annotation Folder")
        if folder:
            self.folders[dataset_type]["path"] = folder
            self.folders[dataset_type]["label"].setText(folder)
            
            # Enable the stats button if any folder is selected
            any_selected = any(self.folders[typ]["path"] for typ in ["train", "val", "test"])
            self.stats_button.setEnabled(any_selected)
            
            # Try to load class names if not loaded yet
            if not self.class_names:
                self.load_class_names(folder)

    def load_class_names(self, folder):
        """Attempts to load class names from classes.txt or infers from annotations."""
        self.class_names = {}
        classes_file = os.path.join(folder, "classes.txt")
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r') as f:
                    for idx, line in enumerate(f):
                        class_name = line.strip()
                        if class_name:
                            self.class_names[str(idx)] = class_name
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not read classes.txt: {str(e)}")
        else:
            # If classes.txt is not found, infer from annotation files
            for dataset_type in ["train", "val", "test"]:
                folder_path = self.folders[dataset_type]["path"]
                if folder_path:
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.txt') and filename != 'classes.txt':
                            try:
                                with open(os.path.join(folder_path, filename), 'r') as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        if parts:
                                            class_id = parts[0]
                                            if class_id not in self.class_names:
                                                self.class_names[class_id] = f"Class {class_id}"
                            except:
                                pass

    def calculate_stats(self):
        """Calculates annotation statistics for selected folders."""
        # Reset state and show progress bar
        self.result_table.setRowCount(0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        
        # Calculate total files for progress bar
        total_files = 0
        for dataset_type in ["train", "val", "test"]:
            folder_path = self.folders[dataset_type]["path"]
            if folder_path and os.path.isdir(folder_path):
                txt_files = [f for f in os.listdir(folder_path) 
                            if f.endswith('.txt') and f != 'classes.txt']
                total_files += len(txt_files)
        
        if total_files == 0:
            QMessageBox.warning(self, "No Annotation Files", "No annotation files (.txt) found in selected folders.")
            self.progress_bar.setVisible(False)
            return
        
        # Initialize stats
        self.all_stats = []
        self.total_objects = 0
        processed_files = 0
        
        # Iterate through each dataset folder
        for dataset_type in ["train", "val", "test"]:
            folder_path = self.folders[dataset_type]["path"]
            if not folder_path or not os.path.isdir(folder_path):
                self.folders[dataset_type]["stats"] = []
                continue
                
            # Get all text files
            txt_files = [f for f in os.listdir(folder_path) 
                        if f.endswith('.txt') and f != 'classes.txt']
            
            # Count class occurrences
            class_counts = defaultdict(int)
            dataset_objects = 0
            
            for filename in txt_files:
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:  # At least class ID
                                class_id = parts[0]
                                class_counts[class_id] += 1
                                dataset_objects += 1
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
                
                # Update progress bar
                processed_files += 1
                progress = int((processed_files) / total_files * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()
            
            # Store dataset statistics
            dataset_stats = []
            for class_id, count in sorted(class_counts.items(), key=lambda x: int(x[0])):
                percentage = (count / dataset_objects * 100) if dataset_objects > 0 else 0
                class_name = self.class_names.get(class_id, f"Class {class_id}")
                dataset_stats.append({
                    "id": class_id,
                    "name": class_name,
                    "count": count,
                    "percentage": percentage,
                    "dataset": dataset_type
                })
            
            self.folders[dataset_type]["stats"] = dataset_stats
            self.total_objects += dataset_objects
        
        # Create combined stats for all datasets
        self.create_all_stats()
        
        # Display results
        self.dataset_selector.setEnabled(True)
        self.update_result_display()
        self.progress_bar.setVisible(False)
        self.export_button.setEnabled(True)
        self.tab_widget.setCurrentIndex(1)  # Switch to the results tab

    def create_all_stats(self):
        """Creates a combined statistics report from all datasets."""
        all_class_counts = defaultdict(int)
        
        # Aggregate counts from all datasets
        for dataset_type in ["train", "val", "test"]:
            for stat in self.folders[dataset_type]["stats"]:
                class_id = stat["id"]
                all_class_counts[class_id] += stat["count"]
        
        # Populate the combined stats list
        self.all_stats = []
        for class_id, count in sorted(all_class_counts.items(), key=lambda x: int(x[0])):
            percentage = (count / self.total_objects * 100) if self.total_objects > 0 else 0
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            self.all_stats.append({
                "id": class_id,
                "name": class_name,
                "count": count,
                "percentage": percentage,
                "dataset": "all"
            })

    def update_result_display(self):
        """Updates the table view based on the selected dataset."""
        selected_index = self.dataset_selector.currentIndex()
        
        if selected_index == 0:  # All datasets
            stats_data = self.all_stats
            dataset_name = "All Datasets"
        elif selected_index == 1:  # Train set
            stats_data = self.folders["train"]["stats"]
            dataset_name = "Train Set"
        elif selected_index == 2:  # Validation set
            stats_data = self.folders["val"]["stats"]
            dataset_name = "Validation Set"
        elif selected_index == 3:  # Test set
            stats_data = self.folders["test"]["stats"]
            dataset_name = "Test Set"
        else:
            return
        
        if not stats_data:
            self.result_table.setRowCount(0)
            return
        
        # Set headers
        self.result_table.setHorizontalHeaderLabels(["Class ID", "Class Name", "Count", "Percentage"])
        
        # Display results in the table
        self.result_table.setRowCount(len(stats_data))
        
        for row, data in enumerate(stats_data):
            # Class ID
            id_item = QTableWidgetItem(data["id"])
            id_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(row, 0, id_item)
            
            # Class Name
            name_item = QTableWidgetItem(data["name"])
            self.result_table.setItem(row, 1, name_item)
            
            # Count
            count_item = QTableWidgetItem(str(data["count"]))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(row, 2, count_item)
            
            # Percentage
            percentage = f"{data['percentage']:.2f}%"
            percent_item = QTableWidgetItem(percentage)
            percent_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(row, 3, percent_item)
        
        # Add a total row
        total_count = sum(item["count"] for item in stats_data)
        total_percentage = "100.00%" if total_count > 0 else "0.00%"
        
        self.result_table.insertRow(len(stats_data))
        
        # Create items for the total row
        total_id_item = QTableWidgetItem("Total")
        total_id_item.setTextAlignment(Qt.AlignCenter)
        self.result_table.setItem(len(stats_data), 0, total_id_item)
        
        total_name_item = QTableWidgetItem(f"Total {dataset_name}")
        self.result_table.setItem(len(stats_data), 1, total_name_item)
        
        total_count_item = QTableWidgetItem(str(total_count))
        total_count_item.setTextAlignment(Qt.AlignCenter)
        self.result_table.setItem(len(stats_data), 2, total_count_item)
        
        total_percent_item = QTableWidgetItem(total_percentage)
        total_percent_item.setTextAlignment(Qt.AlignCenter)
        self.result_table.setItem(len(stats_data), 3, total_percent_item)
        
        # Style the total row
        for col in range(4):
            item = self.result_table.item(len(stats_data), col)
            item.setBackground(QColor(200, 200, 200))  # Light gray background
            font = QFont()
            font.setBold(True)
            item.setFont(font)

    def export_csv(self):
        """Exports the statistics data to a CSV file."""
        if not self.all_stats and not any(self.folders[typ]["stats"] for typ in ["train", "val", "test"]):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv)")
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("Dataset,Class ID,Class Name,Count,Percentage\n")
                
                # Write all datasets stats
                for data in self.all_stats:
                    f.write(f"all,{data['id']},{data['name']},{data['count']},{data['percentage']:.2f}%\n")
                
                # Write train set stats
                for data in self.folders["train"]["stats"]:
                    f.write(f"train,{data['id']},{data['name']},{data['count']},{data['percentage']:.2f}%\n")
                
                # Write validation set stats
                for data in self.folders["val"]["stats"]:
                    f.write(f"val,{data['id']},{data['name']},{data['count']},{data['percentage']:.2f}%\n")
                
                # Write test set stats
                for data in self.folders["test"]["stats"]:
                    f.write(f"test,{data['id']},{data['name']},{data['count']},{data['percentage']:.2f}%\n")
                
            QMessageBox.information(self, "Export Successful", f"Data successfully exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use a modern style
    window = YOLOStatsApp()
    window.show()
    sys.exit(app.exec_())