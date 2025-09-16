import os
import sys
from typing import Tuple, Optional
import shutil
import random
import yaml
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QDoubleSpinBox, QProgressBar
from PyQt5.QtCore import Qt
from PIL import Image
from PyQt5.QtGui import QPalette, QBrush, QPixmap

def split_and_organize_images(
    source_dir: str,
    output_dir: str,
    project_name: str = "my_project",
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    normalize_size: Optional[Tuple[int, int]] = None
):
    """
    Splits images from a source directory into train, val, and test sets,
    organizes them into a YOLO-style directory structure, and generates
    a corresponding YAML configuration file.

    Args:
        source_dir (str): Path to the directory containing all raw images.
        output_dir (str): Path to the parent directory where the new project folder will be created.
        project_name (str): Name of the new project folder.
        ratios (Tuple[float, float, float]): Ratios for train, validation, and test sets.
        seed (int): Random seed for shuffling the image list.
        normalize_size (Optional[Tuple[int, int]]): A tuple (width, height) to resize all images.
                                                    If None, images are not resized.
    """
    # Path validation
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Path does not exist: {source_dir}")
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Not a directory: {source_dir}")

    # Create project directory
    project_root = os.path.join(output_dir, f"{project_name}Data")
    
    # Clear existing directory to prevent file conflicts
    if os.path.exists(project_root):
        shutil.rmtree(project_root)
    
    # Create the YOLO-style directory structure
    # This structure is common for YOLOv5, v7, v8, etc.
    dirs = {
        "images": os.path.join(project_root, "images"),
        "labels": os.path.join(project_root, "labels"),
    }
    for main_dir in ["images", "labels"]:
        for subset in ["train", "val", "test"]:
            os.makedirs(os.path.join(dirs[main_dir], subset), exist_ok=True)

    # Get all image files and shuffle them randomly
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(image_exts)]
    assert len(all_files) > 0, f"No image files found in: {source_dir}"
    random.seed(seed)
    random.shuffle(all_files)

    # Split the dataset based on the specified ratios
    total = len(all_files)
    train_num = int(ratios[0] * total)
    val_num = int(ratios[1] * total)
    test_num = total - train_num - val_num

    splits = {
        "train": all_files[:train_num],
        "val": all_files[train_num:train_num+val_num],
        "test": all_files[train_num+val_num:]
    }

    # Process and move/copy files to their respective new locations
    for subset, files in splits.items():
        for idx, filename in enumerate(files, start=1):
            src_path = os.path.join(source_dir, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Destination path for the new image file
            new_name = f"{subset}_{idx}{file_ext}"
            dest_image = os.path.join(dirs["images"], subset, new_name)
            
            # Resize image if normalize_size is specified
            if normalize_size:
                try:
                    with Image.open(src_path) as img:
                        img = img.resize(normalize_size)
                        img.save(dest_image)
                except Exception as e:
                    print(f"Warning: Failed to process {filename} (original file copied) - {str(e)}")
                    shutil.copy(src_path, dest_image)
            else:
                shutil.copy(src_path, dest_image)
            
            # Create a corresponding empty label file
            label_path = os.path.join(dirs["labels"], subset, f"{subset}_{idx}.txt")
            open(label_path, 'w').close()

    # Generate the YAML configuration file
    yaml_path = os.path.join(project_root, f"{project_name}Data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump({
            "path": os.path.abspath(project_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 0,  # Number of classes (to be filled by the user)
            "names": []  # List of class names (to be filled by the user)
        }, f, sort_keys=False)

    print(f"\nSuccessfully generated dataset structure at: {os.path.abspath(project_root)}")

class ImageProcessor(QWidget):
    """
    A PyQt5-based GUI application for processing and organizing images
    into a YOLO dataset structure.
    """
    def __init__(self):
        super().__init__()
        self.inputFolder = None
        self.outputFolder = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Image Processing Tool')
        self.setGeometry(100, 100, 400, 350)

        layout = QVBoxLayout()
        
        # UI elements for folder selection
        self.inputButton = QPushButton('Select Input Folder')
        self.inputButton.clicked.connect(self.selectInputFolder)
        layout.addWidget(self.inputButton)
        
        self.outputButton = QPushButton('Select Output Folder')
        self.outputButton.clicked.connect(self.selectOutputFolder)
        layout.addWidget(self.outputButton)
        
        # UI element for project name
        self.projectNameEdit = QLineEdit(self)
        self.projectNameEdit.setPlaceholderText('Project Name')
        layout.addWidget(self.projectNameEdit)
        
        # UI elements for setting dataset split ratios
        self.trainRatioSpin = QDoubleSpinBox(self)
        self.trainRatioSpin.setRange(0, 1)
        self.trainRatioSpin.setSingleStep(0.01)
        self.trainRatioSpin.setValue(0.7)
        self.trainRatioSpin.setPrefix('Train Ratio: ')
        layout.addWidget(self.trainRatioSpin)
        
        self.valRatioSpin = QDoubleSpinBox(self)
        self.valRatioSpin.setRange(0, 1)
        self.valRatioSpin.setSingleStep(0.01)
        self.valRatioSpin.setValue(0.15)
        self.valRatioSpin.setPrefix('Validation Ratio: ')
        layout.addWidget(self.valRatioSpin)
        
        self.testRatioSpin = QDoubleSpinBox(self)
        self.testRatioSpin.setRange(0, 1)
        self.testRatioSpin.setSingleStep(0.01)
        self.testRatioSpin.setValue(0.15)
        self.testRatioSpin.setPrefix('Test Ratio: ')
        layout.addWidget(self.testRatioSpin)
        
        # UI element for the processing button
        self.processButton = QPushButton('Process Images')
        self.processButton.clicked.connect(self.processImages)
        layout.addWidget(self.processButton)
        
        # UI element for the progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        layout.addWidget(self.progressBar)
        
        self.setLayout(layout)
        
    def selectInputFolder(self):
        """Slot to handle input folder selection."""
        self.inputFolder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        
    def selectOutputFolder(self):
        """Slot to handle output folder selection."""
        self.outputFolder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        
    def processImages(self):
        """Main method to trigger image processing based on user input."""
        source_dir = self.inputFolder
        output_dir = self.outputFolder
        project_name = self.projectNameEdit.text() or 'my_project'
        ratios = (
            self.trainRatioSpin.value(),
            self.valRatioSpin.value(),
            self.testRatioSpin.value()
        )
        
        # Call the core function to split and organize images
        split_and_organize_images(source_dir, output_dir, project_name, ratios, seed=42, normalize_size=None)
        print(f"Processing images: {source_dir} -> {output_dir}, Project Name: {project_name}, Ratios: {ratios}")

        # Simulate image processing for the progress bar
        image_list = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        total_images = len(image_list)
        
        # Initialize progress bar
        self.progressBar.setValue(0)
        
        # Loop to simulate the processing and update the progress bar
        for i, image in enumerate(image_list):
            # The actual file processing is handled by the `split_and_organize_images` function.
            # This loop is just for updating the GUI's progress bar.
            
            # Update the progress bar
            progress = int((i + 1) / total_images * 100)
            self.progressBar.setValue(progress)
            
            # Ensure the GUI remains responsive
            QApplication.processEvents()
        
        # Set progress bar to 100% upon completion
        self.progressBar.setValue(100)

if __name__ == '__main__':
    # Initialize the PyQt application
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    # Start the application event loop
    sys.exit(app.exec_())