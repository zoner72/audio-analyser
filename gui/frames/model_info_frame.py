# model_info_frame.py
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QComboBox, QSpinBox, QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt


class ModelInfoFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Model information section
        info_section = QFrame()
        info_layout = QVBoxLayout(info_section)
        
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setText("Model Information will be displayed here...")
        
        info_layout.addWidget(self.model_info)
        
        # Model operations section
        operations_section = QFrame()
        operations_layout = QHBoxLayout(operations_section)
        
        self.load_button = QPushButton("Load Model")
        self.save_button = QPushButton("Save Model")
        
        operations_layout.addWidget(self.load_button)
        operations_layout.addWidget(self.save_button)
        
        # Add all sections to main layout
        layout.addWidget(info_section)
        layout.addWidget(operations_section)
