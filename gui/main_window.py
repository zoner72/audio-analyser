# gui/main_window.py
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QPushButton
from .frames.analysis_frame import AnalysisFrame
from .frames.training_frame import TrainingFrame
from .frames.results_frame import ResultsFrame
from .frames.model_info_frame import ModelInfoFrame

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Classification System")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create navigation panel
        nav_panel = QFrame()
        nav_layout = QVBoxLayout(nav_panel)
        nav_panel.setMaximumWidth(200)
        
        # Create content frame
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        
        # Create frames
        self.analysis_frame = AnalysisFrame()
        self.training_frame = TrainingFrame()
        self.results_frame = ResultsFrame()
        self.model_info_frame = ModelInfoFrame()
        
        # Create navigation buttons
        self.analysis_btn = QPushButton("Analysis & Config")
        self.training_btn = QPushButton("Model Training")
        self.results_btn = QPushButton("Results")
        self.model_info_btn = QPushButton("Model Info")
        
        # Add buttons to navigation panel
        nav_layout.addWidget(self.analysis_btn)
        nav_layout.addWidget(self.training_btn)
        nav_layout.addWidget(self.results_btn)
        nav_layout.addWidget(self.model_info_btn)
        nav_layout.addStretch()
        
        # Connect button signals
        self.analysis_btn.clicked.connect(lambda: self.show_frame(self.analysis_frame))
        self.training_btn.clicked.connect(lambda: self.show_frame(self.training_frame))
        self.results_btn.clicked.connect(lambda: self.show_frame(self.results_frame))
        self.model_info_btn.clicked.connect(lambda: self.show_frame(self.model_info_frame))
        
        # Add all frames to content layout
        self.content_layout.addWidget(self.analysis_frame)
        self.content_layout.addWidget(self.training_frame)
        self.content_layout.addWidget(self.results_frame)
        self.content_layout.addWidget(self.model_info_frame)
        
        # Add navigation and content to main layout
        main_layout.addWidget(nav_panel)
        main_layout.addWidget(self.content_frame)
        
        # Show initial frame
        self.show_frame(self.analysis_frame)
    
    def show_frame(self, frame):
        # Hide all frames
        self.analysis_frame.hide()
        self.training_frame.hide()
        self.results_frame.hide()
        self.model_info_frame.hide()
        
        # Show selected frame
        frame.show()
        
        # Update button styles
        self.analysis_btn.setStyleSheet("")
        self.training_btn.setStyleSheet("")
        self.results_btn.setStyleSheet("")
        self.model_info_btn.setStyleSheet("")
        
        # Highlight active button
        if frame == self.analysis_frame:
            self.analysis_btn.setStyleSheet("background-color: #e0e0e0;")
        elif frame == self.training_frame:
            self.training_btn.setStyleSheet("background-color: #e0e0e0;")
        elif frame == self.results_frame:
            self.results_btn.setStyleSheet("background-color: #e0e0e0;")
        elif frame == self.model_info_frame:
            self.model_info_btn.setStyleSheet("background-color: #e0e0e0;")
