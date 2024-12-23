# setup_project.py
import os
import sys

def create_directory_structure(base_path):
    # Define the directory structure
    directories = [
        '',
        'gui',
        'gui/tabs',
        'core',
        'utils',
    ]

    # Create directories
    for dir in directories:
        full_path = os.path.join(base_path, dir)
        os.makedirs(full_path, exist_ok=True)
        # Create __init__.py in each directory
        if dir:
            init_file = os.path.join(full_path, '__init__.py')
            with open(init_file, 'w') as f:
                pass

    # Define file contents (as shown in the previous response)
    files = {
        'main.py': '''import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
''',
        'gui/main_window.py': '''from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from .tabs.analysis_tab import AnalysisTab
from .tabs.training_tab import TrainingTab
from .tabs.results_tab import ResultsTab
from .tabs.model_info_tab import ModelInfoTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Classification System")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add tabs
        self.analysis_tab = AnalysisTab()
        self.training_tab = TrainingTab()
        self.results_tab = ResultsTab()
        self.model_info_tab = ModelInfoTab()
        
        self.tab_widget.addTab(self.analysis_tab, "Analysis & Config")
        self.tab_widget.addTab(self.training_tab, "Model Training")
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.model_info_tab, "Model Info")
''',
        'gui/tabs/analysis_tab.py': '''from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QPushButton, QSpinBox)
from core.analysis import AudioAnalyzer

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.analyzer = AudioAnalyzer()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(QLabel("Audio Directory:"))
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)
        
        # Configuration options
        config_layout = QVBoxLayout()
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1, 60)
        self.chunk_size.setValue(1)
        config_layout.addWidget(QLabel("Chunk Size (seconds):"))
        config_layout.addWidget(self.chunk_size)
        layout.addLayout(config_layout)
        
        # Analysis button
        analyze_btn = QPushButton("Start Analysis")
        analyze_btn.clicked.connect(self.start_analysis)
        layout.addWidget(analyze_btn)
        
    def browse_directory(self):
        # Implementation here
        pass
        
    def start_analysis(self):
        # Implementation here
        pass
''',
        'gui/tabs/training_tab.py': '''from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QPushButton, QSpinBox)
from core.training import ModelTrainer

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.trainer = ModelTrainer()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(QLabel("Training Data Directory:"))
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)
        
        # Training parameters
        param_layout = QVBoxLayout()
        self.n_estimators = QSpinBox()
        self.n_estimators.setRange(1, 1000)
        self.n_estimators.setValue(100)
        param_layout.addWidget(QLabel("Number of Estimators:"))
        param_layout.addWidget(self.n_estimators)
        layout.addLayout(param_layout)
        
        # Train button
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)
        layout.addWidget(train_btn)
        
    def browse_directory(self):
        # Implementation here
        pass
        
    def train_model(self):
        # Implementation here
        pass
''',
        'gui/tabs/results_tab.py': '''from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
    def update_results(self, text):
        self.results_text.setText(text)
''',
        'gui/tabs/model_info_tab.py': '''from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class ModelInfoTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
    def update_info(self, text):
        self.info_text.setText(text)
''',
        'core/analysis.py': '''import librosa
import numpy as np
from datetime import datetime
from .features import FeatureExtractor

class AudioAnalyzer:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        
    def analyze_file(self, file_path, chunk_size, model):
        # Implementation here
        pass
''',
        'core/training.py': '''from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from .features import FeatureExtractor

class ModelTrainer:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        
    def train(self, training_dir, n_estimators):
        # Implementation here
        pass
''',
        'core/features.py': '''import librosa
import numpy as np

class FeatureExtractor:
    def extract_features(self, audio_data, sr):
        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        
        # Calculate statistics
        features = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'centroid_mean': np.mean(spectral_centroid),
            'rolloff_mean': np.mean(spectral_rolloff)
        }
        
        return features
''',
        'utils/settings.py': '''class Settings:
    def __init__(self):
        self.chunk_size = 1
        self.n_estimators = 100
        self.model_path = "models/audio_classifier.joblib"
        
    def save_settings(self):
        # Implementation here
        pass
        
    def load_settings(self):
        # Implementation here
        pass
''',
        'requirements.txt': '''PyQt5
numpy
pandas
scikit-learn
librosa
soundfile
joblib
'''
    }

    # Create all files
    for file_path, content in files.items():
        full_path = os.path.join(base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)

    print(f"Project structure created at {base_path}")

if __name__ == "__main__":
    base_path = r"C:\Users\karsd\Documents\Python\audio_analyser"
    create_directory_structure(base_path)
