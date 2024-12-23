# analysis_frame.py
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QFileDialog, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt

class AnalysisFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # File selection section
        file_section = QFrame()
        file_layout = QHBoxLayout(file_section)
        
        self.file_label = QLabel("Select Audio File:")
        self.file_path = QLabel("No file selected")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.browse_button)
        
        # Analysis configuration section
        config_section = QFrame()
        config_layout = QVBoxLayout(config_section)
        
        # Sample rate selection
        sample_rate_layout = QHBoxLayout()
        self.sample_rate_label = QLabel("Sample Rate:")
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(['22050', '44100', '48000'])
        sample_rate_layout.addWidget(self.sample_rate_label)
        sample_rate_layout.addWidget(self.sample_rate_combo)
        
        # Frame size selection
        frame_size_layout = QHBoxLayout()
        self.frame_size_label = QLabel("Frame Size:")
        self.frame_size_spin = QSpinBox()
        self.frame_size_spin.setRange(256, 4096)
        self.frame_size_spin.setValue(2048)
        frame_size_layout.addWidget(self.frame_size_label)
        frame_size_layout.addWidget(self.frame_size_spin)
        
        config_layout.addLayout(sample_rate_layout)
        config_layout.addLayout(frame_size_layout)
        
        # Analysis button
        self.analyze_button = QPushButton("Analyze Audio")
        self.analyze_button.clicked.connect(self.analyze_audio)
        
        # Add all sections to main layout
        layout.addWidget(file_section)
        layout.addWidget(config_section)
        layout.addWidget(self.analyze_button)
        layout.addStretch()

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3)"
        )
        if file_name:
            self.file_path.setText(file_name)

    def analyze_audio(self):
        # Implement your audio analysis logic here
        pass
