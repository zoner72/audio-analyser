# gui/frames/model_info_frame.py
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QPushButton, 
    QFileDialog, QMessageBox, QGroupBox, QGridLayout
)
from utils.model_manager import ModelManager
import os

class ModelInfoFrame(QFrame):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Model Loading Section
        load_group = QGroupBox("Model Loading")
        load_layout = QVBoxLayout()
        load_group.setLayout(load_layout)
        
        self.model_label = QLabel("No model loaded")
        self.load_button = QPushButton("Load Model")
        self.save_button = QPushButton("Save Model")
        
        load_layout.addWidget(self.model_label)
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.save_button)
        
        # Model Info Section
        info_group = QGroupBox("Model Information")
        info_layout = QGridLayout()
        info_group.setLayout(info_layout)
        
        # Create labels for model information
        self.layer_count_label = QLabel("Layers: -")
        self.params_label = QLabel("Parameters: -")
        self.input_shape_label = QLabel("Input Shape: -")
        self.output_shape_label = QLabel("Output Shape: -")
        
        # Add labels to info layout
        info_layout.addWidget(QLabel("Model Architecture:"), 0, 0)
        info_layout.addWidget(self.layer_count_label, 0, 1)
        info_layout.addWidget(QLabel("Total Parameters:"), 1, 0)
        info_layout.addWidget(self.params_label, 1, 1)
        info_layout.addWidget(QLabel("Input Shape:"), 2, 0)
        info_layout.addWidget(self.input_shape_label, 2, 1)
        info_layout.addWidget(QLabel("Output Shape:"), 3, 0)
        info_layout.addWidget(self.output_shape_label, 3, 1)
        
        # Connect buttons to their functions
        self.load_button.clicked.connect(self.load_model)
        self.save_button.clicked.connect(self.save_model)
        
        # Initially disable save button until a model is loaded
        self.save_button.setEnabled(False)
        
        # Add groups to main layout
        layout.addWidget(load_group)
        layout.addWidget(info_group)
        layout.addStretch()

    def update_model_info(self, model):
        """Update the model information display"""
        if model is None:
            self.layer_count_label.setText("Layers: -")
            self.params_label.setText("Parameters: -")
            self.input_shape_label.setText("Input Shape: -")
            self.output_shape_label.setText("Output Shape: -")
            return

        try:
            # Get model information
            layer_count = len(model.layers)
            total_params = model.count_params()
            
            # Get input shape - handle different model types
            if hasattr(model, 'input_shape'):
                input_shape = str(model.input_shape)
            else:
                # Try to get from first layer
                first_layer = model.layers[0]
                if hasattr(first_layer, 'input_shape'):
                    input_shape = str(first_layer.input_shape)
                else:
                    input_shape = "Not available"
            
            # Get output shape - handle different model types
            if hasattr(model, 'output_shape'):
                output_shape = str(model.output_shape)
            else:
                # Try to get from last layer
                last_layer = model.layers[-1]
                if hasattr(last_layer, 'output_shape'):
                    output_shape = str(last_layer.output_shape)
                else:
                    output_shape = "Not available"
            
            # Update labels
            self.layer_count_label.setText(f"Layers: {layer_count}")
            self.params_label.setText(f"Parameters: {total_params:,}")
            self.input_shape_label.setText(f"Input Shape: {input_shape}")
            self.output_shape_label.setText(f"Output Shape: {output_shape}")
            
            # Add model summary to the display
            model.summary()
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error getting model info: {str(e)}")
            # Reset labels on error
            self.layer_count_label.setText("Layers: -")
            self.params_label.setText("Parameters: -")
            self.input_shape_label.setText("Input Shape: -")
            self.output_shape_label.setText("Output Shape: -")
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error getting model info: {str(e)}")

    def get_main_window(self):
        """Helper method to get reference to main window"""
        return self.window()

    def load_model(self):
        """Handle model loading"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Model",
                "",
                "Model Files (*.h5);;All Files (*)"
            )
            
            if file_name:
                model = ModelManager.load_model(file_name)
                # Store model in main window
                main_window = self.get_main_window()
                setattr(main_window, 'model', model)
                self.model_label.setText(f"Model loaded: {os.path.basename(file_name)}")
                self.save_button.setEnabled(True)
                
                # Update model information display
                self.update_model_info(model)
                
                QMessageBox.information(self, "Success", "Model loaded successfully")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.update_model_info(None)

    def save_model(self):
        """Handle model saving"""
        main_window = self.get_main_window()
        
        if not hasattr(main_window, 'model') or main_window.model is None:
            QMessageBox.warning(self, "Warning", "No model to save")
            return

        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model",
                "",
                "Model Files (*.h5);;All Files (*)"
            )
            
            if file_name:
                ModelManager.save_model(main_window.model, file_name)
                QMessageBox.information(self, "Success", "Model saved successfully")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
