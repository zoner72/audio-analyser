# File: core/analysis.py
import librosa
import numpy as np
import soundfile as sf
import os
import shutil
import pandas as pd
from datetime import datetime

class AudioAnalyzer:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 10  # chunk size in seconds
        self.threshold = 0.1  # detection threshold
        self.base_directory = None  # Will be set when analyzing files
        self.categories = [
            'fauna',
            'mobile_equipment',
            'fixed_plant',
            'unknown'
        ]
        
        # Create necessary directories
        os.makedirs('temp', exist_ok=True)
        os.makedirs('training_data', exist_ok=True)
        for category in self.categories:
            os.makedirs(os.path.join('training_data', category), exist_ok=True)

    def load_audio(self, file_path):
        """Load audio file with proper error handling"""
        try:
            print(f"Attempting to load: {file_path}")  # Debug print
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            print(f"Successfully loaded audio file: {file_path}, duration: {len(audio)/sr:.2f}s")
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            raise

    def analyze_file(self, file_path):
        """Analyze an audio file and detect segments of interest"""
        try:
            # Set base directory for relative path calculations
            if self.base_directory is None:
                self.base_directory = os.path.dirname(file_path)

            # Load the audio file
            audio, sr = self.load_audio(file_path)
            
            # Calculate frame size and hop length
            frame_size = int(sr * self.duration)
            hop_length = frame_size // 2  # 50% overlap
            
            # Split audio into chunks
            segments = []
            for i in range(0, len(audio), hop_length):
                chunk = audio[i:i + frame_size]
                
                # Skip if chunk is too short
                if len(chunk) < frame_size:
                    continue
                
                # Calculate features
                rms = librosa.feature.rms(y=chunk)[0]
                zero_crossings = librosa.feature.zero_crossing_rate(chunk)[0]
                spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
                
                # Simple detection based on RMS energy
                if np.mean(rms) > self.threshold:
                    # Create temporary file for the segment
                    segment_filename = f"segment_{len(segments)}_{os.path.basename(file_path)}"
                    segment_path = os.path.join('temp', segment_filename)
                    
                    # Save segment to file
                    sf.write(segment_path, chunk, sr)
                    
                    # Calculate segment times
                    start_time = i / sr
                    end_time = min((i + frame_size) / sr, len(audio) / sr)
                    
                    # Store segment information
                    segment_info = {
                        'original_file': file_path,
                        'segment_file': segment_path,
                        'start_time': start_time,
                        'end_time': end_time,
                        'datetime': datetime.now(),
                        'features': {
                            'rms_mean': float(np.mean(rms)),
                            'zero_crossings_mean': float(np.mean(zero_crossings)),
                            'spectral_centroid_mean': float(np.mean(spectral_centroid))
                        }
                    }
                    segments.append(segment_info)
            
            return segments

        except Exception as e:
            print(f"Error analyzing file {file_path}: {str(e)}")
            raise

    def update_training_dataset(self, segment_data, classification):
        """Update the training dataset with new classified segments"""
        try:
            # Create directory for classified samples if it doesn't exist
            class_dir = os.path.join('training_data', classification)
            os.makedirs(class_dir, exist_ok=True)
            
            # Get relative path structure from original file
            original_rel_path = os.path.relpath(
                os.path.dirname(segment_data['original_file']), 
                start=self.base_directory
            )
            
            # Create subdirectory structure in training data
            target_dir = os.path.join(class_dir, original_rel_path)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the audio segment to the appropriate class directory
            segment_filename = os.path.basename(segment_data['segment_file'])
            destination = os.path.join(target_dir, segment_filename)
            shutil.copy2(segment_data['segment_file'], destination)
            
            # Update the metadata CSV
            metadata_file = os.path.join('training_data', 'metadata.csv')
            new_row = {
                'filename': os.path.relpath(destination, 'training_data'),
                'target': classification,
                'datetime': segment_data['datetime'],
                'original_file': segment_data['original_file'],
                'original_directory': original_rel_path,
                'start_time': segment_data['start_time'],
                'end_time': segment_data['end_time'],
                'rms_mean': segment_data['features']['rms_mean'],
                'zero_crossings_mean': segment_data['features']['zero_crossings_mean'],
                'spectral_centroid_mean': segment_data['features']['spectral_centroid_mean']
            }
            
            if os.path.exists(metadata_file):
                metadata_df = pd.read_csv(metadata_file)
            else:
                metadata_df = pd.DataFrame(columns=[
                    'filename', 'target', 'datetime', 'original_file',
                    'original_directory', 'start_time', 'end_time',
                    'rms_mean', 'zero_crossings_mean', 'spectral_centroid_mean'
                ])
            
            metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], 
                                  ignore_index=True)
            metadata_df.to_csv(metadata_file, index=False)
            
            return True
        except Exception as e:
            print(f"Error updating training dataset: {str(e)}")
            return False

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir('temp'):
                file_path = os.path.join('temp', file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
        except Exception as e:
            print(f"Error cleaning up temp files: {str(e)}")
