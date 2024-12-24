# utils/audio_processor.py
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate, frame_size, hop_length):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.target_length = 50
        self.debug_plots = True  # Set to False to disable debug plots

    def process_audio(self, audio_file):
        """Process a single audio file to match model input requirements"""
        try:
            logger.debug(f"Processing file: {audio_file}")
            logger.debug(f"Parameters: sr={self.sample_rate}, frame_size={self.frame_size}, hop_length={self.hop_length}")
            
            # Validate audio file
            self.validate_audio_file(audio_file)
            
            # Load audio file
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            logger.debug(f"Loaded audio shape: {audio.shape}")
            
            # Plot waveform if debug enabled
            if self.debug_plots:
                self._plot_debug_waveform(audio, sr, audio_file)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.frame_size,
                hop_length=self.hop_length,
                n_mels=self.target_length
            )
            logger.debug(f"Mel spectrogram shape: {mel_spec.shape}")
            
            # Plot mel spectrogram if debug enabled
            if self.debug_plots:
                self._plot_debug_mel_spectrogram(mel_spec, audio_file)
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to get (time_steps, features)
            mel_spec_db = mel_spec_db.T
            logger.debug(f"Transposed mel_spec shape: {mel_spec_db.shape}")
            
            # Ensure we have exactly 50 time steps
            if mel_spec_db.shape[0] > self.target_length:
                mel_spec_db = mel_spec_db[:self.target_length, :]
            elif mel_spec_db.shape[0] < self.target_length:
                pad_length = self.target_length - mel_spec_db.shape[0]
                mel_spec_db = np.pad(
                    mel_spec_db,
                    ((0, pad_length), (0, 0)),
                    mode='constant'
                )
            
            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(mel_spec_db)
            logger.debug(f"Scaled features shape: {features_scaled.shape}")
            
            # Ensure final shape is (batch_size, 50)
            if features_scaled.shape[1] != 1:
                features_scaled = np.mean(features_scaled, axis=1)
            
            features_scaled = features_scaled.reshape(1, self.target_length)
            logger.debug(f"Final features shape: {features_scaled.shape}")
            
            return features_scaled
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            raise

    def validate_audio_file(self, audio_file):
        """Validate if the audio file can be processed"""
        try:
            duration = librosa.get_duration(path=audio_file)
            logger.debug(f"Audio duration: {duration}s")
            
            if duration < 0.1:
                raise ValueError("Audio file too short")
                
            # Try to load a small portion to verify format
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, duration=1.0)
            return True
        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            raise ValueError(f"Invalid audio file: {str(e)}")

    def _plot_debug_waveform(self, audio, sr, filename):
        """Create debug plot of waveform"""
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
        plt.title(f'Waveform: {filename}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig(f'debug_waveform_{filename}.png')
        plt.close()

    def _plot_debug_mel_spectrogram(self, mel_spec, filename):
        """Create debug plot of mel spectrogram"""
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(mel_spec, ref=np.max),
            y_axis='mel',
            x_axis='time'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel spectrogram: {filename}')
        plt.savefig(f'debug_mel_{filename}.png')
        plt.close()
